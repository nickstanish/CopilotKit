/**
 * Copilot Runtime adapter for OpenAI.
 *
 * ## Example
 *
 * ```ts
 * import { CopilotRuntime, OpenAIAdapter } from "@copilotkit/runtime";
 * import OpenAI from "openai";
 *
 * const copilotKit = new CopilotRuntime();
 *
 * const openai = new OpenAI({
 *   organization: "<your-organization-id>", // optional
 *   apiKey: "<your-api-key>",
 * });
 *
 * return new OpenAIAdapter({ openai });
 * ```
 *
 * ## Example with Azure OpenAI
 *
 * ```ts
 * import { CopilotRuntime, OpenAIAdapter } from "@copilotkit/runtime";
 * import OpenAI from "openai";
 *
 * // The name of your Azure OpenAI Instance.
 * // https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
 * const instance = "<your instance name>";
 *
 * // Corresponds to your Model deployment within your OpenAI resource, e.g. my-gpt35-16k-deployment
 * // Navigate to the Azure OpenAI Studio to deploy a model.
 * const model = "<your model>";
 *
 * const apiKey = process.env["AZURE_OPENAI_API_KEY"];
 * if (!apiKey) {
 *   throw new Error("The AZURE_OPENAI_API_KEY environment variable is missing or empty.");
 * }
 *
 * const copilotKit = new CopilotRuntime();
 *
 * const openai = new OpenAI({
 *   apiKey,
 *   baseURL: `https://${instance}.openai.azure.com/openai/deployments/${model}`,
 *   defaultQuery: { "api-version": "2024-04-01-preview" },
 *   defaultHeaders: { "api-key": apiKey },
 * });
 *
 * return new OpenAIAdapter({ openai });
 * ```
 */
import OpenAI from "openai";
import {
  CopilotServiceAdapter,
  CopilotRuntimeChatCompletionRequest,
  CopilotRuntimeChatCompletionResponse,
} from "../service-adapter";
import {
  convertActionInputToOpenAITool,
  convertMessageToOpenAIMessage,
  limitMessagesToTokenCount,
} from "./utils";
import { randomUUID } from "@copilotkit/shared";

const DEFAULT_MODEL = "gpt-4o";

export interface OpenAIAdapterParams {
  /**
   * An optional OpenAI instance to use.  If not provided, a new instance will be
   * created.
   */
  openai?: OpenAI;

  /**
   * The model to use.
   */
  model?: string;

  /**
   * Whether to disable parallel tool calls.
   * You can disable parallel tool calls to force the model to execute tool calls sequentially.
   * This is useful if you want to execute tool calls in a specific order so that the state changes
   * introduced by one tool call are visible to the next tool call. (i.e. new actions or readables)
   *
   * @default false
   */
  disableParallelToolCalls?: boolean;
}

export class OpenAIAdapter implements CopilotServiceAdapter {
  private model: string = DEFAULT_MODEL;

  private disableParallelToolCalls: boolean = false;
  private _openai: OpenAI;
  public get openai(): OpenAI {
    return this._openai;
  }

  constructor(params?: OpenAIAdapterParams) {
    this._openai = params?.openai || new OpenAI({});
    if (params?.model) {
      this.model = params.model;
    }
    this.disableParallelToolCalls = params?.disableParallelToolCalls || false;
  }

  async process(
    request: CopilotRuntimeChatCompletionRequest,
  ): Promise<CopilotRuntimeChatCompletionResponse> {
    const {
      threadId: threadIdFromRequest,
      model = this.model,
      messages,
      actions,
      eventSource,
      forwardedParameters,
    } = request;
    const tools = actions.map(convertActionInputToOpenAITool);
    const threadId = threadIdFromRequest ?? randomUUID();

    let openaiMessages = messages.map((m) => convertMessageToOpenAIMessage(m));

    // Debug logging to help trace the issue
    // TODO: Remove debug logging once the fix is confirmed
    console.log(
      "BEFORE FILTERING - Message structure:",
      openaiMessages.map((msg, i) => ({
        index: i,
        role: msg.role,
        tool_call_id: msg.role === "tool" ? msg.tool_call_id : undefined,
        has_tool_calls: msg.role === "assistant" && !!msg.tool_calls,
        tool_call_ids:
          msg.role === "assistant" && msg.tool_calls
            ? msg.tool_calls.map((tc) => tc.id)
            : undefined,
      })),
    );

    // Enhanced filtering for message sequence validation
    // Two-pass algorithm to guarantee valid message ordering

    // First pass: Map tool calls and their positions
    const toolCallPositions = new Map();
    const toolResponseMap = new Map();

    // Track all assistant tool calls and their positions
    for (let i = 0; i < openaiMessages.length; i++) {
      const msg = openaiMessages[i];
      if (msg.role === "assistant" && msg.tool_calls) {
        for (const toolCall of msg.tool_calls) {
          if (toolCall.id) {
            toolCallPositions.set(toolCall.id, i);
          }
        }
      }
    }

    // Second pass: Collect all tool responses by ID (including duplicates)
    for (let i = 0; i < openaiMessages.length; i++) {
      const msg = openaiMessages[i];
      if (msg.role === "tool" && msg.tool_call_id) {
        if (!toolResponseMap.has(msg.tool_call_id)) {
          toolResponseMap.set(msg.tool_call_id, []);
        }
        // Store the message along with its position
        toolResponseMap.get(msg.tool_call_id).push({ msg, position: i });
      }
    }

    // Build the reordered message list
    const reorderedMessages = [];

    // Process non-tool messages first
    for (let i = 0; i < openaiMessages.length; i++) {
      const msg = openaiMessages[i];
      if (msg.role !== "tool") {
        reorderedMessages.push(msg);
      }
    }

    // Now process tool messages, selecting only one response per tool call
    for (const [toolCallId, responses] of Array.from(toolResponseMap.entries())) {
      // Skip if no matching tool call found
      if (!toolCallPositions.has(toolCallId)) {
        console.warn(`Skipping tool message with ID ${toolCallId} - no matching tool call found`);
        continue;
      }

      const toolCallPos = toolCallPositions.get(toolCallId);

      // Skip responses that appear before their corresponding tool call
      const validResponses = responses.filter((r) => r.position > toolCallPos);

      if (validResponses.length === 0) {
        console.warn(`No valid tool responses found for tool call ID ${toolCallId}`);
        continue;
      }

      // If we have multiple valid responses, select the one with content
      // Sort by content length as a basic heuristic (longer response might have more information)
      validResponses.sort((a, b) => {
        const aContentLength = a.msg.content ? a.msg.content.length : 0;
        const bContentLength = b.msg.content ? b.msg.content.length : 0;
        return bContentLength - aContentLength; // Descending order
      });

      // Add the best response
      reorderedMessages.push(validResponses[0].msg);
      console.log(
        `Selected 1 response out of ${validResponses.length} for tool call ID ${toolCallId}`,
      );
    }

    // Sort the messages to maintain the original order
    reorderedMessages.sort((a, b) => {
      const aIndex = openaiMessages.indexOf(a);
      const bIndex = openaiMessages.indexOf(b);
      return aIndex - bIndex;
    });

    // Use our reordered messages
    openaiMessages = reorderedMessages;

    // Final validation pass - make sure we don't have any remaining invalid sequences
    // This ensures that even after our filtering, the message sequence is valid
    try {
      // Check if any 'tool' message exists without a preceding matching tool call
      const finalToolCallIds = new Set();
      const invalidToolResponses = [];

      for (let i = 0; i < openaiMessages.length; i++) {
        const msg = openaiMessages[i];

        if (msg.role === "assistant" && msg.tool_calls) {
          for (const toolCall of msg.tool_calls) {
            if (toolCall.id) finalToolCallIds.add(toolCall.id);
          }
        } else if (msg.role === "tool") {
          // For each tool message, verify there is a preceding tool call with matching ID
          const toolCallId = msg.tool_call_id;

          // If this ID isn't in our set, or we haven't seen a tool call with this ID before this message,
          // then this is an invalid tool response
          let hasPrecedingToolCall = false;

          for (let j = 0; j < i; j++) {
            const prevMsg = openaiMessages[j];
            if (prevMsg.role === "assistant" && prevMsg.tool_calls) {
              if (prevMsg.tool_calls.some((tc) => tc.id === toolCallId)) {
                hasPrecedingToolCall = true;
                break;
              }
            }
          }

          if (!hasPrecedingToolCall) {
            invalidToolResponses.push(i);
          }
        }
      }

      // Remove any invalid tool responses we found in our final check
      if (invalidToolResponses.length > 0) {
        console.warn(
          `Final validation found ${invalidToolResponses.length} invalid tool responses, removing them`,
        );
        openaiMessages = openaiMessages.filter((_, i) => !invalidToolResponses.includes(i));
      }
    } catch (e) {
      console.error("Error during final message validation:", e);
    }

    // Debug logging after filtering
    // TODO: Remove debug logging once the fix is confirmed
    console.log(
      "AFTER FILTERING - Message structure:",
      openaiMessages.map((msg, i) => ({
        index: i,
        role: msg.role,
        tool_call_id: msg.role === "tool" ? msg.tool_call_id : undefined,
        has_tool_calls: msg.role === "assistant" && !!msg.tool_calls,
        tool_call_ids:
          msg.role === "assistant" && msg.tool_calls
            ? msg.tool_calls.map((tc) => tc.id)
            : undefined,
      })),
    );

    openaiMessages = limitMessagesToTokenCount(openaiMessages, tools, model);

    let toolChoice: any = forwardedParameters?.toolChoice;
    if (forwardedParameters?.toolChoice === "function") {
      toolChoice = {
        type: "function",
        function: { name: forwardedParameters.toolChoiceFunctionName },
      };
    }

    const stream = this.openai.beta.chat.completions.stream({
      model: model,
      stream: true,
      messages: openaiMessages,
      ...(tools.length > 0 && { tools }),
      ...(forwardedParameters?.maxTokens && { max_tokens: forwardedParameters.maxTokens }),
      ...(forwardedParameters?.stop && { stop: forwardedParameters.stop }),
      ...(toolChoice && { tool_choice: toolChoice }),
      ...(this.disableParallelToolCalls && { parallel_tool_calls: false }),
      ...(forwardedParameters?.temperature && { temperature: forwardedParameters.temperature }),
    });

    eventSource.stream(async (eventStream$) => {
      let mode: "function" | "message" | null = null;
      let currentMessageId: string;
      let currentToolCallId: string;
      for await (const chunk of stream) {
        if (chunk.choices.length === 0) {
          continue;
        }

        const toolCall = chunk.choices[0].delta.tool_calls?.[0];
        const content = chunk.choices[0].delta.content;

        // When switching from message to function or vice versa,
        // send the respective end event.
        // If toolCall?.id is defined, it means a new tool call starts.
        if (mode === "message" && toolCall?.id) {
          mode = null;
          eventStream$.sendTextMessageEnd({ messageId: currentMessageId });
        } else if (mode === "function" && (toolCall === undefined || toolCall?.id)) {
          mode = null;
          eventStream$.sendActionExecutionEnd({ actionExecutionId: currentToolCallId });
        }

        // If we send a new message type, send the appropriate start event.
        if (mode === null) {
          if (toolCall?.id) {
            mode = "function";
            currentToolCallId = toolCall!.id;
            eventStream$.sendActionExecutionStart({
              actionExecutionId: currentToolCallId,
              parentMessageId: chunk.id,
              actionName: toolCall!.function!.name,
            });
          } else if (content) {
            mode = "message";
            currentMessageId = chunk.id;
            eventStream$.sendTextMessageStart({ messageId: currentMessageId });
          }
        }

        // send the content events
        if (mode === "message" && content) {
          eventStream$.sendTextMessageContent({
            messageId: currentMessageId,
            content: content,
          });
        } else if (mode === "function" && toolCall?.function?.arguments) {
          eventStream$.sendActionExecutionArgs({
            actionExecutionId: currentToolCallId,
            args: toolCall.function.arguments,
          });
        }
      }

      // send the end events
      if (mode === "message") {
        eventStream$.sendTextMessageEnd({ messageId: currentMessageId });
      } else if (mode === "function") {
        eventStream$.sendActionExecutionEnd({ actionExecutionId: currentToolCallId });
      }

      eventStream$.complete();
    });

    return {
      threadId,
    };
  }
}
