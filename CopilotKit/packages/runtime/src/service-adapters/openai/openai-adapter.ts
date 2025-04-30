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
    const processedToolResponses = new Set();

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

    // Second pass: Build properly ordered message list
    const reorderedMessages = [];

    // Process messages in order
    for (let i = 0; i < openaiMessages.length; i++) {
      const msg = openaiMessages[i];

      if (msg.role === "tool") {
        const toolCallId = msg.tool_call_id;

        // Skip if we don't have this tool call ID or we've already processed a response for it
        if (!toolCallPositions.has(toolCallId) || processedToolResponses.has(toolCallId)) {
          console.warn(
            `Skipping tool message with ID ${toolCallId} - ${
              !toolCallPositions.has(toolCallId)
                ? "no matching tool call found"
                : "duplicate response"
            }`,
          );
          continue;
        }

        // Check if this tool response appears after its corresponding tool call
        const toolCallPos = toolCallPositions.get(toolCallId);
        if (toolCallPos >= i) {
          console.warn(
            `Filtering out of order tool message with ID ${toolCallId} - tool call at pos ${toolCallPos}, response at ${i}`,
          );
          continue;
        }

        // Valid tool response
        reorderedMessages.push(msg);
        processedToolResponses.add(toolCallId);
      } else {
        // Non-tool messages are always included
        reorderedMessages.push(msg);
      }
    }

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
