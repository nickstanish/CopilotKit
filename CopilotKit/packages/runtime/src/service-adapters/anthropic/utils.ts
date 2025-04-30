import {
  ActionExecutionMessage,
  Message,
  ResultMessage,
  TextMessage,
} from "../../graphql/types/converted";
import { ActionInput } from "../../graphql/inputs/action.input";
import { Anthropic } from "@anthropic-ai/sdk";

export function limitMessagesToTokenCount(
  messages: any[],
  tools: any[],
  model: string,
  maxTokens?: number,
): any[] {
  maxTokens ||= MAX_TOKENS;

  const result: any[] = [];
  const toolsNumTokens = countToolsTokens(model, tools);
  if (toolsNumTokens > maxTokens) {
    throw new Error(`Too many tokens in function definitions: ${toolsNumTokens} > ${maxTokens}`);
  }
  maxTokens -= toolsNumTokens;

  for (const message of messages) {
    if (message.role === "system") {
      const numTokens = countMessageTokens(model, message);
      maxTokens -= numTokens;

      if (maxTokens < 0) {
        throw new Error("Not enough tokens for system message.");
      }
    }
  }

  let cutoff: boolean = false;

  const reversedMessages = [...messages].reverse();
  for (const message of reversedMessages) {
    if (message.role === "system") {
      result.unshift(message);
      continue;
    } else if (cutoff) {
      continue;
    }
    let numTokens = countMessageTokens(model, message);
    if (maxTokens < numTokens) {
      cutoff = true;
      continue;
    }
    result.unshift(message);
    maxTokens -= numTokens;
  }

  return result;
}

const MAX_TOKENS = 128000;

function countToolsTokens(model: string, tools: any[]): number {
  if (tools.length === 0) {
    return 0;
  }
  const json = JSON.stringify(tools);
  return countTokens(model, json);
}

function countMessageTokens(model: string, message: any): number {
  return countTokens(model, JSON.stringify(message.content) || "");
}

function countTokens(model: string, text: string): number {
  return text.length / 3;
}

export function convertActionInputToAnthropicTool(action: ActionInput): Anthropic.Messages.Tool {
  return {
    name: action.name,
    description: action.description,
    input_schema: JSON.parse(action.jsonSchema),
  };
}

export function convertMessageToAnthropicMessage(
  message: Message,
): Anthropic.Messages.MessageParam {
  if (message.isTextMessage()) {
    if (message.role === "system") {
      return {
        role: "assistant",
        content: [
          { type: "text", text: "THE FOLLOWING MESSAGE IS A SYSTEM MESSAGE: " + message.content },
        ],
      };
    } else {
      return {
        role: message.role === "user" ? "user" : "assistant",
        content: [{ type: "text", text: message.content }],
      };
    }
  } else if (message.isImageMessage()) {
    let mediaType: Anthropic.Messages.ImageBlockParam.Source["media_type"];
    switch (message.format) {
      case "jpeg":
        mediaType = "image/jpeg";
        break;
      case "png":
        mediaType = "image/png";
        break;
      case "webp":
        mediaType = "image/webp";
        break;
      case "gif":
        mediaType = "image/gif";
        break;
      default:
        throw new Error(`Unsupported image format: ${message.format}`);
    }

    return {
      role: "user",
      content: [
        {
          type: "image",
          source: {
            type: "base64",
            media_type: mediaType,
            data: message.bytes,
          },
        },
      ],
    };
  } else if (message.isActionExecutionMessage()) {
    return {
      role: "assistant",
      content: [
        {
          id: message.id,
          type: "tool_use",
          input: message.arguments,
          name: message.name,
        },
      ],
    };
  } else if (message.isResultMessage()) {
    return {
      role: "user",
      content: [
        {
          type: "tool_result",
          content: message.result,
          tool_use_id: message.actionExecutionId,
        },
      ],
    };
  }
}

export function groupAnthropicMessagesByRole(
  messageParams: Anthropic.Messages.MessageParam[],
): Anthropic.Messages.MessageParam[] {
  // First, handle duplicate tool results by keeping only one result per tool_use_id
  const toolResultsMap = new Map<string, any[]>();

  // Collect all tool results by their tool_use_id
  messageParams.forEach((message) => {
    if (message.role === "user" && Array.isArray(message.content)) {
      message.content.forEach((contentItem) => {
        if (contentItem.type === "tool_result" && contentItem.tool_use_id) {
          if (!toolResultsMap.has(contentItem.tool_use_id)) {
            toolResultsMap.set(contentItem.tool_use_id, []);
          }
          toolResultsMap.get(contentItem.tool_use_id).push({
            message,
            contentItem,
            contentLength: contentItem.content ? JSON.stringify(contentItem.content).length : 0,
          });
        }
      });
    }
  });

  // For each tool_use_id with multiple results, keep only the one with the most content
  const duplicateResults = Array.from(toolResultsMap.entries()).filter(
    ([_, results]) => results.length > 1,
  );

  if (duplicateResults.length > 0) {
    console.log(`Found ${duplicateResults.length} tool_use_ids with duplicate results`);

    // Create a map of messages to skip
    const skipMessages = new Set<Anthropic.Messages.MessageParam>();

    duplicateResults.forEach(([toolUseId, results]) => {
      // Sort by content length (descending)
      results.sort((a, b) => b.contentLength - a.contentLength);

      // Skip all but the first (most detailed) result
      for (let i = 1; i < results.length; i++) {
        skipMessages.add(results[i].message);
      }

      console.log(`Selected 1 response out of ${results.length} for tool_use_id ${toolUseId}`);
    });

    // Filter out messages to skip
    messageParams = messageParams.filter((msg) => !skipMessages.has(msg));
  }

  // Now proceed with normal grouping
  return messageParams.reduce((acc, message) => {
    const lastGroup = acc[acc.length - 1];

    if (lastGroup && lastGroup.role === message.role) {
      lastGroup.content = lastGroup.content.concat(message.content as any);
    } else {
      acc.push({
        role: message.role,
        content: [...(message.content as any)],
      });
    }

    return acc;
  }, [] as Anthropic.Messages.MessageParam[]);
}
