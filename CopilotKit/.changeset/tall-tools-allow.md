---
"@copilotkit/runtime": patch
---

- fix(runtime): filter orphaned tool messages in OpenAIAdapter

Prevents 'messages with role tool must be a response to a preceding message with tool_calls' error by filtering out any tool messages without matching tool calls before sending to OpenAI.
- retrigger CI
