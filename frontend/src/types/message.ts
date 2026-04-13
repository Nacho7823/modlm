export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
  tool_calls: ToolCall[] | null;
}

export interface ToolCall {
  tool_name: string;
  arguments: Record<string, unknown>;
  result: string | null;
}

export interface Tool {
  name: string;
  description: string;
  function: string;
}