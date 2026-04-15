export interface LLMProviderConfig {
  provider: string;
  model: string;
  api_key: string;
  base_url: string | null;
  temperature: number;
  max_tokens: number | null;
  system_prompt: string | null;
}

export interface LLMConfig {
  providers: Record<string, LLMProviderConfig>;
  active_provider: string;
}

export interface MCPServerConfig {
  name: string;
  type: "local" | "remote";
  command?: string[];
  url?: string;
  headers?: Record<string, string>;
  env?: Record<string, string>;
  timeout: number;
  enabled: boolean;
}

export interface MCPConfig {
  servers: MCPServerConfig[];
}

export interface MCPTool {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export interface MCPToolResult {
  content: Array<{ type: string; text?: string; data?: unknown }>;
  is_error: boolean;
}