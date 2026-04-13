export interface LLMProviderConfig {
  provider: string;
  model: string;
  api_key: string;
  base_url: string | null;
  temperature: number;
  max_tokens: number | null;
}

export interface LLMConfig {
  providers: Record<string, LLMProviderConfig>;
  active_provider: string;
}