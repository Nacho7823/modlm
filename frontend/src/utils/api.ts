import type { Chat, History, LLMConfig, MCPConfig, MCPTool, MCPToolResult } from "../types";

const API_BASE = "/api";

export interface ApiError {
  error: string;
  detail?: string;
  details?: Record<string, unknown>;
}

export async function sendMessage(
  message: string,
  chatId?: string,
): Promise<{ chat_id: string; response: string }> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, chat_id: chatId }),
  });

  if (!response.ok) {
    const data = (await response.json()) as ApiError;
    throw new Error(data.detail || data.error || "Failed to send message");
  }

  return response.json();
}

export async function getHistory(): Promise<History> {
  const response = await fetch(`${API_BASE}/chat/history`);

  if (!response.ok) {
    throw new Error("Failed to get history");
  }

  return response.json();
}

export async function getChat(chatId: string): Promise<Chat> {
  const response = await fetch(`${API_BASE}/chat/${chatId}`);

  if (!response.ok) {
    throw new Error("Chat not found");
  }

  return response.json();
}

export async function createChat(title?: string): Promise<Chat> {
  const response = await fetch(`${API_BASE}/chat/new`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });

  if (!response.ok) {
    throw new Error("Failed to create chat");
  }

  return response.json();
}

export async function deleteChat(chatId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/${chatId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error("Failed to delete chat");
  }
}

export async function clearHistory(): Promise<void> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error("Failed to clear history");
  }
}

export async function getConfig(): Promise<LLMConfig> {
  const response = await fetch(`${API_BASE}/config`);

  if (!response.ok) {
    throw new Error("Failed to get config");
  }

  return response.json();
}

export async function updateConfig(
  config: Record<string, unknown>,
): Promise<LLMConfig> {
  const response = await fetch(`${API_BASE}/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });

  if (!response.ok) {
    throw new Error("Failed to update config");
  }

  return response.json();
}

export async function fetchModels(provider: string): Promise<string[]> {
  const response = await fetch(`${API_BASE}/config/models?provider=${provider}`);

  if (!response.ok) {
    throw new Error("Failed to fetch models");
  }

  return response.json();
}

export async function getMCPConfig(): Promise<MCPConfig> {
  const response = await fetch(`${API_BASE}/mcp/config`);

  if (!response.ok) {
    throw new Error("Failed to get MCP config");
  }

  return response.json();
}

export async function updateMCPConfig(
  servers: MCPConfig["servers"],
): Promise<MCPConfig> {
  const response = await fetch(`${API_BASE}/mcp/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(servers),
  });

  if (!response.ok) {
    throw new Error("Failed to update MCP config");
  }

  return response.json();
}

export async function listMCPTools(serverName: string): Promise<MCPTool[]> {
  const response = await fetch(`${API_BASE}/mcp/${serverName}/tools`);

  if (!response.ok) {
    throw new Error("Failed to list MCP tools");
  }

  return response.json();
}

export async function executeMCPTool(
  serverName: string,
  toolName: string,
  arguments_: Record<string, unknown> = {},
): Promise<MCPToolResult> {
  const response = await fetch(`${API_BASE}/mcp/${serverName}/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool_name: toolName, arguments: arguments_ }),
  });

  if (!response.ok) {
    throw new Error("Failed to execute MCP tool");
  }

  return response.json();
}

export interface TestConnectionResult {
  status: "connected" | "error";
  tools: string[];
  tool_count: number;
  error?: string;
}

export async function testMCPConnection(
  serverName: string,
): Promise<TestConnectionResult> {
  const response = await fetch(`${API_BASE}/mcp/${serverName}/test`, {
    method: "POST",
  });

  if (!response.ok) {
    return {
      status: "error",
      tools: [],
      tool_count: 0,
      error: "Failed to test connection",
    };
  }

  return response.json();
}
