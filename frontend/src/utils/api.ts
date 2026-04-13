import type { Chat, History, LLMConfig } from "../types";

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
