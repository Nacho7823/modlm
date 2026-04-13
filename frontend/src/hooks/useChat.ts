import { useCallback, useState } from "react";
import type { Chat, History } from "../types";
import * as api from "../utils/api";

interface UseChatState {
  activeChat: Chat | null;
  history: History;
  loading: boolean;
  error: string | null;
}

interface UseChatReturn extends UseChatState {
  sendMessage: (message: string) => Promise<void>;
  loadHistory: () => Promise<void>;
  createNewChat: () => Promise<void>;
  deleteChat: (chatId: string) => Promise<void>;
  clearHistory: () => Promise<void>;
  selectChat: (chatId: string) => Promise<void>;
}

export function useChat(): UseChatReturn {
  const [state, setState] = useState<UseChatState>({
    activeChat: null,
    history: { chats: [], active_chat_id: null },
    loading: false,
    error: null,
  });

  const sendMessage = useCallback(async (message: string) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const { chat_id } = await api.sendMessage(
        message,
        state.activeChat?.id,
      );

      const updatedChat = await api.getChat(chat_id);
      setState((prev) => ({
        ...prev,
        activeChat: updatedChat,
        loading: false,
      }));

      await api.getHistory().then((history) => {
        setState((prev) => ({ ...prev, history }));
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to send message";
      setState((prev) => ({ ...prev, loading: false, error: errorMessage }));
    }
  }, [state.activeChat?.id]);

  const loadHistory = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const history = await api.getHistory();
      const activeChat = history.active_chat_id
        ? await api.getChat(history.active_chat_id)
        : null;
      setState({ activeChat, history, loading: false, error: null });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to load history";
      setState((prev) => ({ ...prev, loading: false, error: errorMessage }));
    }
  }, []);

  const createNewChat = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const chat = await api.createChat();
      setState((prev) => ({
        ...prev,
        activeChat: chat,
        loading: false,
      }));
      await loadHistory();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to create chat";
      setState((prev) => ({ ...prev, loading: false, error: errorMessage }));
    }
  }, [loadHistory]);

  const deleteChat = useCallback(async (chatId: string) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      await api.deleteChat(chatId);
      await loadHistory();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to delete chat";
      setState((prev) => ({ ...prev, loading: false, error: errorMessage }));
    }
  }, [loadHistory]);

  const clearHistory = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      await api.clearHistory();
      setState({
        activeChat: null,
        history: { chats: [], active_chat_id: null },
        loading: false,
        error: null,
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to clear history";
      setState((prev) => ({ ...prev, loading: false, error: errorMessage }));
    }
  }, []);

  const selectChat = useCallback(async (chatId: string) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const chat = await api.getChat(chatId);
      setState((prev) => ({
        ...prev,
        activeChat: chat,
        history: { ...prev.history, active_chat_id: chatId },
        loading: false,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to select chat";
      setState((prev) => ({ ...prev, loading: false, error: errorMessage }));
    }
  }, []);

  return {
    ...state,
    sendMessage,
    loadHistory,
    createNewChat,
    deleteChat,
    clearHistory,
    selectChat,
  };
}