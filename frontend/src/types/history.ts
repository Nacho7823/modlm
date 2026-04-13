import { Chat } from "./chat";

export interface History {
  chats: Chat[];
  active_chat_id: string | null;
}