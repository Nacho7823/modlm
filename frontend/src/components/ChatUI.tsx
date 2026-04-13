import { useEffect, useRef, useState } from "react";
import { useChat } from "../hooks";
import { formatDate } from "../utils";

export function ChatUI() {
  const {
    activeChat,
    history,
    loading,
    error,
    sendMessage,
    createNewChat,
    selectChat,
    deleteChat,
  } = useChat();

  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!activeChat) {
      createNewChat();
    }
  }, [activeChat, createNewChat]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages]);

  const handleSend = () => {
    if (input.trim() && !loading) {
      sendMessage(input);
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const closeSidebar = () => setSidebarOpen(false);

  const handleChatSelect = (chatId: string) => {
    selectChat(chatId);
    closeSidebar();
  };

  return (
    <div className="chat-container">
      <div className="navbar">
        <button className="menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4 6H20M4 12H20M4 18H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        <h1 className="navbar-title">Chat</h1>
      </div>

      <div
        className={`sidebar-overlay ${sidebarOpen ? "visible" : ""}`}
        onClick={closeSidebar}
      />

      <div className={`chat-sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-header">
          <button className="sidebar-back" onClick={closeSidebar} aria-label="Close menu">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <h2>Chats</h2>
          <button onClick={createNewChat} disabled={loading}>
            + New
          </button>
        </div>
        <div className="chat-list">
          {history.chats.map((chat) => (
            <div
              key={chat.id}
              className={`chat-item ${activeChat?.id === chat.id ? "active" : ""}`}
              onClick={() => handleChatSelect(chat.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && handleChatSelect(chat.id)}
            >
              <div className="chat-item-content">
                <span className="chat-title">{chat.title}</span>
                <span className="chat-date">
                  {formatDate(chat.updated_at)}
                </span>
              </div>
                <button 
                  className="chat-delete" 
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteChat(chat.id);
                  }}
                  aria-label="Delete chat"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>
            </div>
          ))}
        </div>
      </div>

      <div className="chat-main">
        <div className="chat-messages">
          {activeChat?.messages.map((msg, i) => (
            <div key={i} className={`message message-${msg.role}`}>
              <div className="message-role">{msg.role}</div>
              <div className="message-content">{msg.content}</div>
            </div>
          ))}
          {loading && (
            <div className="message message-assistant">
              <div className="typing-indicator">Thinking</div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input">
          <input
            className="chat-input-text"
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            disabled={loading}
            aria-label="Message input"
          />
          <button 
            className="chat-input-button"
            onClick={handleSend}
            disabled={loading || !input.trim()}
            aria-label="Send message"
          >
            <svg width="52" height="52" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M2.01 21L23 12L2.01 3L2 10L17 12L2 14L2.01 21Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>

        {error && <div className="error-message" role="alert">{error}</div>}
      </div>
    </div>
  );
}