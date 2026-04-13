import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useChat } from "../hooks";
import { formatDate } from "../utils";

export function ChatUI() {
  const navigate = useNavigate();
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
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages]);

  const handleSend = () => {
    if (input.trim() && !loading) {
      const hasChatsWithMessages = history.chats.some(c => c.messages.length > 0);
      
      if (!activeChat && !hasChatsWithMessages) {
        createNewChat().then(() => {
          sendMessage(input);
        });
      } else {
        sendMessage(input);
      }
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
        <button className="new-chat-btn" onClick={createNewChat} disabled={loading} aria-label="New chat">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
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
        <div className="sidebar-actions">
          <button className="settings-btn" onClick={() => navigate("/settings")}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M12 15a3 3 0 100-6 3 3 0 000 6z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Settings
          </button>
        </div>
        <div className="chat-list">
          {history.chats
            .filter((chat) => chat.messages.length > 0)
            .map((chat) => (
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
          {(!activeChat || activeChat.messages.length === 0) ? (
            <div className="empty-chat">
              <p>Start a conversation by typing below.</p>
            </div>
          ) : (
            activeChat.messages.map((msg, i) => (
              <div key={i} className={`message message-${msg.role}`}>
                <div className="message-role">{msg.role}</div>
                <div className="message-content">{msg.content}</div>
              </div>
            ))
          )}
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