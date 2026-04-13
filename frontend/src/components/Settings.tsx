import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTheme } from "../hooks";
import { getConfig, updateConfig } from "../utils/api";

interface FormData {
  provider: string;
  model: string;
  api_key: string;
  base_url: string;
  temperature: number;
  max_tokens: string;
  system_prompt: string;
}

export function Settings() {
  const navigate = useNavigate();
  const { theme, toggleTheme } = useTheme();
  
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [showApiKey, setShowApiKey] = useState(false);
  
  const [formData, setFormData] = useState<FormData>({
    provider: "openai",
    model: "gpt-3.5-turbo",
    api_key: "",
    base_url: "",
    temperature: 0.7,
    max_tokens: "2048",
    system_prompt: "",
  });

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const config = await getConfig();
      const activeProvider = config.providers[config.active_provider];
      
      if (activeProvider) {
        setFormData({
          provider: config.active_provider,
          model: activeProvider.model,
          api_key: activeProvider.api_key || "",
          base_url: activeProvider.base_url || "",
          temperature: activeProvider.temperature,
          max_tokens: activeProvider.max_tokens?.toString() || "2048",
          system_prompt: activeProvider.system_prompt || "",
        });
      }
    } catch (err) {
      setError("Failed to load configuration");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      await updateConfig({
        provider: formData.provider,
        model: formData.model,
        api_key: formData.api_key,
        base_url: formData.base_url || null,
        temperature: formData.temperature,
        max_tokens: formData.max_tokens ? parseInt(formData.max_tokens, 10) : null,
        system_prompt: formData.system_prompt || null,
      });
      setSuccess("Configuration saved successfully!");
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError("Failed to save configuration");
    } finally {
      setSaving(false);
    }
  };

  const handleChange = (field: keyof FormData, value: string | number) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  if (loading) {
    return (
      <div className="settings-container">
        <div className="settings-loading">Loading...</div>
      </div>
    );
  }

  return (
    <div className="settings-container">
      <div className="settings-header">
        <button className="back-btn" onClick={() => navigate("/")} aria-label="Back to chat">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M19 12H5M12 19l-7-7 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        <h1>Settings</h1>
      </div>

      <div className="settings-content">
        <section className="settings-section">
          <h2>Appearance</h2>
          <div className="setting-row">
            <label htmlFor="theme-toggle">Dark Mode</label>
            <button
              id="theme-toggle"
              className={`toggle-switch ${theme === "dark" ? "active" : ""}`}
              onClick={toggleTheme}
              role="switch"
              aria-checked={theme === "dark"}
            >
              <span className="toggle-thumb" />
            </button>
          </div>
        </section>

        <section className="settings-section">
          <h2>LLM Configuration</h2>
          
          <div className="form-group">
            <label htmlFor="provider">Provider</label>
            <select
              id="provider"
              value={formData.provider}
              onChange={(e) => handleChange("provider", e.target.value)}
              className="form-input"
            >
              <option value="openai">OpenAI</option>
              <option value="ollama">Ollama</option>
              <option value="lmstudio">LM Studio</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="api_key">API Key</label>
            <div className="input-with-toggle">
              <input
                id="api_key"
                type={showApiKey ? "text" : "password"}
                value={formData.api_key}
                onChange={(e) => handleChange("api_key", e.target.value)}
                className="form-input"
                placeholder="sk-..."
              />
              <button
                type="button"
                className="toggle-visibility"
                onClick={() => setShowApiKey(!showApiKey)}
                aria-label={showApiKey ? "Hide API key" : "Show API key"}
              >
                {showApiKey ? (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24M1 1l22 22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                )}
              </button>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="model">Model</label>
            <input
              id="model"
              type="text"
              value={formData.model}
              onChange={(e) => handleChange("model", e.target.value)}
              className="form-input"
              placeholder="gpt-4o, gpt-3.5-turbo, llama2, etc."
            />
          </div>

          <div className="form-group">
            <label htmlFor="base_url">Base URL (optional)</label>
            <input
              id="base_url"
              type="text"
              value={formData.base_url}
              onChange={(e) => handleChange("base_url", e.target.value)}
              className="form-input"
              placeholder="https://api.openai.com/v1"
            />
          </div>

          <div className="form-group">
            <label htmlFor="system_prompt">System Prompt</label>
            <textarea
              id="system_prompt"
              value={formData.system_prompt}
              onChange={(e) => handleChange("system_prompt", e.target.value)}
              className="form-input form-textarea"
              placeholder="You are a helpful assistant..."
              rows={4}
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="temperature">Temperature: {formData.temperature}</label>
              <input
                id="temperature"
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={formData.temperature}
                onChange={(e) => handleChange("temperature", parseFloat(e.target.value))}
                className="form-range"
              />
            </div>

            <div className="form-group">
              <label htmlFor="max_tokens">Max Tokens</label>
              <input
                id="max_tokens"
                type="number"
                value={formData.max_tokens}
                onChange={(e) => handleChange("max_tokens", e.target.value)}
                className="form-input"
                min="1"
                max="128000"
              />
            </div>
          </div>

          {error && <div className="settings-error">{error}</div>}
          {success && <div className="settings-success">{success}</div>}

          <button
            className="save-btn"
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? "Saving..." : "Save Configuration"}
          </button>
        </section>

        <section className="settings-section">
          <h2>About</h2>
          <p className="about-text">modllm v1.0.0</p>
        </section>
      </div>
    </div>
  );
}