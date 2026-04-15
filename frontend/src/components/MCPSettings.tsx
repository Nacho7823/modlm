import { useEffect, useState } from "react";
import {
  getMCPConfig,
  updateMCPConfig,
  testMCPConnection,
  TestConnectionResult,
} from "../utils/api";
import type { MCPServerConfig } from "../types";

const EXAMPLE_MCP_CONFIG = `[
  {
    "name": "memory-server",
    "type": "local",
    "command": ["npx", "-y", "@modelcontextprotocol/server-memory"],
    "enabled": true
  },
  {
    "name": "filesystem",
    "type": "local",
    "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path/to/folder"],
    "enabled": false
  },
  {
    "name": "remote-mcp",
    "type": "remote",
    "url": "https://remotemcp.example.com/mcp",
    "headers": {
      "Authorization": "Bearer your-api-key-here"
    },
    "timeout": 30,
    "enabled": false
  }
]`;

function validateMCPConfig(json: unknown): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!Array.isArray(json)) {
    return { valid: false, errors: ["Config must be an array of MCP servers"] };
  }

  for (let i = 0; i < json.length; i++) {
    const server = json[i] as Record<string, unknown>;

    if (typeof server.name !== "string" || !server.name) {
      errors.push(`Server[${i}]: name is required and must be a string`);
    }

    if (server.type !== "local" && server.type !== "remote") {
      errors.push(`Server[${i}]: type must be "local" or "remote"`);
    }

    if (server.type === "local" && !Array.isArray(server.command)) {
      errors.push(`Server[${i}]: command is required for local type`);
    }

    if (server.type === "remote" && typeof server.url !== "string") {
      errors.push(`Server[${i}]: url is required for remote type`);
    }

    if (typeof server.timeout === "number") {
      if (server.timeout < 1 || server.timeout > 300) {
        errors.push(`Server[${i}]: timeout must be between 1 and 300`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

export function MCPSettings() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [jsonEditor, setJsonEditor] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [testingServer, setTestingServer] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<
    Record<string, TestConnectionResult>
  >({});

  useEffect(() => {
    loadMCPConfig();
  }, []);

  const loadMCPConfig = async () => {
    try {
      const config = await getMCPConfig();
      setJsonEditor(JSON.stringify(config.servers, null, 2));
    } catch {
      setJsonEditor("[]");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setJsonError(null);
    setSuccess(null);

    try {
      const parsed = JSON.parse(jsonEditor);
      const validation = validateMCPConfig(parsed);

      if (!validation.valid) {
        setJsonError(validation.errors.join("\n"));
        setSaving(false);
        return;
      }

      await updateMCPConfig(parsed);
      setSuccess("Configuration saved successfully!");
      setTimeout(() => setSuccess(null), 3000);
    } catch {
      setJsonError("Invalid JSON syntax");
    } finally {
      setSaving(false);
    }
  };

  const handleCopyExample = async () => {
    await navigator.clipboard.writeText(EXAMPLE_MCP_CONFIG);
    setSuccess("Example copied to clipboard!");
    setTimeout(() => setSuccess(null), 2000);
  };

  const handleTest = async (serverName: string) => {
    setTestingServer(serverName);

    try {
      const result = await testMCPConnection(serverName);
      setTestResults((prev) => ({ ...prev, [serverName]: result }));
    } catch (err) {
      setTestResults((prev) => ({
        ...prev,
        [serverName]: {
          status: "error",
          tools: [],
          tool_count: 0,
          error: String(err),
        },
      }));
    } finally {
      setTestingServer(null);
    }
  };

  if (loading) {
    return (
      <div className="mcp-settings-container">
        <div className="mcp-settings-loading">Loading...</div>
      </div>
    );
  }

  let parsedServers: MCPServerConfig[] = [];
  try {
    parsedServers = JSON.parse(jsonEditor);
  } catch {
    parsedServers = [];
  }

  return (
    <div className="settings-section">
      <h2>MCP Servers</h2>

      <div className="form-group">
        <label htmlFor="mcp-json">Server Configuration (JSON)</label>
        <textarea
          id="mcp-json"
          value={jsonEditor}
          onChange={(e) => setJsonEditor(e.target.value)}
          className="form-input form-json-editor"
          placeholder={`[
  {
    "name": "memory-server",
    "type": "http",
    "url": "http://localhost:3000/mcp",
    "enabled": true
  },
  {
    "name": "filesystem-server", 
    "type": "local",
    "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    "enabled": false
  }
]`}
          rows={12}
        />
      </div>

      {jsonError && <div className="settings-error">{jsonError}</div>}
      {success && <div className="settings-success">{success}</div>}

      <div className="mcp-buttons-row">
        <button
          className="save-btn"
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? "Saving..." : "Save MCP Configuration"}
        </button>
        <button
          className="copy-example-btn"
          onClick={handleCopyExample}
        >
          Copy Example
        </button>
      </div>

      {parsedServers.length > 0 && (
        <div className="mcp-server-list">
          <h3>Test Connections</h3>
          {parsedServers.map((server) => (
            <div key={server.name} className="mcp-server-item">
              <div className="mcp-server-info">
                <strong>{server.name}</strong>
                <span className="mcp-server-type">{server.type}</span>
                {testResults[server.name] && (
                  <span
                    className={`mcp-status mcp-status-${testResults[server.name].status}`}
                  >
                    {testResults[server.name].status === "connected"
                      ? `Connected (${testResults[server.name].tool_count} tools)`
                      : testResults[server.name].error || "Error"}
                  </span>
                )}
              </div>
              <button
                className="test-connection-btn"
                onClick={() => handleTest(server.name)}
                disabled={testingServer === server.name}
              >
                {testingServer === server.name ? "Testing..." : "Test Connection"}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}