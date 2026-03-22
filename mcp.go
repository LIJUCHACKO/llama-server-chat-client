package main

// ══════════════════════════════════════════════════════════════════════════════
// MCP (Model Context Protocol) client
//
// Reads ~/.clichat/mcp-config.json, spawns each configured server as a child
// process, performs the JSON-RPC 2.0 handshake (initialize → tools/list),
// converts discovered tools into ToolDef objects understood by the LLM, and
// routes tool-call results back through tools/call.
//
// JSON-RPC framing: one JSON object per newline (stdio transport).
// ══════════════════════════════════════════════════════════════════════════════

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
)

// ── Config types ──────────────────────────────────────────────────────────────

// MCPFileConfig is the top-level structure of ~/.clichat/mcp-config.json.
type MCPFileConfig struct {
	Tools struct {
		MCPServers map[string]MCPServerDef `json:"mcpServers"`
	} `json:"tools"`
}

// MCPServerDef describes one MCP server entry.
type MCPServerDef struct {
	Command string            `json:"command"`
	Args    []string          `json:"args"`
	Env     map[string]string `json:"env"`
}

// ── JSON-RPC 2.0 wire types ───────────────────────────────────────────────────

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcNotification struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id"`
	Result  json.RawMessage `json:"result"`
	Error   *rpcError       `json:"error"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// ── MCP tool list/call wire types ─────────────────────────────────────────────

type mcpToolsListResult struct {
	Tools []mcpToolInfo `json:"tools"`
}

type mcpToolInfo struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

type mcpCallToolParams struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type mcpCallToolResult struct {
	Content []mcpContent `json:"content"`
	IsError bool         `json:"isError"`
}

type mcpContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ── MCPServer – one running child process ─────────────────────────────────────

// MCPServer holds the state for one live MCP server child process.
type MCPServer struct {
	name   string
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	mu     sync.Mutex // serialise requests/responses

	idCounter atomic.Int64
}

// start spawns the child process and performs the MCP handshake.
func (s *MCPServer) start(def MCPServerDef) error {
	cmd := exec.Command(def.Command, def.Args...)

	// Pass through the current environment, then overlay any extras.
	cmd.Env = os.Environ()
	for k, v := range def.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("stdin pipe: %w", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	// Redirect stderr to our own stderr so error output from child is visible.
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start: %w", err)
	}

	s.cmd = cmd
	s.stdin = stdin
	s.stdout = bufio.NewReader(stdoutPipe)

	// ── initialize handshake ──────────────────────────────────────────────────
	initParams, _ := json.Marshal(map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "clichat", "version": "1.0"},
	})
	resp, err := s.call("initialize", initParams)
	if err != nil {
		return fmt.Errorf("initialize: %w", err)
	}
	_ = resp // we don't need the server capabilities

	// send initialized notification (no response expected)
	notif := rpcNotification{JSONRPC: "2.0", Method: "notifications/initialized"}
	if err := s.sendNotification(notif); err != nil {
		return fmt.Errorf("initialized notification: %w", err)
	}

	return nil
}

// stop terminates the child process.
func (s *MCPServer) stop() {
	if s.stdin != nil {
		_ = s.stdin.Close()
	}
	if s.cmd != nil && s.cmd.Process != nil {
		_ = s.cmd.Process.Kill()
		_ = s.cmd.Wait()
	}
}

// call sends a JSON-RPC request and returns the raw result.
func (s *MCPServer) call(method string, params json.RawMessage) (json.RawMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	id := s.idCounter.Add(1)
	req := rpcRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}
	line, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	line = append(line, '\n')
	if _, err := s.stdin.Write(line); err != nil {
		return nil, fmt.Errorf("write to server: %w", err)
	}

	// Read lines until we get a response matching our id.
	// (Servers may send notifications before the response.)
	for {
		respLine, err := s.stdout.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("read from server: %w", err)
		}
		respLine = strings.TrimSpace(respLine)
		if respLine == "" {
			continue
		}
		var rpcResp rpcResponse
		if err := json.Unmarshal([]byte(respLine), &rpcResp); err != nil {
			// Might be a notification – skip it.
			continue
		}
		if rpcResp.ID == nil {
			// Notification – skip.
			continue
		}
		if *rpcResp.ID != id {
			// Response for a different id – skip (shouldn't happen in serial mode).
			continue
		}
		if rpcResp.Error != nil {
			return nil, fmt.Errorf("rpc error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
		}
		return rpcResp.Result, nil
	}
}

// sendNotification writes a JSON-RPC notification (no id, no response).
func (s *MCPServer) sendNotification(n rpcNotification) error {
	line, err := json.Marshal(n)
	if err != nil {
		return err
	}
	line = append(line, '\n')
	_, err = s.stdin.Write(line)
	return err
}

// listTools calls tools/list and converts the results to ToolDef objects.
// Each tool name is prefixed with "<serverName>__" to avoid collisions.
func (s *MCPServer) listTools() ([]ToolDef, error) {
	result, err := s.call("tools/list", nil)
	if err != nil {
		return nil, err
	}
	var tlr mcpToolsListResult
	if err := json.Unmarshal(result, &tlr); err != nil {
		return nil, fmt.Errorf("parse tools/list: %w", err)
	}

	defs := make([]ToolDef, 0, len(tlr.Tools))
	for _, t := range tlr.Tools {
		// Use the raw inputSchema as the parameter schema.
		// Fall back to an empty object schema if absent.
		schema := t.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{},"required":[]}`)
		}
		// Prefix the name so dispatchTool can route to the right server.
		defs = append(defs, ToolDef{
			Type: "function",
			Function: ToolFuncDef{
				Name:        s.name + "__" + t.Name,
				Description: t.Description,
				Parameters:  schema,
			},
		})
	}
	return defs, nil
}

// callTool dispatches a tools/call request to this server.
// toolName must be the bare (un-prefixed) tool name.
func (s *MCPServer) callTool(toolName string, rawArgs string) string {
	var arguments map[string]any
	if rawArgs != "" {
		_ = json.Unmarshal([]byte(rawArgs), &arguments)
	}
	if arguments == nil {
		arguments = map[string]any{}
	}
	params, err := json.Marshal(mcpCallToolParams{Name: toolName, Arguments: arguments})
	if err != nil {
		return "error: marshal call params: " + err.Error()
	}
	result, err := s.call("tools/call", params)
	if err != nil {
		return "error: " + err.Error()
	}
	var ctr mcpCallToolResult
	if err := json.Unmarshal(result, &ctr); err != nil {
		// Fall back: return raw JSON.
		return string(result)
	}
	if ctr.IsError {
		texts := make([]string, 0, len(ctr.Content))
		for _, c := range ctr.Content {
			if c.Type == "text" {
				texts = append(texts, c.Text)
			}
		}
		return "error: " + strings.Join(texts, "\n")
	}
	texts := make([]string, 0, len(ctr.Content))
	for _, c := range ctr.Content {
		if c.Type == "text" {
			texts = append(texts, c.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// ── MCPManager – collection of all running servers ────────────────────────────

// MCPManager owns all live MCPServer instances for the session.
type MCPManager struct {
	servers map[string]*MCPServer // key: server name
}

// NewMCPManager returns an empty manager.
func NewMCPManager() *MCPManager {
	return &MCPManager{servers: make(map[string]*MCPServer)}
}

// LoadAndStart reads ~/.clichat/mcp-config.json and starts every listed server.
// It returns the aggregated list of ToolDef objects and any startup warnings.
func (m *MCPManager) LoadAndStart() ([]ToolDef, []string) {
	cfgPath := filepath.Join(os.Getenv("HOME"), ".clichat", "mcp-config.json")
	data, err := os.ReadFile(cfgPath)
	if err != nil {
		if os.IsNotExist(err) {
			// Create the directory and a blank config file for the user.
			cfgDir := filepath.Dir(cfgPath)
			if mkErr := os.MkdirAll(cfgDir, 0755); mkErr != nil {
				return nil, []string{fmt.Sprintf("mcp: could not create config dir %s: %v", cfgDir, mkErr)}
			}
			blank := []byte("{\n  \"tools\": {\n    \"mcpServers\": {}\n  }\n}\n")
			if wErr := os.WriteFile(cfgPath, blank, 0644); wErr != nil {
				return nil, []string{fmt.Sprintf("mcp: could not create %s: %v", cfgPath, wErr)}
			}
			fmt.Printf("\033[90m[mcp] created blank config: %s\033[0m\n", cfgPath)
			return nil, nil // no servers configured yet
		}
		return nil, []string{fmt.Sprintf("mcp-config.json read error: %v", err)}
	}

	var cfg MCPFileConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, []string{fmt.Sprintf("mcp-config.json parse error: %v", err)}
	}

	var allTools []ToolDef
	var warnings []string

	for name, def := range cfg.Tools.MCPServers {
		if def.Command == "" {
			warnings = append(warnings, fmt.Sprintf("[mcp] %q: no command specified, skipping", name))
			continue
		}
		srv := &MCPServer{name: name}
		if err := srv.start(def); err != nil {
			warnings = append(warnings, fmt.Sprintf("[mcp] %q: failed to start: %v", name, err))
			continue
		}
		tools, err := srv.listTools()
		if err != nil {
			warnings = append(warnings, fmt.Sprintf("[mcp] %q: tools/list failed: %v", name, err))
			srv.stop()
			continue
		}
		m.servers[name] = srv
		allTools = append(allTools, tools...)
		fmt.Printf("\033[90m[mcp] started %q  (%d tool(s))\033[0m\n", name, len(tools))
	}
	return allTools, warnings
}

// Dispatch routes a prefixed tool call (e.g. "filesystem__read_file") to the
// correct server.  Returns an error string if the server is not found.
func (m *MCPManager) Dispatch(prefixedName string, rawArgs string) (string, bool) {
	idx := strings.Index(prefixedName, "__")
	if idx < 0 {
		return "", false // not an MCP tool
	}
	serverName := prefixedName[:idx]
	toolName := prefixedName[idx+2:]
	srv, ok := m.servers[serverName]
	if !ok {
		return fmt.Sprintf("error: mcp server %q not running", serverName), true
	}
	return srv.callTool(toolName, rawArgs), true
}

// StopAll terminates all child processes.
func (m *MCPManager) StopAll() {
	for name, srv := range m.servers {
		srv.stop()
		fmt.Printf("\033[90m[mcp] stopped %q\033[0m\n", name)
	}
}

// ── Config path helper ────────────────────────────────────────────────────────

// MCPConfigPath returns the canonical path to the config file.
func MCPConfigPath() string {
	return filepath.Join(os.Getenv("HOME"), ".clichat", "mcp-config.json")
}
