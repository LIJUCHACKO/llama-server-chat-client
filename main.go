package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/chzyer/readline"
)

// ══════════════════════════════════════════════════════════════════════════════
// Types – OpenAI chat + tool-calling
// ══════════════════════════════════════════════════════════════════════════════

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	Name       string     `json:"name,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Index    int          `json:"index"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ── Tool definitions ─────────────────────────────────────────────────────────

type ToolDef struct {
	Type     string      `json:"type"`
	Function ToolFuncDef `json:"function"`
}

type ToolFuncDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ── Request / Response ────────────────────────────────────────────────────────

type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Stream      bool      `json:"stream"`
	Tools       []ToolDef `json:"tools,omitempty"`
	ToolChoice  string    `json:"tool_choice,omitempty"`
}

type Delta struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type Choice struct {
	Index        int    `json:"index"`
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason"`
}

type StreamChunk struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
}

type NonStreamChoice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

type ChatResponse struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []NonStreamChoice `json:"choices"`
}

// ══════════════════════════════════════════════════════════════════════════════
// Config
// ══════════════════════════════════════════════════════════════════════════════

type Config struct {
	BaseURL      string
	APIKey       string
	Model        string
	SystemPrompt string
	Temperature  float64
	MaxTokens    int
	Stream       bool
	WorkDir      string
}

// ══════════════════════════════════════════════════════════════════════════════
// File-server tools
// ══════════════════════════════════════════════════════════════════════════════

// privateFiles are internal application files that tools must never access.
var privateFiles = map[string]bool{
	".history":      true,
	".instructions": true,
}

func resolveSafe(cfg Config, p string) (string, error) {
	if !filepath.IsAbs(p) {
		p = filepath.Join(cfg.WorkDir, p)
	}
	abs, err := filepath.Abs(p)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(cfg.WorkDir, abs)
	if err != nil || strings.HasPrefix(rel, "..") {
		return "", fmt.Errorf("path %q is outside the working directory %q", p, cfg.WorkDir)
	}
	// Block internal application files at every level of the path.
	for _, part := range strings.Split(filepath.ToSlash(rel), "/") {
		if privateFiles[part] {
			return "", fmt.Errorf("access to %q is not permitted", rel)
		}
	}
	return abs, nil
}

func toolListDir(cfg Config, args map[string]any) string {
	rawPath, _ := args["path"].(string)
	if rawPath == "" {
		rawPath = "."
	}
	path, err := resolveSafe(cfg, rawPath)
	if err != nil {
		return "error: " + err.Error()
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return "error: " + err.Error()
	}
	var sb strings.Builder
	for _, e := range entries {
		if privateFiles[e.Name()] {
			continue
		}
		info, _ := e.Info()
		size := int64(0)
		if info != nil {
			size = info.Size()
		}
		kind := "file"
		if e.IsDir() {
			kind = "dir"
		}
		sb.WriteString(fmt.Sprintf("%-6s  %8d  %s\n", kind, size, e.Name()))
	}
	return sb.String()
}

func toolReadFile(cfg Config, args map[string]any) string {
	rawPath, _ := args["path"].(string)
	path, err := resolveSafe(cfg, rawPath)
	if err != nil {
		return "error: " + err.Error()
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "error: " + err.Error()
	}
	return string(data)
}

func toolWriteFile(cfg Config, args map[string]any) string {
	rawPath, _ := args["path"].(string)
	content, _ := args["content"].(string)
	path, err := resolveSafe(cfg, rawPath)
	if err != nil {
		return "error: " + err.Error()
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return "error: " + err.Error()
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "error: " + err.Error()
	}
	return fmt.Sprintf("written %d bytes to %s", len(content), path)
}

func toolAppendFile(cfg Config, args map[string]any) string {
	rawPath, _ := args["path"].(string)
	content, _ := args["content"].(string)
	path, err := resolveSafe(cfg, rawPath)
	if err != nil {
		return "error: " + err.Error()
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return "error: " + err.Error()
	}
	defer f.Close()
	n, err := f.WriteString(content)
	if err != nil {
		return "error: " + err.Error()
	}
	return fmt.Sprintf("appended %d bytes to %s", n, path)
}

func toolCreateDir(cfg Config, args map[string]any) string {
	rawPath, _ := args["path"].(string)
	path, err := resolveSafe(cfg, rawPath)
	if err != nil {
		return "error: " + err.Error()
	}
	if err := os.MkdirAll(path, 0755); err != nil {
		return "error: " + err.Error()
	}
	return "created: " + path
}

func toolDeletePath(cfg Config, args map[string]any) string {
	rawPath, _ := args["path"].(string)
	recursive, _ := args["recursive"].(bool)
	path, err := resolveSafe(cfg, rawPath)
	if err != nil {
		return "error: " + err.Error()
	}
	if recursive {
		err = os.RemoveAll(path)
	} else {
		err = os.Remove(path)
	}
	if err != nil {
		return "error: " + err.Error()
	}
	return "deleted: " + path
}

func toolMovePath(cfg Config, args map[string]any) string {
	rawSrc, _ := args["src"].(string)
	rawDst, _ := args["dst"].(string)
	src, err := resolveSafe(cfg, rawSrc)
	if err != nil {
		return "error: " + err.Error()
	}
	dst, err := resolveSafe(cfg, rawDst)
	if err != nil {
		return "error: " + err.Error()
	}
	if err := os.Rename(src, dst); err != nil {
		return "error: " + err.Error()
	}
	return fmt.Sprintf("moved %s -> %s", src, dst)
}

func toolSearchFiles(cfg Config, args map[string]any) string {
	rawPath, _ := args["path"].(string)
	pattern, _ := args["pattern"].(string)
	if rawPath == "" {
		rawPath = "."
	}
	root, err := resolveSafe(cfg, rawPath)
	if err != nil {
		return "error: " + err.Error()
	}
	const maxResults = 500
	var results []string
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		name := d.Name()
		// Skip hidden directories (e.g. .git, .svn) and private files.
		if d.IsDir() {
			if privateFiles[name] || (name != "." && strings.HasPrefix(name, ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		if privateFiles[name] {
			return nil
		}
		matched, _ := filepath.Match(pattern, name)
		if matched {
			rel, _ := filepath.Rel(cfg.WorkDir, p)
			results = append(results, rel)
			if len(results) >= maxResults {
				return filepath.SkipAll
			}
		}
		return nil
	})
	if len(results) == 0 {
		return "no matches"
	}
	out := strings.Join(results, "\n")
	if len(results) >= maxResults {
		out += fmt.Sprintf("\n(results capped at %d)", maxResults)
	}
	return out
}

func toolGetWorkDir(cfg Config) string {
	return cfg.WorkDir
}

// isBinary reports whether data looks like a binary (non-text) file by
// checking for a NUL byte in the first 8 KB.
func isBinary(data []byte) bool {
	check := data
	if len(check) > 8192 {
		check = check[:8192]
	}
	return bytes.ContainsRune(check, 0)
}

func toolGrepFiles(cfg Config, args map[string]any) string {
	rawPath, _ := args["path"].(string)
	pattern, _ := args["pattern"].(string)
	if pattern == "" {
		return "error: pattern is required"
	}
	if rawPath == "" {
		rawPath = "."
	}
	root, err := resolveSafe(cfg, rawPath)
	if err != nil {
		return "error: " + err.Error()
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return "error: invalid regexp: " + err.Error()
	}
	const maxResults = 500
	var results []string
	capped := false
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || capped {
			return nil
		}
		name := d.Name()
		// Skip hidden directories (e.g. .git, .svn) and private files.
		if d.IsDir() {
			if privateFiles[name] || (name != "." && strings.HasPrefix(name, ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		if privateFiles[name] {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil || isBinary(data) {
			return nil
		}
		lines := strings.Split(string(data), "\n")
		rel, _ := filepath.Rel(cfg.WorkDir, p)
		for i, line := range lines {
			if re.MatchString(line) {
				results = append(results, fmt.Sprintf("%s:%d: %s", rel, i+1, line))
				if len(results) >= maxResults {
					capped = true
					return filepath.SkipAll
				}
			}
		}
		return nil
	})
	if len(results) == 0 {
		return "no matches"
	}
	out := strings.Join(results, "\n")
	if capped {
		out += fmt.Sprintf("\n(results capped at %d)", maxResults)
	}
	return out
}

// ── Tool registry ─────────────────────────────────────────────────────────────

func buildTools() []ToolDef {
	param := func(s string) json.RawMessage { return json.RawMessage(s) }
	return []ToolDef{
		{Type: "function", Function: ToolFuncDef{
			Name:        "list_dir",
			Description: "List the contents of a directory. Returns name, type (file/dir), and size.",
			Parameters:  param(`{"type":"object","properties":{"path":{"type":"string","description":"Directory path relative to workdir (default '.')"}},"required":[]}`),
		}},
		{Type: "function", Function: ToolFuncDef{
			Name:        "read_file",
			Description: "Read and return the full text contents of a file.",
			Parameters:  param(`{"type":"object","properties":{"path":{"type":"string","description":"File path relative to workdir"}},"required":["path"]}`),
		}},
		{Type: "function", Function: ToolFuncDef{
			Name:        "write_file",
			Description: "Write (overwrite) a file with the given content. Creates parent directories automatically.",
			Parameters:  param(`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string","description":"Text content to write"}},"required":["path","content"]}`),
		}},
		{Type: "function", Function: ToolFuncDef{
			Name:        "append_file",
			Description: "Append text to the end of an existing file (creates it if absent).",
			Parameters:  param(`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string","description":"Text to append"}},"required":["path","content"]}`),
		}},
		{Type: "function", Function: ToolFuncDef{
			Name:        "create_dir",
			Description: "Create a directory and all parent directories.",
			Parameters:  param(`{"type":"object","properties":{"path":{"type":"string","description":"Directory path to create"}},"required":["path"]}`),
		}},
		{Type: "function", Function: ToolFuncDef{
			Name:        "delete_path",
			Description: "Delete a file or directory.",
			Parameters:  param(`{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean","description":"If true, delete directory and all its contents"}},"required":["path"]}`),
		}},
		{Type: "function", Function: ToolFuncDef{
			Name:        "move_path",
			Description: "Move or rename a file or directory.",
			Parameters:  param(`{"type":"object","properties":{"src":{"type":"string","description":"Source path"},"dst":{"type":"string","description":"Destination path"}},"required":["src","dst"]}`),
		}},
		{Type: "function", Function: ToolFuncDef{
			Name:        "search_files_with_name",
			Description: "Recursively search for files/dirs names matching a glob pattern (e.g. '*.go').",
			Parameters:  param(`{"type":"object","properties":{"path":{"type":"string","description":"Root directory to search from (default '.')"},"pattern":{"type":"string","description":"Glob pattern e.g. '*.txt'"}},"required":["pattern"]}`),
		}},
		{Type: "function", Function: ToolFuncDef{
			Name:        "get_workdir",
			Description: "Return the current working directory root accessible to file tools.",
			Parameters:  param(`{"type":"object","properties":{},"required":[]}`),
		}},
		{Type: "function", Function: ToolFuncDef{
			Name:        "search_file_contents",
			Description: "Recursively search file contents under a directory for lines matching a regular expression. Returns matching lines in the format 'path:line_number: line_content'.",
			Parameters:  param(`{"type":"object","properties":{"path":{"type":"string","description":"Root directory to search from (default '.')"},"pattern":{"type":"string","description":"Regular expression to match against file contents"}},"required":["pattern"]}`),
		}},
	}
}

func dispatchTool(cfg Config, name string, rawArgs string) string {
	var args map[string]any
	_ = json.Unmarshal([]byte(rawArgs), &args)
	if args == nil {
		args = map[string]any{}
	}
	switch name {
	case "list_dir":
		return toolListDir(cfg, args)
	case "read_file":
		return toolReadFile(cfg, args)
	case "write_file":
		return toolWriteFile(cfg, args)
	case "append_file":
		return toolAppendFile(cfg, args)
	case "create_dir":
		return toolCreateDir(cfg, args)
	case "delete_path":
		return toolDeletePath(cfg, args)
	case "move_path":
		return toolMovePath(cfg, args)
	case "search_files_with_name":
		return toolSearchFiles(cfg, args)
	case "get_workdir":
		return toolGetWorkDir(cfg)
	case "search_file_contents":
		return toolGrepFiles(cfg, args)
	default:
		return fmt.Sprintf("error: unknown tool %q", name)
	}
}

// ══════════════════════════════════════════════════════════════════════════════
// HTTP helpers
// ══════════════════════════════════════════════════════════════════════════════

func doRequest(client *http.Client, cfg Config, history []Message, tools []ToolDef) (Message, error) {
	// Prepend system message to a copy of history; never stored in the caller's history slice
	msgs := make([]Message, 0, len(history)+1)
	msgs = append(msgs, Message{Role: "system", Content: cfg.SystemPrompt})
	msgs = append(msgs, history...)
	if cfg.Stream {
		return doStream(client, cfg, msgs, tools)
	}
	return doNoStream(client, cfg, msgs, tools)
}

func doNoStream(client *http.Client, cfg Config, history []Message, tools []ToolDef) (Message, error) {
	req := ChatRequest{
		Model:       cfg.Model,
		Messages:    history,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxTokens,
		Stream:      false,
		Tools:       tools,
	}
	if len(tools) > 0 {
		req.ToolChoice = "auto"
	}
	body, err := json.Marshal(req)
	if err != nil {
		return Message{}, err
	}
	httpReq, err := http.NewRequest("POST", cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Message{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return Message{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return Message{}, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(b))
	}
	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return Message{}, err
	}
	if len(chatResp.Choices) == 0 {
		return Message{}, fmt.Errorf("no choices in response")
	}
	return chatResp.Choices[0].Message, nil
}

func doStream(client *http.Client, cfg Config, history []Message, tools []ToolDef) (Message, error) {
	req := ChatRequest{
		Model:       cfg.Model,
		Messages:    history,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxTokens,
		Stream:      true,
		Tools:       tools,
	}
	if len(tools) > 0 {
		req.ToolChoice = "auto"
	}
	body, err := json.Marshal(req)
	if err != nil {
		return Message{}, err
	}
	httpReq, err := http.NewRequest("POST", cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Message{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return Message{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return Message{}, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(b))
	}

	type tcAcc struct {
		id      string
		typ     string
		name    string
		argsBuf strings.Builder
	}
	tcMap := map[int]*tcAcc{}
	var contentSB strings.Builder
	printing := false

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta

		if delta.Content != "" {
			if !printing {
				fmt.Print("\n\033[1;36mAssistant:\033[0m ")
				printing = true
			}
			fmt.Print(delta.Content)
			contentSB.WriteString(delta.Content)
		}

		for _, tc := range delta.ToolCalls {
			acc, ok := tcMap[tc.Index]
			if !ok {
				acc = &tcAcc{}
				tcMap[tc.Index] = acc
			}
			if tc.ID != "" {
				acc.id = tc.ID
			}
			if tc.Type != "" {
				acc.typ = tc.Type
			}
			if tc.Function.Name != "" {
				acc.name = tc.Function.Name
			}
			acc.argsBuf.WriteString(tc.Function.Arguments)
		}
	}
	if printing {
		fmt.Println()
	}
	if err := scanner.Err(); err != nil {
		return Message{}, err
	}

	msg := Message{Role: "assistant", Content: contentSB.String()}
	for i := 0; i < len(tcMap); i++ {
		acc, ok := tcMap[i]
		if !ok {
			continue
		}
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{
			ID:   acc.id,
			Type: acc.typ,
			Function: FunctionCall{
				Name:      acc.name,
				Arguments: acc.argsBuf.String(),
			},
		})
	}
	return msg, nil
}

// ══════════════════════════════════════════════════════════════════════════════
// Agentic loop
// ══════════════════════════════════════════════════════════════════════════════

func agentLoop(client *http.Client, cfg Config, history []Message, tools []ToolDef) ([]Message, error) {
	for {
		msg, err := doRequest(client, cfg, history, tools)
		if err != nil {
			return history, err
		}
		history = append(history, msg)

		if len(msg.ToolCalls) == 0 {
			// Final text answer already printed by doStream / doNoStream
			if !cfg.Stream && msg.Content != "" {
				fmt.Printf("\n\033[1;36mAssistant:\033[0m %s\n", msg.Content)
			}
			return history, nil
		}

		// Execute tools
		for _, tc := range msg.ToolCalls {
			fmt.Printf("\033[90m[tool] %s(%s ..)\033[0m\n", tc.Function.Name, truncate(tc.Function.Arguments, 100))
			result := dispatchTool(cfg, tc.Function.Name, tc.Function.Arguments)
			fmt.Printf("\033[90m[result] %s  ...\033[0m\n", truncate(result, 100))
			history = append(history, Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Name:       tc.Function.Name,
				Content:    result,
			})
		}
	}
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", "↵ ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// ══════════════════════════════════════════════════════════════════════════════
// Input reader
// ══════════════════════════════════════════════════════════════════════════════

func readInput(rl *readline.Instance) (string, error) {
	// First line – readline gives us history navigation (↑/↓) for free.
	rl.SetPrompt("\033[1;32mYou:\033[0m ")
	firstLine, err := rl.Readline()
	if err != nil {
		return "", err // io.EOF on Ctrl-D
	}
	firstLine = strings.TrimRight(firstLine, "\r\n")

	// Single-line input that doesn't request continuation.
	if !strings.HasSuffix(firstLine, "\\") && firstLine != "//" {
		return firstLine, nil
	}

	// Multiline mode: keep reading continuation lines.
	lines := []string{strings.TrimSuffix(firstLine, "\\")}
	stdinReader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\033[90m... \033[0m")
		line, err := stdinReader.ReadString('\n')
		if err != nil {
			break
		}
		trimmed := strings.TrimRight(line, "\r\n")
		if trimmed == "//" {
			break
		}
		toBeContinued := strings.HasSuffix(trimmed, "\\")
		if toBeContinued {
			trimmed = strings.TrimSuffix(trimmed, "\\")
		}
		lines = append(lines, trimmed)
		if !toBeContinued {
			break
		}
	}
	return strings.Join(lines, "\n"), nil
}

// ══════════════════════════════════════════════════════════════════════════════
// History persistence
// ══════════════════════════════════════════════════════════════════════════════

func saveHistory(workDir string, history []Message) {
	histPath := filepath.Join(workDir, ".history")
	if len(history) == 0 {
		// Make .history file blank
		if err := os.WriteFile(histPath, []byte{}, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "\033[33m[warn] could not blank .history: %v\033[0m\n", err)
		}
		return
	}
	data, err := json.MarshalIndent(history, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "\033[33m[warn] could not marshal history: %v\033[0m\n", err)
		return
	}
	if err := os.WriteFile(histPath, data, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "\033[33m[warn] could not save history: %v\033[0m\n", err)
		return
	}
	fmt.Printf("\033[90m[history saved to %s]\033[0m\n", histPath)
}

func loadHistory(workDir string) []Message {
	histPath := filepath.Join(workDir, ".history")
	data, err := os.ReadFile(histPath)
	if err != nil {
		return nil
	}
	var msgs []Message
	if err := json.Unmarshal(data, &msgs); err != nil {
		fmt.Fprintf(os.Stderr, "\033[33m[warn] could not parse history file: %v\033[0m\n", err)
		return nil
	}
	return msgs
}

func printHistory(history []Message) {
	if len(history) == 0 {
		return
	}
	fmt.Println("\033[90m──────────────── Loaded History ────────────────\033[0m")
	for _, m := range history {
		switch m.Role {
		case "user":
			fmt.Printf("\033[1;32mYou:\033[0m %s\n", m.Content)
		case "assistant":
			fmt.Printf("\033[1;36mAssistant:\033[0m %s\n", m.Content)
		case "tool":
			fmt.Printf("\033[90m[tool result] %s: %s\033[0m\n", m.Name, truncate(m.Content, 120))
		}
	}
	fmt.Println("\033[90m────────────────────────────────────────────────\033[0m")
}

// ══════════════════════════════════════════════════════════════════════════════
// Model auto-detection
// ══════════════════════════════════════════════════════════════════════════════

// fetchModelName queries GET /v1/models and returns the first model id found.
func fetchModelName(baseURL string) (string, error) {
	resp, err := http.Get(baseURL + "/models")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("server returned %d: %s", resp.StatusCode, string(b))
	}
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return "", err
	}
	// prefer the OpenAI-style "data" array, fall back to llama-style "models"
	if len(payload.Data) > 0 && payload.Data[0].ID != "" {
		return payload.Data[0].ID, nil
	}
	if len(payload.Models) > 0 && payload.Models[0].Name != "" {
		return payload.Models[0].Name, nil
	}
	return "", fmt.Errorf("no models found in response")
}

// ══════════════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════════════

func main() {
	fmt.Printf("\033[50;1mVersion 1.0.1\033[0m\n")
	ip := flag.String("ip", "10.11.0.9", "llama.cpp server IP address")
	port := flag.Int("port", 8089, "llama.cpp server port")
	apiKey := flag.String("key", "", "API key (optional)")
	modelFlag := flag.String("model", "", "model name (default: auto-detected from server)")
	system := flag.String("system", "You are a helpful assistant.", "system prompt")
	temp := flag.Float64("temp", 0.7, "temperature (0.0-2.0)")
	maxTokens := flag.Int("max", 0, "max tokens (0 = server default)")
	noStream := flag.Bool("nostream", false, "disable streaming")
	noTools := flag.Bool("notools", false, "disable file-server tools")
	workDir := flag.String("workdir", "", "root directory for file tools (default: cwd)")

	// Custom usage printer
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: cmdchat [options]\n ( eg: cmdchat -ip 10.11.0.9 -port 8089 )\n\nOptions:")
		flag.VisitAll(func(f *flag.Flag) {
			fmt.Fprintf(os.Stderr, "  - %-10s  %s (default: %q)\n", f.Name, f.Usage, f.DefValue)
		})
		fmt.Fprintf(os.Stderr, "\nIn-chat commands: /quit  /clear  /history  /workdir  /help\n")
	}

	flag.Parse()

	// Support 'help' as a bare positional argument
	if args := flag.Args(); len(args) > 0 && strings.ToLower(args[0]) == "help" {
		flag.Usage()
		return
	}

	baseURL := fmt.Sprintf("http://%s:%d/v1", *ip, *port)

	// Auto-detect model name from the server
	modelName := *modelFlag
	if modelName == "" {
		fmt.Printf("\033[90mDetecting model from %s/models …\033[0m ", baseURL)
		var err error
		modelName, err = fetchModelName(baseURL)
		if err != nil {
			fmt.Printf("\033[33mfailed (%v)'\033[0m\n", err)
			flag.Usage()
			os.Exit(1)
		} else {
			fmt.Printf("\033[32m%s\033[0m\n", modelName)
		}
	}

	wd := *workDir
	if wd == "" {
		var err error
		wd, err = os.Getwd()
		if err != nil {
			fmt.Fprintln(os.Stderr, "cannot determine cwd:", err)
			os.Exit(1)
		}
	} else {
		var err error
		wd, err = filepath.Abs(wd)
		if err != nil {
			fmt.Fprintln(os.Stderr, "invalid workdir:", err)
			os.Exit(1)
		}
	}

	cfg := Config{
		BaseURL:      baseURL,
		APIKey:       *apiKey,
		Model:        modelName,
		SystemPrompt: *system,
		Temperature:  *temp,
		MaxTokens:    *maxTokens,
		Stream:       !*noStream,
		WorkDir:      wd,
	}

	var tools []ToolDef
	if !*noTools {
		tools = buildTools()
	}

	client := &http.Client{}

	systemContent := cfg.SystemPrompt
	if len(tools) > 0 {
		systemContent += fmt.Sprintf(
			"\n\nYou have access to file-system tools. The working directory is: %s\n"+
				"Available tools: list_dir, read_file, write_file, append_file, create_dir, delete_path, move_path, search_files_with_name, search_file_contents, get_workdir.",
			cfg.WorkDir,
		)
	}

	// Load .instructions file and enhance system content
	instructionsPath := filepath.Join(cfg.WorkDir, ".instructions")
	instrData, instrErr := os.ReadFile(instructionsPath)
	if instrErr != nil {
		// Create blank .instructions file if it doesn't exist
		_ = os.WriteFile(instructionsPath, []byte(""), 0644)
	} else if len(strings.TrimSpace(string(instrData))) > 0 {
		systemContent += "\n\n" + strings.TrimSpace(string(instrData))
	}

	// Store final systemContent back into cfg so doRequest can prepend it
	cfg.SystemPrompt = systemContent

	fmt.Println("\033[1;33m╔═════════════════════════════════════════════════════════════╗\033[0m")
	fmt.Println("\033[1;33m║   llama.cpp Chat  (type /quit to exit, /help for commands)  ║\033[0m")
	fmt.Println("\033[1;33m╚═════════════════════════════════════════════════════════════╝\033[0m")
	toolsLabel := "disabled"
	if len(tools) > 0 {
		toolsLabel = fmt.Sprintf("enabled  (workdir: %s)", cfg.WorkDir)
	}
	fmt.Printf("\033[90mURL: %s  |  Model: %s  |  Stream: %v\033[0m\n", cfg.BaseURL, cfg.Model, cfg.Stream)
	fmt.Printf("\033[90mFile tools: %s\033[0m\n\n", toolsLabel)

	// Load previous history from .history file and display it
	history := loadHistory(cfg.WorkDir)
	if history == nil {
		history = []Message{}
	} else {
		printHistory(history)
	}

	rl, err := readline.NewEx(&readline.Config{
		HistoryLimit: 200,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "readline init error:", err)
		os.Exit(1)
	}
	defer rl.Close()

	for {
		input, err := readInput(rl)
		if err != nil {
			if err == io.EOF || err == readline.ErrInterrupt {
				saveHistory(cfg.WorkDir, history)
				fmt.Println("\nBye!")
				return
			}
			fmt.Fprintln(os.Stderr, "read error:", err)
			return
		}

		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}

		switch strings.ToLower(input) {
		case "/quit", "/exit", "/q":
			saveHistory(cfg.WorkDir, history)
			fmt.Println("Bye!")
			return
		case "/clear", "/reset":
			history = []Message{}
			fmt.Println("\033[90m[conversation cleared]\033[0m")
			continue
		case "/history":
			for i, m := range history {
				if m.Role == "system" {
					continue
				}
				fmt.Printf("\033[90m[%d] %s: %s\033[0m\n", i, m.Role, truncate(m.Content, 120))
			}
			continue
		case "/workdir":
			fmt.Printf("\033[90mWorkdir: %s\033[0m\n", cfg.WorkDir)
			continue
		case "/help":
			fmt.Println("\033[90mCommands  		 : /quit  /clear  /history  /workdir  /help\033[0m")
			fmt.Println("\033[90mMultiline prompt: line ending should be '\\'\033[0m")
			fmt.Println("\033[90mFile tools      : list_dir  read_file  write_file  append_file\033[0m")
			fmt.Println("\033[90m                  create_dir  delete_path  move_path  search_files_with_name  search_file_contents  get_workdir\033[0m")
			continue
		}

		history = append(history, Message{Role: "user", Content: input})

		history, err = agentLoop(client, cfg, history, tools)
		if err != nil {
			fmt.Fprintln(os.Stderr, "\033[1;31merror:\033[0m", err)
			history = history[:len(history)-1]
		}
	}
}
