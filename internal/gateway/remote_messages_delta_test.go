package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// Responses metadata and argument formatting differ from a Messages client's replay.
func remoteMessagesReply(
	n int,
	calls bool,
) string {
	id := fmt.Sprintf("resp_messages_%d", n)
	message := map[string]any{
		"type": "message", "id": fmt.Sprintf("msg_%d", n), "status": "completed", "role": "assistant",
		"content": []any{map[string]any{"type": "output_text", "text": fmt.Sprintf("answer %d", n), "annotations": []any{}, "logprobs": []any{}}},
	}
	output := []any{message}
	if calls {
		output = []any{
			map[string]any{"type": "reasoning", "id": "rs_messages", "summary": []any{map[string]any{"type": "summary_text", "text": "Checking the two files."}}, "encrypted_content": "synthetic-sealed-reasoning"},
			message,
			map[string]any{"type": "function_call", "id": "fc_1", "status": "completed", "call_id": "call_1", "name": "read_file", "arguments": `{"path":"first","limit":2}`},
			map[string]any{"type": "function_call", "id": "fc_2", "status": "completed", "call_id": "call_2", "name": "read_file", "arguments": `{"path":"second","limit":2}`},
		}
	}
	line := func(v any) string {
		b, _ := json.Marshal(v)
		return "data: " + string(b)
	}
	lines := []string{line(map[string]any{"type": "response.created", "response": map[string]any{"id": id, "model": "m1"}})}
	for i, value := range output {
		item := value.(map[string]any)
		lines = append(lines, line(map[string]any{"type": "response.output_item.added", "output_index": i, "item": item}))
		switch item["type"] {
		case "reasoning":
			lines = append(lines, line(map[string]any{"type": "response.reasoning_summary_text.delta", "delta": "Checking the two files."}))
		case "message":
			lines = append(lines, line(map[string]any{"type": "response.output_text.delta", "delta": fmt.Sprintf("answer %d", n)}))
		case "function_call":
			lines = append(lines, line(map[string]any{"type": "response.function_call_arguments.delta", "item_id": item["id"], "delta": item["arguments"]}))
		}
		lines = append(lines, line(map[string]any{"type": "response.output_item.done", "output_index": i, "item": item}))
	}
	lines = append(lines, line(map[string]any{"type": "response.completed", "response": map[string]any{
		"id": id, "object": "response", "model": "m1", "status": "completed", "output": output,
		"usage": map[string]any{"input_tokens": 10, "output_tokens": 5},
	}}))
	return sse(lines...)
}

type remoteMessagesFixture struct {
	handler http.Handler
	hop     *hopLog
	vendor  *hopLog
}

func messagesRemote(
	t *testing.T,
	keep bool,
	lost bool,
	calls bool,
) remoteMessagesFixture {
	t.Helper()
	freshRemoteTurns(t)
	noRests()
	t.Cleanup(noRests)
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	vendor, hop := &hopLog{}, &hopLog{}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		vendor.add(r)
		n := vendor.n()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, remoteMessagesReply(n, calls && n == 1))
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{ID: "fake", Name: "Fake", Key: "k", Models: []string{"m1"}, Responses: up.URL + "/v1"}); err != nil {
		t.Fatal(err)
	}
	h := New().Handler()
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			b := hop.add(r)
			if lost && hop.body(hop.n() - 1)["previous_response_id"] == "resp_messages_1" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, `{"error":{"type":"invalid_request_error","code":"previous_response_not_found","message":"previous_response_id was lost"}}`)
				return
			}
			r.Body = io.NopCloser(strings.NewReader(string(b)))
		}
		if !keep {
			r.Header.Del(KeepTurnHeader)
		}
		h.ServeHTTP(w, r)
	}))
	t.Cleanup(remote.Close)
	if err := provider.Save(provider.Provider{ID: "office", Name: "Office", Preset: provider.RemoteMagpiePreset, Key: "rk", Models: []string{"fake/m1"},
		Chat: remote.URL + "/v1", Responses: remote.URL + "/v1", Anthropic: remote.URL}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelAPI("office/fake/m1", "responses"); err != nil {
		t.Fatal(err)
	}
	return remoteMessagesFixture{h, hop, vendor}
}

func messagesTurn(
	stream bool,
	messages []any,
) string {
	b, _ := json.Marshal(map[string]any{
		"model": "office/fake/m1", "max_tokens": 1000, "stream": stream, "system": "Keep the history marker.",
		"messages": messages, "tools": []any{map[string]any{"name": "read_file", "input_schema": map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}, "limit": map[string]any{"type": "integer"}}}}},
	})
	return string(b)
}

func askMessagesTurn(
	t *testing.T,
	h http.Handler,
	body string,
) []any {
	t.Helper()
	rec := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	r.RemoteAddr = "127.0.0.1:5555"
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("X-Session-ID", "synthetic-messages-delta")
	h.ServeHTTP(rec, r)
	if rec.Code != http.StatusOK {
		t.Fatalf("Messages reply: %d %.300s", rec.Code, rec.Body.String())
	}
	b := rec.Body.Bytes()
	if strings.Contains(rec.Header().Get("Content-Type"), "event-stream") {
		var c collector
		dec := decoder(provider.Anthropic)
		if err := readSSE(strings.NewReader(rec.Body.String()), func(_, data string) error { return dec(data, c.add) }); err != nil {
			t.Fatal(err)
		}
		if c.err != "" {
			t.Fatalf("stream error: %s", c.err)
		}
		b = renderAnthropic(c.finish(), "m1")
	}
	var reply struct {
		Content []any `json:"content"`
	}
	if err := json.Unmarshal(b, &reply); err != nil || len(reply.Content) == 0 {
		t.Fatalf("no Messages content: %v %.200s", err, b)
	}
	return reply.Content
}

func initialMessages() []any {
	return []any{map[string]any{"role": "user", "content": "HISTORY_MARKER " + strings.Repeat("old history ", 8000)}}
}

func continueMessages(
	messages []any,
	content []any,
	calls bool,
) []any {
	messages = append(messages, map[string]any{"role": "assistant", "content": content})
	if calls {
		var results []any
		for _, value := range content {
			block := value.(map[string]any)
			if block["type"] == "tool_use" {
				results = append(results, map[string]any{"type": "tool_result", "tool_use_id": block["id"], "content": "synthetic file result"})
			}
		}
		return append(messages, map[string]any{"role": "user", "content": results})
	}
	return append(messages, map[string]any{"role": "user", "content": "NEW_TURN"})
}

func TestRemoteMessagesDelta(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, calls := range []bool{false, true} {
			t.Run(fmt.Sprintf("stream=%v/calls=%v", stream, calls), func(t *testing.T) {
				f := messagesRemote(t, true, false, calls)
				messages := initialMessages()
				answer := askMessagesTurn(t, f.handler, messagesTurn(stream, messages))
				if !strings.HasPrefix(f.hop.header(0).Get(KeepTurnHeader), "magpie/") {
					t.Fatal("translated Responses turn was not asked kept")
				}
				for turn := 2; turn <= 3; turn++ {
					messages = continueMessages(messages, answer, calls && turn == 2)
					answer = askMessagesTurn(t, f.handler, messagesTurn(stream, messages))
					q := f.hop.body(turn - 1)
					if q["previous_response_id"] == nil || strings.Contains(f.hop.raw(turn-1), "HISTORY_MARKER") || len(f.hop.raw(turn-1)) > 4000 {
						t.Fatalf("turn %d was not incremental: %d bytes, previous=%v", turn, len(f.hop.raw(turn-1)), q["previous_response_id"])
					}
					v := f.vendor.raw(turn - 1)
					if !strings.Contains(v, "HISTORY_MARKER") || !strings.Contains(v, "answer 1") || !strings.Contains(v, "Keep the history marker") {
						t.Fatalf("turn %d lost context at the vendor", turn)
					}
					if calls && (!strings.Contains(v, "synthetic-sealed-reasoning") || !strings.Contains(v, "call_2") || !strings.Contains(v, "synthetic file result")) {
						t.Fatalf("turn %d lost reasoning, parallel calls or results", turn)
					}
					if q["tools"] == nil || q["instructions"] == nil {
						t.Fatal("delta omitted the current tools or instructions")
					}
				}
				if f.hop.n() != 3 || f.vendor.n() != 3 {
					t.Fatalf("extra asks: hop=%d vendor=%d", f.hop.n(), f.vendor.n())
				}
				for i := range f.vendor.n() {
					if f.vendor.body(i)["previous_response_id"] != nil || f.vendor.header(i).Get(KeepTurnHeader) != "" {
						t.Fatal("hop state reached the vendor")
					}
				}
			})
		}
	}
}

// The other magpie relays the built Responses body as it came, so the
// conversation's cache key is on every hop body, a delta's too, and the
// vendor behind it is sent the same one each turn.
func TestRemoteMessagesCacheKey(t *testing.T) {
	f := messagesRemote(t, true, false, false)
	messages := initialMessages()
	answer := askMessagesTurn(t, f.handler, messagesTurn(true, messages))
	for turn := 2; turn <= 3; turn++ {
		messages = continueMessages(messages, answer, false)
		answer = askMessagesTurn(t, f.handler, messagesTurn(true, messages))
	}
	if f.hop.n() != 3 || f.vendor.n() != 3 {
		t.Fatalf("asks: hop=%d vendor=%d", f.hop.n(), f.vendor.n())
	}
	if f.hop.body(2)["previous_response_id"] == nil {
		t.Fatal("the key kept the turn from going as a delta")
	}
	key, _ := f.hop.body(0)["prompt_cache_key"].(string)
	if key == "" {
		t.Fatal("the first turn went without a prompt cache key")
	}
	for i := range 3 {
		if got := f.hop.body(i)["prompt_cache_key"]; got != key {
			t.Errorf("turn %d: hop key %v, want %s", i+1, got, key)
		}
		if got := f.vendor.body(i)["prompt_cache_key"]; got != key {
			t.Errorf("turn %d: vendor key %v, want %s", i+1, got, key)
		}
	}
}

func TestRemoteMessagesDeltaFallback(t *testing.T) {
	for _, keep := range []bool{false, true} {
		t.Run(fmt.Sprintf("keeps=%v", keep), func(t *testing.T) {
			f := messagesRemote(t, keep, true, false)
			messages := initialMessages()
			answer := askMessagesTurn(t, f.handler, messagesTurn(true, messages))
			messages = continueMessages(messages, answer, false)
			answer = askMessagesTurn(t, f.handler, messagesTurn(true, messages))
			want := 2
			if keep {
				want = 3
				if f.hop.body(1)["previous_response_id"] == nil {
					t.Fatal("cache miss was not tried as a delta")
				}
			}
			if f.hop.n() != want || f.vendor.n() != 2 || !strings.Contains(f.hop.raw(want-1), "HISTORY_MARKER") {
				t.Fatalf("full fallback: hop=%d vendor=%d", f.hop.n(), f.vendor.n())
			}
			messages = continueMessages(messages, answer, false)
			askMessagesTurn(t, f.handler, messagesTurn(true, messages))
			if f.hop.n() != want+1 || f.vendor.n() != 3 {
				t.Fatalf("next turn: hop=%d vendor=%d", f.hop.n(), f.vendor.n())
			}
			if keep {
				if f.hop.body(want)["previous_response_id"] != "resp_messages_2" || strings.Contains(f.hop.raw(want), "HISTORY_MARKER") {
					t.Fatal("the newly kept response was not continued as a delta")
				}
			} else if f.hop.body(want)["previous_response_id"] != nil || !strings.Contains(f.hop.raw(want), "HISTORY_MARKER") {
				t.Fatal("an old remote was sent a delta without an acknowledgment")
			}
		})
	}
}

// A receiver restart loses both conversations. Rebuilding one lets it
// continue immediately, while the other's old acknowledgment stays held.
func TestRemoteMessagesDeltaAfterReceiverRestart(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			f := messagesRemote(t, true, false, false)
			messages := initialMessages()
			answer := askMessagesTurn(t, f.handler, messagesTurn(stream, messages))
			other := initialMessages()
			other[0].(map[string]any)["content"] = "OTHER_HISTORY_MARKER"
			otherAnswer := askMessagesTurn(t, f.handler, messagesTurn(stream, other))

			// The sender keeps its metadata, but the receiver starts empty.
			prevResponses = newPrevStore(prevCap, prevTTL)
			messages = continueMessages(messages, answer, false)
			answer = askMessagesTurn(t, f.handler, messagesTurn(stream, messages))
			if f.hop.n() != 4 || f.vendor.n() != 3 {
				t.Fatalf("rebuild: hop=%d vendor=%d", f.hop.n(), f.vendor.n())
			}
			if f.hop.body(2)["previous_response_id"] != "resp_messages_1" || f.hop.body(3)["previous_response_id"] != nil {
				t.Fatal("the lost delta was not rebuilt as a full request")
			}

			messages = continueMessages(messages, answer, false)
			askMessagesTurn(t, f.handler, messagesTurn(stream, messages))
			if f.hop.n() != 5 || f.vendor.n() != 4 {
				t.Fatalf("continuation: hop=%d vendor=%d", f.hop.n(), f.vendor.n())
			}
			if f.hop.body(4)["previous_response_id"] != "resp_messages_3" || strings.Contains(f.hop.raw(4), "HISTORY_MARKER") {
				t.Fatal("the rebuilt conversation remained full during the route cooldown")
			}
			for _, text := range []string{"HISTORY_MARKER", "answer 1", "answer 3"} {
				if !strings.Contains(f.vendor.raw(3), text) {
					t.Fatalf("restored conversation lost %q", text)
				}
			}

			other = continueMessages(other, otherAnswer, false)
			otherAnswer = askMessagesTurn(t, f.handler, messagesTurn(stream, other))
			if f.hop.n() != 6 || f.vendor.n() != 5 || f.hop.body(5)["previous_response_id"] != nil {
				t.Fatal("the other conversation retried its stale acknowledgment")
			}
			other = continueMessages(other, otherAnswer, false)
			askMessagesTurn(t, f.handler, messagesTurn(stream, other))
			if f.hop.n() != 7 || f.vendor.n() != 6 {
				t.Fatalf("other continuation: hop=%d vendor=%d", f.hop.n(), f.vendor.n())
			}
			if f.hop.body(6)["previous_response_id"] != "resp_messages_5" || strings.Contains(f.hop.raw(6), "OTHER_HISTORY_MARKER") {
				t.Fatal("the other rebuilt conversation remained full")
			}
		})
	}
}

func TestRemoteMessagesDeltaEdits(t *testing.T) {
	for _, edit := range []string{"history", "answer", "arguments"} {
		t.Run(edit, func(t *testing.T) {
			f := messagesRemote(t, true, false, true)
			messages := initialMessages()
			answer := askMessagesTurn(t, f.handler, messagesTurn(true, messages))
			switch edit {
			case "history":
				messages[0].(map[string]any)["content"] = "EDITED_HISTORY"
			case "answer":
				for _, value := range answer {
					block := value.(map[string]any)
					if block["type"] == "text" {
						block["text"] = "EDITED_ANSWER"
					}
				}
			case "arguments":
				for _, value := range answer {
					block := value.(map[string]any)
					if block["type"] == "tool_use" {
						block["input"] = map[string]any{"path": "EDITED_ARGUMENT", "limit": 2}
					}
				}
			}
			messages = continueMessages(messages, answer, true)
			askMessagesTurn(t, f.handler, messagesTurn(true, messages))
			if f.hop.n() != 2 || f.hop.body(1)["previous_response_id"] != nil {
				t.Fatal("edited conversation substituted the cached answer")
			}
			marker := map[string]string{"history": "EDITED_HISTORY", "answer": "EDITED_ANSWER", "arguments": "EDITED_ARGUMENT"}[edit]
			if !strings.Contains(f.vendor.raw(1), marker) {
				t.Fatal("the vendor did not receive the edit")
			}
		})
	}
}

func TestRemoteMessagesDeltaMatchesFullInput(t *testing.T) {
	f := messagesRemote(t, true, false, true)
	messages := initialMessages()
	answer := askMessagesTurn(t, f.handler, messagesTurn(false, messages))
	messages = continueMessages(messages, answer, true)
	body := messagesTurn(false, messages)
	r, err := parseTranslated(provider.Anthropic, provider.Responses, []byte(body), false)
	if err != nil {
		t.Fatal(err)
	}
	full := buildResponses(r, "m1", "", false)
	askMessagesTurn(t, f.handler, body)
	// Compare parsed message content, not response-only IDs or JSON key order.
	parseInput := func(raw []byte) any {
		t.Helper()
		r, err := parseRequest(provider.Responses, raw, false, true)
		if err != nil {
			t.Fatal(err)
		}
		for i := range r.Messages {
			for j := range r.Messages[i].Parts {
				p := &r.Messages[i].Parts[j]
				if len(p.Args) > 0 {
					var args any
					if err := json.Unmarshal(p.Args, &args); err != nil {
						t.Fatal(err)
					}
					p.Args, _ = json.Marshal(args)
				}
			}
		}
		return r.Messages
	}
	if !reflect.DeepEqual(parseInput(full), parseInput([]byte(f.vendor.raw(1)))) {
		t.Fatal("restored vendor history differs from the full translated input")
	}
	if f.hop.body(1)["previous_response_id"] == nil {
		t.Fatal("equivalent history was sent whole instead of incrementally")
	}
}
