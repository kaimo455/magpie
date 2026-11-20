package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/access"
	"github.com/yetone/magpie/internal/provider"
)

func TestRemoteMessagesReplayIdentity(t *testing.T) {
	const message = `{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}`
	const call = `{"type":"function_call","call_id":"c1","name":"read_file","arguments":"{\"path\":\"a\",\"limit\":9007199254740993}"}`
	for _, tc := range []struct {
		name   string
		kept   string
		replay string
		equal  bool
	}{
		{"response metadata", `{"type":"message","id":"msg_1","status":"completed","phase":"final_answer","role":"assistant","content":[{"type":"output_text","text":"answer","annotations":[],"logprobs":[]}]}`, message, true},
		{"commentary metadata", `{"type":"message","phase":"commentary","role":"assistant","content":[{"type":"output_text","text":"answer"}]}`, message, true},
		{"edited text", message, strings.Replace(message, "answer", "edited", 1), false},
		{"argument formatting", call, `{"type":"function_call","status":"completed","id":"fc1","call_id":"c1","name":"read_file","arguments":"{ \"limit\":9007199254740993, \"path\":\"a\" }"}`, true},
		{"large integer changed", call, strings.Replace(call, "9007199254740993", "9007199254740992", 1), false},
		{"argument changed", call, strings.Replace(call, `\"a\"`, `\"b\"`, 1), false},
		{"call id changed", call, strings.Replace(call, "c1", "c2", 1), false},
		{"function changed", call, strings.Replace(call, "read_file", "delete_file", 1), false},
		{"citation lost", `{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer","annotations":[{"type":"url_citation","url":"https://example.com"}]}]}`, message, false},
		{"unknown phase", `{"type":"message","phase":"future_phase","role":"assistant","content":[{"type":"output_text","text":"answer"}]}`, message, false},
		{"unknown output", `{"type":"future_item","text":"answer"}`, message, false},
		{"summary only reasoning", `{"type":"reasoning","id":"rs1","summary":[{"type":"summary_text","text":"thinking"}]}`, `{"type":"reasoning","id":"rs1","summary":[{"type":"summary_text","text":"thinking"}]}`, false},
		{"sealed reasoning", `{"type":"reasoning","id":"rs1","encrypted_content":"sealed","summary":[]}`, `{"type":"reasoning","id":"rs1","encrypted_content":"sealed","summary":[]}`, true},
		{"sealed reasoning changed", `{"type":"reasoning","id":"rs1","encrypted_content":"sealed","summary":[]}`, `{"type":"reasoning","id":"rs1","encrypted_content":"edited","summary":[]}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kept, keptOK := remoteReplaySums([]json.RawMessage{json.RawMessage(tc.kept)})
			replay, replayOK := remoteReplaySums([]json.RawMessage{json.RawMessage(tc.replay)})
			equal := keptOK && replayOK && sumsEqual(kept, replay)
			if equal != tc.equal {
				t.Fatalf("replay equality = %v, want %v", equal, tc.equal)
			}
		})
	}
}

func readRemoteMessagesBody(
	t *testing.T,
	s *Server,
	ctx context.Context,
	p provider.Provider,
	body string,
) {
	t.Helper()
	res, err := s.forwardRemoteMessages(ctx, p, provider.Responses, "/responses", []byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = io.Copy(io.Discard, res.Body)
	closeErr := res.Body.Close()
	if err != nil || closeErr != nil || res.StatusCode != http.StatusOK {
		t.Fatalf("remote response: status=%d read=%v close=%v", res.StatusCode, err, closeErr)
	}
}

func TestRemoteMessagesDeltaRouteIsolation(t *testing.T) {
	for _, change := range []string{"scope", "provider", "key", "endpoint", "model", "unchanged"} {
		t.Run(change, func(t *testing.T) {
			hop, _ := fakeRemote(t, true, false)
			found, err := provider.Find("office")
			if err != nil {
				t.Fatal(err)
			}
			p := *found
			s := New()
			ctx := context.WithValue(context.Background(), remoteMessagesScope{}, "key:one")
			readRemoteMessagesBody(t, s, ctx, p, storeFalseTurn("m1", userAsks("HISTORY_MARKER")))
			model := "m1"
			switch change {
			case "scope":
				ctx = context.WithValue(ctx, remoteMessagesScope{}, "key:two")
			case "provider":
				p.ID = "another"
			case "key":
				p.Key = "another-key"
			case "endpoint":
				p.Responses += "/another"
			case "model":
				model = "m2"
			}
			readRemoteMessagesBody(t, s, ctx, p, storeFalseTurn(model, userAsks("HISTORY_MARKER"), userSaw("answer 1"), userAsks("next")))
			delta := hop.body(1)["previous_response_id"] != nil
			if delta != (change == "unchanged") {
				t.Fatalf("delta after %s = %v", change, delta)
			}
		})
	}
}

// The scope must come from the incoming Messages request, not a synthetic
// Responses request that has forgotten its key, hop or remote address.
func TestRemoteMessagesDeltaCallerIsolation(t *testing.T) {
	for _, change := range []string{"key", "via", "remote address", "unchanged"} {
		t.Run(change, func(t *testing.T) {
			f := messagesRemote(t, true, false, false)
			key, via, address := "one", "one", "192.0.2.1:5000"
			if change == "remote address" {
				key = ""
			}
			h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				r = r.WithContext(access.WithIdentity(r.Context(), access.Identity{KeyID: key, KeyName: key}))
				r.RemoteAddr = address
				r.Header.Set("User-Agent", "magpie/test")
				r.Header.Set(AgentHeader, "claude")
				r.Header.Set(ViaHeader, via)
				f.handler.ServeHTTP(w, r)
			})
			messages := initialMessages()
			answer := askMessagesTurn(t, h, messagesTurn(true, messages))
			switch change {
			case "key":
				key = "two"
			case "via":
				via = "two"
			case "remote address":
				address = "192.0.2.2:5000"
			}
			messages = continueMessages(messages, answer, false)
			askMessagesTurn(t, h, messagesTurn(true, messages))
			delta := f.hop.body(1)["previous_response_id"] != nil
			if delta != (change == "unchanged") {
				t.Fatalf("delta for another %s = %v", change, delta)
			}
		})
	}
}

func TestRemoteMessagesDeltaLeavesOtherRoutesAlone(t *testing.T) {
	for _, route := range []string{"unmarked context", "non-remote", "chat", "explicit previous response"} {
		t.Run(route, func(t *testing.T) {
			hop, _ := fakeRemote(t, true, false)
			found, err := provider.Find("office")
			if err != nil {
				t.Fatal(err)
			}
			p := *found
			ctx := context.WithValue(context.Background(), remoteMessagesScope{}, "key:one")
			to, path := provider.Responses, "/responses"
			body := storeFalseTurn("m1", userAsks("hi"))
			switch route {
			case "unmarked context":
				ctx = context.Background()
			case "non-remote":
				p.Preset = ""
			case "chat":
				to, path = provider.Chat, "/chat/completions"
				body = `{"model":"m1","messages":[{"role":"user","content":"hi"}]}`
			case "explicit previous response":
				body = `{"model":"m1","previous_response_id":"resp_client","input":[{"role":"user","content":"hi"}]}`
			}
			res, err := New().forwardRemoteMessages(ctx, p, to, path, []byte(body), nil)
			if err != nil {
				t.Fatal(err)
			}
			io.Copy(io.Discard, res.Body)
			res.Body.Close()
			if hop.n() != 1 || hop.header(0).Get(KeepTurnHeader) != "" || hop.raw(0) != body {
				t.Fatal("an unrelated route was asked kept or rewritten")
			}
		})
	}
}

func TestRemoteMessagesDeltaOnlyFinishedBody(t *testing.T) {
	for _, state := range []string{"complete", "truncated", "closed early", "cancelled", "unsupported output"} {
		t.Run(state, func(t *testing.T) {
			freshRemoteTurns(t)
			t.Setenv("XDG_CONFIG_HOME", t.TempDir())
			t.Setenv("XDG_CACHE_HOME", t.TempDir())
			hop := &hopLog{}
			remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hop.add(r)
				body := remoteMessagesReply(hop.n(), false)
				if hop.n() == 1 {
					switch state {
					case "truncated":
						body = body[:strings.LastIndex(body, "data: ")]
					case "unsupported output":
						body = strings.ReplaceAll(body, `"annotations":[]`, `"annotations":[{"type":"url_citation","url":"https://example.com"}]`)
					}
				}
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set(KeptTurnHeader, "1")
				io.WriteString(w, body)
			}))
			t.Cleanup(remote.Close)
			p := provider.Provider{ID: "office", Preset: provider.RemoteMagpiePreset, Key: "rk", Responses: remote.URL + "/v1"}
			ctx := context.WithValue(context.Background(), remoteMessagesScope{}, "key:one")
			first, cancel := context.WithCancel(ctx)
			defer cancel()
			s := New()
			res, err := s.forwardRemoteMessages(first, p, provider.Responses, "/responses", []byte(storeFalseTurn("m1", userAsks("HISTORY_MARKER"))), nil)
			if err != nil {
				t.Fatal(err)
			}
			if state == "closed early" || state == "cancelled" {
				var firstByte [1]byte
				if _, err := io.ReadFull(res.Body, firstByte[:]); err != nil {
					t.Fatal(err)
				}
				if state == "cancelled" {
					cancel()
				}
			} else if _, err := io.Copy(io.Discard, res.Body); err != nil {
				t.Fatal(err)
			}
			res.Body.Close()
			// A duplicate Close must neither retain twice nor double-discard.
			res.Body.Close()
			readRemoteMessagesBody(t, s, ctx, p, storeFalseTurn("m1", userAsks("HISTORY_MARKER"), userSaw("answer 1"), userAsks("next")))
			delta := hop.body(1)["previous_response_id"] != nil
			if delta != (state == "complete") {
				t.Fatalf("delta after %s body = %v", state, delta)
			}
		})
	}
}

// Hold the upstream's terminal event until the Messages client has actually
// received text. Whole-reply buffering cannot pass this check.
func TestRemoteMessagesDeltaDoesNotBufferStream(t *testing.T) {
	freshRemoteTurns(t)
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	gate := make(chan struct{})
	var release sync.Once
	defer release.Do(func() { close(gate) })
	var completed atomic.Bool
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		body := remoteMessagesReply(1, false)
		at := strings.LastIndex(body, "data: ")
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set(KeptTurnHeader, "1")
		io.WriteString(w, body[:at])
		w.(http.Flusher).Flush()
		select {
		case <-gate:
			completed.Store(true)
			io.WriteString(w, body[at:])
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(remote.Close)
	if err := provider.Save(provider.Provider{ID: "office", Preset: provider.RemoteMagpiePreset, Key: "rk", Models: []string{"fake/m1"}, Responses: remote.URL + "/v1"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelAPI("office/fake/m1", "responses"); err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(New().Handler())
	t.Cleanup(local.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, local.URL+"/v1/messages", strings.NewReader(messagesTurn(true, initialMessages())))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	seen := false
	dec := decoder(provider.Anthropic)
	err = readSSE(res.Body, func(_, data string) error {
		return dec(data, func(ev Event) {
			if ev.Kind == KText && ev.Text != "" && !seen {
				seen = true
				if completed.Load() {
					t.Error("first text arrived only after the upstream completed")
				}
				release.Do(func() { close(gate) })
			}
		})
	})
	if err != nil || res.StatusCode != http.StatusOK || !seen {
		t.Fatalf("stream: status=%d saw text=%v err=%v", res.StatusCode, seen, err)
	}
}
