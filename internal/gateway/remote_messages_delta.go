package gateway

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
)

// remoteMessagesScope carries the original caller's scope through translation.
type remoteMessagesScope struct{}

// forwardRemoteMessages keeps a translated Responses turn on another magpie.
// Other destinations and translations retain their normal forwarding path.
func (s *Server) forwardRemoteMessages(ctx context.Context, p provider.Provider, to provider.Protocol, path string, body []byte, in http.Header) (*http.Response, error) {
	scope, enabled := ctx.Value(remoteMessagesScope{}).(string)
	if !enabled || to != provider.Responses || !p.IsRemoteMagpie() {
		return s.forward(ctx, p, to, path, body, in)
	}
	remote := remoteTurnForScope(p, body, scope)
	if remote == nil {
		return s.forward(ctx, p, to, path, body, in)
	}
	// Native Responses output marks are not a Messages replay's identity.
	remote.ctx = ctx
	remote.strict = true
	remote.key += "\x00messages\x00" + p.Base(to) + "\x00" + gjson.GetBytes(body, "model").String()
	ctx = withKeepTurn(ctx)
	wire, delta := remote.delta(body)
	res, err := s.forward(ctx, p, to, path, wire, in)
	if err != nil {
		return res, err
	}
	if delta && (badRequest(res.StatusCode) || res.StatusCode == http.StatusNotFound) {
		b, _ := io.ReadAll(io.LimitReader(res.Body, 1<<20))
		res.Body.Close()
		res.Body = io.NopCloser(bytes.NewReader(b))
		if prevRefused(res.StatusCode, b) {
			remote.refused()
			res, err = s.forward(ctx, p, to, path, body, in)
			if err != nil {
				return res, err
			}
		}
	}
	if res.StatusCode < 400 && res.Header.Get(KeptTurnHeader) != "" {
		tracked, finish := remote.tee(res.Body, res.Header.Get("Content-Type"))
		res.Body = &remoteMessagesBody{ReadCloser: tracked, finish: finish}
	}
	return res, nil
}

type remoteMessagesBody struct {
	io.ReadCloser
	finish func()
	once   sync.Once
}

func (b *remoteMessagesBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(b.finish)
	return err
}

// remoteReplaySums accepts only items a Messages client can return unchanged
// in meaning. Unknown output and nonempty annotations keep the turn whole.
func remoteReplaySums(items []json.RawMessage) ([]itemSum, bool) {
	if len(items) == 0 {
		return nil, false
	}
	sums := make([]itemSum, len(items))
	for i, item := range items {
		var v map[string]any
		if decodeNumbers(item, &v) != nil || v == nil {
			return nil, false
		}
		switch v["type"] {
		case "message":
			if v["role"] != "assistant" {
				return nil, false
			}
			parts, ok := v["content"].([]any)
			if !ok || len(parts) == 0 {
				return nil, false
			}
			for _, part := range parts {
				p, ok := part.(map[string]any)
				if !ok || p["type"] != "output_text" {
					return nil, false
				}
				for _, field := range []string{"annotations", "logprobs"} {
					if value, exists := p[field]; exists {
						list, ok := value.([]any)
						if !ok || len(list) != 0 {
							return nil, false
						}
						delete(p, field)
					}
				}
			}
			// phaseTidy labels Responses messages for clients such as Codex;
			// Messages has no corresponding field to replay. Keep unknown
			// phases conservative rather than guessing their meaning.
			if phase, exists := v["phase"]; exists && phase != phaseCommentary && phase != phaseFinal {
				return nil, false
			}
			delete(v, "id")
			delete(v, "status")
			delete(v, "phase")
		case "function_call":
			args, ok := v["arguments"].(string)
			if !ok {
				return nil, false
			}
			var value any
			if !json.Valid([]byte(args)) || decodeNumbers([]byte(args), &value) != nil {
				return nil, false
			}
			canonical, err := json.Marshal(value)
			if err != nil {
				return nil, false
			}
			v["arguments"] = string(canonical)
			delete(v, "id")
			delete(v, "status")
		case "reasoning":
			if encrypted, ok := v["encrypted_content"].(string); !ok || encrypted == "" {
				return nil, false
			}
		default:
			return nil, false
		}
		canonical, err := json.Marshal(v)
		if err != nil {
			return nil, false
		}
		sums[i] = sha256.Sum256(canonical)
	}
	return sums, true
}

// remoteSSECompleted reports whether reply is an SSE stream with a complete
// response.completed event (streamTranslated closes res.Body on that event
// without waiting for EOF, #5936).
func remoteSSECompleted(ctype string, reply []byte) bool {
	if !strings.Contains(ctype, "event-stream") && !bytes.HasPrefix(reply, []byte("event:")) && !bytes.HasPrefix(reply, []byte("data:")) {
		return false
	}
	done := false
	readSSE(bytes.NewReader(reply), func(event, data string) error {
		t := gjson.Get(data, "type").String()
		if t == "" {
			t = event
		}
		if t == "response.completed" && gjson.Valid(data) {
			done = true
		}
		return nil
	})
	return done
}
