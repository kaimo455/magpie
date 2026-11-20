package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
)

// hopLog records what a server was asked: each request's body and headers.
type hopLog struct {
	mu      sync.Mutex
	bodies  [][]byte // as the other magpie reads them, decoded
	wires   [][]byte // as they went
	headers []http.Header
}

// add logs r's body, and gives it back as it went.
func (l *hopLog) add(r *http.Request) []byte {
	b, _ := io.ReadAll(r.Body)
	l.mu.Lock()
	defer l.mu.Unlock()
	l.bodies = append(l.bodies, wireDecoded(r.Header, b))
	l.wires = append(l.wires, b)
	l.headers = append(l.headers, r.Header.Clone())
	return b
}

func (l *hopLog) wire(i int) []byte {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.wires[i]
}

func (l *hopLog) n() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.bodies)
}

func (l *hopLog) body(i int) map[string]any {
	l.mu.Lock()
	defer l.mu.Unlock()
	var q map[string]any
	json.Unmarshal(l.bodies[i], &q)
	return q
}

func (l *hopLog) raw(i int) string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return string(l.bodies[i])
}

func (l *hopLog) header(i int) http.Header {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.headers[i]
}

func freshRemoteTurns(t *testing.T) {
	old, oldPrev := remoteTurns, prevResponses
	remoteTurns, prevResponses = newRemoteSentStore(), newPrevStore(prevCap, prevTTL)
	t.Cleanup(func() { remoteTurns, prevResponses = old, oldPrev })
}

// askAs posts a Responses request to h as the agent with ua.
func askHop(t *testing.T, h http.Handler, ua, body string, header map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	req.RemoteAddr = "127.0.0.1:5555"
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", ua)
	for k, v := range header {
		req.Header.Set(k, v)
	}
	h.ServeHTTP(rec, req)
	return rec
}

var bigImage = `{"type":"message","role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,` + strings.Repeat("QUFB", 50000) + `"},{"type":"input_text","text":"what is this?"}]}`

// codexTurn is a Responses request as Codex sends it: store false, the
// conversation whole, its items without their ids.
func storeFalseTurn(model string, items ...string) string {
	return `{"model":"` + model + `","store":false,"input":[` + strings.Join(items, ",") + `]}`
}

func userSaw(text string) string {
	return `{"type":"message","role":"assistant","content":[{"type":"output_text","text":"` + text + `"}]}`
}

func userAsks(text string) string {
	return `{"type":"message","role":"user","content":[{"type":"input_text","text":"` + text + `"}]}`
}

// Two magpies, each the other's provider (LotusDecoder on X): Codex says
// store false and sends the conversation whole, and the leg between the
// magpies carried all of it each turn. Now the second turn goes to the
// other magpie as its new input alone after the id it kept the first
// under, and the vendor behind it is still sent the whole conversation.
// Here one gateway is both computers', "office" its own address.
func TestRemoteMagpieSentTheNewInputAfterAKeptTurn(t *testing.T) {
	freshRemoteTurns(t)
	vendor, hop := &hopLog{}, &hopLog{}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		vendor.add(r)
		n := vendor.n()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id":"resp_v%d","object":"response","model":"m1","status":"completed","output":[{"type":"message","id":"msg_v%d","role":"assistant","content":[{"type":"output_text","text":"answer %d"}]}],"usage":{"input_tokens":3,"output_tokens":1}}`, n, n, n)
	}))
	t.Cleanup(up.Close)
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	if err := provider.Save(provider.Provider{ID: "fake", Name: "Fake", Key: "k", Models: []string{"m1"}, Responses: up.URL + "/v1"}); err != nil {
		t.Fatal(err)
	}
	h := New().Handler()
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			b := hop.add(r)
			r.Body = io.NopCloser(strings.NewReader(string(b)))
		}
		h.ServeHTTP(w, r)
	}))
	t.Cleanup(remote.Close)
	if err := provider.Save(provider.Provider{ID: "office", Name: "Office", Preset: provider.RemoteMagpiePreset, Key: "rk", Models: []string{"fake/m1"},
		Chat: remote.URL + "/v1", Responses: remote.URL + "/v1", Anthropic: remote.URL}); err != nil {
		t.Fatal(err)
	}
	const ua = "codex_cli_rs/0.50.0"
	if rec := askHop(t, h, ua, storeFalseTurn("office/fake/m1", bigImage), nil); rec.Code != 200 || !strings.Contains(rec.Body.String(), "answer 1") {
		t.Fatalf("turn 1: %d %s", rec.Code, rec.Body.String())
	}
	if !strings.HasPrefix(hop.header(0).Get(KeepTurnHeader), "magpie/") {
		t.Fatalf("turn 1 not asked kept: %v", hop.header(0))
	}
	if rec := askHop(t, h, ua, storeFalseTurn("office/fake/m1", bigImage, userSaw("answer 1"), userAsks("and now?")), nil); rec.Code != 200 || !strings.Contains(rec.Body.String(), "answer 2") {
		t.Fatalf("turn 2: %d %s", rec.Code, rec.Body.String())
	}
	if hop.n() != 2 {
		t.Fatalf("the other magpie asked %d times", hop.n())
	}
	q := hop.body(1)
	in, _ := json.Marshal(q["input"])
	if q["previous_response_id"] == nil || strings.Contains(string(in), "QUFB") || !strings.Contains(string(in), "and now?") || len(hop.raw(1)) > 4000 || q["store"] != false {
		t.Fatalf("turn 2 to the other magpie, %d bytes: prev %v store %v input %.300s", len(hop.raw(1)), q["previous_response_id"], q["store"], in)
	}
	// the vendor still sees the whole conversation, store false as asked
	v := vendor.raw(1)
	for _, want := range []string{"QUFB", "answer 1", "and now?"} {
		if !strings.Contains(v, want) {
			t.Fatalf("vendor's turn 2 went without %q: %.300s", want, v)
		}
	}
	if vq := vendor.body(1); vq["previous_response_id"] != nil || vq["store"] != false {
		t.Fatalf("vendor asked with prev %v store %v", vq["previous_response_id"], vq["store"])
	}
	// and a third turn follows the second the same way
	if rec := askHop(t, h, ua, storeFalseTurn("office/fake/m1", bigImage, userSaw("answer 1"), userAsks("and now?"), userSaw("answer 2"), userAsks("third")), nil); rec.Code != 200 {
		t.Fatalf("turn 3: %d %s", rec.Code, rec.Body.String())
	}
	if q := hop.body(2); q["previous_response_id"] == nil || strings.Contains(hop.raw(2), "QUFB") {
		t.Fatalf("turn 3 to the other magpie: %.300s", hop.raw(2))
	}
	for _, want := range []string{"QUFB", "answer 1", "and now?", "answer 2", "third"} {
		if !strings.Contains(vendor.raw(2), want) {
			t.Fatalf("vendor's turn 3 went without %q", want)
		}
	}
	// no vendor is ever sent the header
	for i := range vendor.n() {
		for k := range vendor.header(i) {
			if strings.HasPrefix(strings.ToLower(k), "x-magpie-") {
				t.Fatalf("vendor sent %s", k)
			}
		}
	}
}

// fakeRemote answers as a magpie would on /v1/responses: keeps says it
// kept the turn, and lost forgets the first reply as a restart would.
func fakeRemote(t *testing.T, keeps, lost bool) (*hopLog, http.Handler) {
	t.Helper()
	freshRemoteTurns(t)
	hop := &hopLog{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hop.add(r)
		n := hop.n()
		w.Header().Set("Content-Type", "application/json")
		if keeps && r.Header.Get(KeepTurnHeader) != "" {
			w.Header().Set(KeptTurnHeader, "1")
		}
		if lost && hop.body(n - 1)["previous_response_id"] == "resp_r1" {
			w.WriteHeader(400)
			io.WriteString(w, `{"error":{"message":"Previous response with id 'resp_r1' not found.","type":"invalid_request_error","param":"previous_response_id","code":"previous_response_not_found"}}`)
			return
		}
		fmt.Fprintf(w, `{"id":"resp_r%d","object":"response","model":"m1","status":"completed","output":[{"type":"message","id":"msg_r%d","role":"assistant","content":[{"type":"output_text","text":"answer %d"}]}],"usage":{"input_tokens":3,"output_tokens":1}}`, n, n, n)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	if err := provider.Save(provider.Provider{ID: "office", Name: "Office", Preset: provider.RemoteMagpiePreset, Key: "rk", Models: []string{"m1"},
		Chat: srv.URL + "/v1", Responses: srv.URL + "/v1", Anthropic: srv.URL}); err != nil {
		t.Fatal(err)
	}
	return hop, New().Handler()
}

// A magpie that restarted loses the first reply. Once the full retry
// succeeds and is acknowledged, the next turn continues that new reply.
func TestRemoteMagpieLostTurnSentWhole(t *testing.T) {
	hop, h := fakeRemote(t, true, true)
	const ua = "codex_cli_rs/0.50.0"
	askHop(t, h, ua, storeFalseTurn("office/m1", bigImage), nil)
	rec := askHop(t, h, ua, storeFalseTurn("office/m1", bigImage, userSaw("answer 1"), userAsks("and now?")), nil)
	if rec.Code != 200 {
		t.Fatalf("turn 2: %d %s", rec.Code, rec.Body.String())
	}
	if hop.n() != 3 || hop.body(1)["previous_response_id"] != "resp_r1" {
		t.Fatalf("turn 2 tried as a delta first: %d asks, %.200s", hop.n(), hop.raw(min(1, hop.n()-1)))
	}
	if q := hop.body(2); q["previous_response_id"] != nil || !strings.Contains(hop.raw(2), "QUFB") || !strings.Contains(hop.raw(2), "and now?") {
		t.Fatalf("turn 2 not sent again whole: %.300s", hop.raw(2))
	}
	rec = askHop(t, h, ua, storeFalseTurn("office/m1", bigImage, userSaw("answer 1"), userAsks("and now?"), userSaw("answer 3"), userAsks("third")), nil)
	if rec.Code != 200 || hop.n() != 4 {
		t.Fatalf("turn 3: %d, %d asks", rec.Code, hop.n())
	}
	if q := hop.body(3); q["previous_response_id"] != "resp_r3" || strings.Contains(hop.raw(3), "QUFB") {
		t.Fatalf("turn 3 did not continue the newly kept reply: %.300s", hop.raw(3))
	}
}

// A replay can match a longer branch kept before the receiver restarted
// and a shorter one kept anew. Only the usable acknowledgment may win,
// while another conversation's pre-refusal acknowledgment stays held.
func TestRemoteMagpieNewTurnBeforeStaleLongerHistory(t *testing.T) {
	freshRemoteTurns(t)
	now := time.Unix(100, 0)
	remoteTurns.now = func() time.Time { return now }
	p := provider.Provider{ID: "office", Preset: provider.RemoteMagpiePreset, Key: "rk"}
	body := []byte(storeFalseTurn(
		"m1",
		userAsks("first question"),
		userSaw("first answer"),
		userAsks("second question"),
		userSaw("second answer"),
		userAsks("third question"),
	))
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	turn := remoteTurnFor(req, p, body)
	if turn == nil {
		t.Fatal("remote turn was not recognized")
	}
	otherBody := []byte(storeFalseTurn(
		"m1",
		userAsks("other question"),
		userSaw("other answer"),
		userAsks("other follow-up"),
	))
	otherTurn := remoteTurnFor(req, p, otherBody)
	if otherTurn == nil {
		t.Fatal("other remote turn was not recognized")
	}
	remoteTurns.put(
		&remoteSent{
			key:    turn.key,
			input:  turn.sums[:3],
			output: []itemMark{markOf([]byte(userSaw("second answer")))},
			id:     "resp_old_branch",
		},
		nil,
	)
	remoteTurns.put(
		&remoteSent{
			key:    otherTurn.key,
			input:  otherTurn.sums[:1],
			output: []itemMark{markOf([]byte(userSaw("other answer")))},
			id:     "resp_other_stale",
		},
		nil,
	)
	now = now.Add(time.Second)
	remoteTurns.refuse(turn.route)
	now = now.Add(time.Second)
	remoteTurns.put(
		&remoteSent{
			key:    turn.key,
			input:  turn.sums[:1],
			output: []itemMark{markOf([]byte(userSaw("first answer")))},
			id:     "resp_new_branch",
		},
		nil,
	)
	wire, delta := turn.delta(body)
	var request map[string]any
	if !delta || json.Unmarshal(wire, &request) != nil || request["previous_response_id"] != "resp_new_branch" {
		t.Fatal("a stale longer history hid the newly acknowledged response")
	}
	input, _ := request["input"].([]any)
	if len(input) != 3 || strings.Contains(string(wire), "first question") || !strings.Contains(string(wire), "second answer") {
		t.Fatal("the delta did not carry the input after the usable response")
	}
	if _, otherDelta := otherTurn.delta(otherBody); otherDelta {
		t.Fatal("a second conversation's pre-refusal acknowledgment was not held during remoteRefusedWait")
	}
}

// A magpie from before this never says it kept a turn: it is never sent
// a delta, which it might relay to a vendor that ignores the id.
func TestOlderRemoteMagpieSentWhole(t *testing.T) {
	hop, h := fakeRemote(t, false, false)
	const ua = "codex_cli_rs/0.50.0"
	askHop(t, h, ua, storeFalseTurn("office/m1", bigImage), nil)
	askHop(t, h, ua, storeFalseTurn("office/m1", bigImage, userSaw("answer 1"), userAsks("and now?")), nil)
	if hop.n() != 2 {
		t.Fatalf("%d asks", hop.n())
	}
	if q := hop.body(1); q["previous_response_id"] != nil || !strings.Contains(hop.raw(1), "QUFB") {
		t.Fatalf("turn 2 sent as a delta: %.300s", hop.raw(1))
	}
}

// A history that isn't the last turn sent and its answer goes whole: an
// earlier item edited, the answer left out, or another answer in its place.
func TestRemoteMagpieEditedHistorySentWhole(t *testing.T) {
	for _, c := range []struct {
		name  string
		items []string
	}{
		{"edited", []string{userAsks("what is that?"), userSaw("answer 1"), userAsks("and now?")}},
		{"answer left out", []string{bigImage, userAsks("and now?")}},
		{"other answer", []string{bigImage, `{"type":"function_call","call_id":"c1","name":"ls","arguments":"{}"}`, userAsks("and now?")}},
		{"nothing new", []string{bigImage, userSaw("answer 1")}},
	} {
		t.Run(c.name, func(t *testing.T) {
			hop, h := fakeRemote(t, true, false)
			const ua = "codex_cli_rs/0.50.0"
			askHop(t, h, ua, storeFalseTurn("office/m1", bigImage), nil)
			askHop(t, h, ua, storeFalseTurn("office/m1", c.items...), nil)
			if hop.n() != 2 {
				t.Fatalf("%d asks", hop.n())
			}
			// sent whole: no id, and every item there (a lone call may be
			// answered for the vendor on the way, so at least those)
			if q := hop.body(1); q["previous_response_id"] != nil || len(q["input"].([]any)) < len(c.items) {
				t.Fatalf("sent as a delta: %.300s", hop.raw(1))
			}
		})
	}
	// and the control: the same history continued is a delta
	hop, h := fakeRemote(t, true, false)
	askHop(t, h, "codex_cli_rs/0.50.0", storeFalseTurn("office/m1", bigImage), nil)
	askHop(t, h, "codex_cli_rs/0.50.0", storeFalseTurn("office/m1", bigImage, userSaw("answer 1"), userAsks("and now?")), nil)
	if hop.body(1)["previous_response_id"] != "resp_r1" {
		t.Fatalf("control not a delta: %.300s", hop.raw(1))
	}
}

// Only another magpie can ask its store false turn kept: the header from
// any other client is ignored, nothing is kept, and an id it names later
// isn't there. A magpie asking it gets the turn kept and is told so; one
// naming an id not held is answered previous_response_not_found before
// any vendor is asked, rather than the id relayed to one that may ignore
// it.
func TestKeepTurnOnlyFromAMagpie(t *testing.T) {
	freshRemoteTurns(t)
	vendor := &hopLog{}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		vendor.add(r)
		n := vendor.n()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id":"resp_v%d","object":"response","model":"m1","status":"completed","output":[{"type":"message","id":"msg_v%d","role":"assistant","content":[{"type":"output_text","text":"answer %d"}]}],"usage":{"input_tokens":3,"output_tokens":1}}`, n, n, n)
	}))
	t.Cleanup(up.Close)
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	// the agent's own headers passed on, the keep header listed too: it
	// is gone before anything could pass it
	if err := provider.Save(provider.Provider{ID: "fake", Name: "Fake", Key: "k", Models: []string{"m1"}, Responses: up.URL + "/v1", PassHeaders: []string{KeepTurnHeader, KeptTurnHeader}}); err != nil {
		t.Fatal(err)
	}
	h := New().Handler()
	// plain clients: the header as a magpie writes it, or another value
	// from Codex as a magpie's hop carries it on
	for _, c := range []struct{ ua, keep string }{
		{"curl/8.0", "magpie/0.1.2300"},
		{"pi/1.0", "magpie/0.1.2300"},
		{"codex_cli_rs/0.50.0", "1"},
		{"magpie/0.1.2300", "1"},
	} {
		rec := askHop(t, h, c.ua, storeFalseTurn("fake/m1", userAsks("my name is Ada")), map[string]string{KeepTurnHeader: c.keep})
		if rec.Code != 200 || rec.Header().Get(KeptTurnHeader) != "" {
			t.Fatalf("%s: %d, told kept %q", c.ua, rec.Code, rec.Header().Get(KeptTurnHeader))
		}
		if prevResponses.size != 0 {
			t.Fatalf("%d bytes kept for %s", prevResponses.size, c.ua)
		}
		before := vendor.n()
		rec = askHop(t, h, c.ua, `{"model":"fake/m1","store":false,"previous_response_id":"resp_v`+fmt.Sprint(before)+`","input":[`+userAsks("what is my name?")+`]}`, nil)
		if vendor.n() > before && strings.Contains(vendor.raw(before), "my name is Ada") {
			t.Fatalf("%s's turn was kept and given back: %d", c.ua, rec.Code)
		}
	}

	// another magpie
	magpie := map[string]string{KeepTurnHeader: "magpie/0.1.2300", AgentHeader: "codex", ViaHeader: "studio"}
	n0 := vendor.n()
	rec := askHop(t, h, "magpie/0.1.2300", storeFalseTurn("fake/m1", userAsks("my name is Ada")), magpie)
	if rec.Code != 200 || rec.Header().Get(KeptTurnHeader) != "1" {
		t.Fatalf("magpie: %d, told kept %q", rec.Code, rec.Header().Get(KeptTurnHeader))
	}
	rec = askHop(t, h, "magpie/0.1.2300", `{"model":"fake/m1","store":false,"previous_response_id":"resp_v`+fmt.Sprint(n0+1)+`","input":[`+userAsks("what is my name?")+`]}`, magpie)
	if rec.Code != 200 {
		t.Fatalf("magpie's delta: %d %s", rec.Code, rec.Body.String())
	}
	if v := vendor.raw(n0 + 1); !strings.Contains(v, "my name is Ada") || !strings.Contains(v, "answer "+fmt.Sprint(n0+1)) || strings.Contains(v, "previous_response_id") {
		t.Fatalf("vendor not given the conversation whole: %s", v)
	}
	// an id not held, from a magpie asking its turns kept: not relayed
	n := vendor.n()
	rec = askHop(t, h, "magpie/0.1.2300", `{"model":"fake/m1","store":false,"previous_response_id":"resp_gone","input":[`+userAsks("hi")+`]}`, magpie)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "previous_response_not_found") || vendor.n() != n {
		t.Fatalf("lost id: %d %s, vendor asked %d times", rec.Code, rec.Body.String(), vendor.n()-n)
	}
	// the header never reached the vendor, from either
	for i := range vendor.n() {
		if vendor.header(i).Get(KeepTurnHeader) != "" || vendor.header(i).Get(KeptTurnHeader) != "" {
			t.Fatalf("vendor sent the header: %v", vendor.header(i))
		}
	}
}
