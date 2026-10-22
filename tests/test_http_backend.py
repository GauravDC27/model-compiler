import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dmc.http_backend import OpenAICompatBackend

# A fake completions server: every character is one token with log-prob -0.1, except the
# continuation "no", whose characters get -0.01 (so "no" scores higher than "yes").


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["prompt"]
        if body.get("echo"):
            tokens = list(text) + ["!"]
            prompt_end = text.rfind(":") + 1
            lps = [None] + [(-0.01 if i >= prompt_end and text.endswith("no") else -0.1)
                            for i in range(1, len(text))] + [-1.0]
            payload = {"choices": [{"text": text + "!", "logprobs": {
                "tokens": tokens, "token_logprobs": lps, "text_offset": list(range(len(tokens)))}}]}
        else:
            payload = {"choices": [{"text": "greedy" if body.get("temperature") == 0 else f"seed{body['seed']}"}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_score_sums_only_continuation_tokens(server):
    b = OpenAICompatBackend(server, "m")
    yes, no = b.score("Q:", ["yes", "no"])
    assert yes == pytest.approx(3 * -0.1)
    assert no == pytest.approx(2 * -0.01)


def test_generate_greedy_and_seeded(server):
    b = OpenAICompatBackend(server, "m")
    assert b.generate("hi", 5) == "greedy"
    assert b.generate("hi", 5, seed=7) == "seed7"


def test_render_requires_a_renderer(server):
    with pytest.raises(ValueError):
        OpenAICompatBackend(server, "m").render_chat("s", "u")
