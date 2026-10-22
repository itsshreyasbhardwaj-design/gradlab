"""The Microscope's CLI and its local server — including the security boundary.

The server is the one component that accepts input from outside the process, so
its refusals are tested as carefully as its successes.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from gradlab.inspection import cli, presets, server


class TestCLI:
    def test_trace_prints_the_forward_and_backward_passes(self, capsys):
        assert cli.main(["trace", "x^2 + 2*x", "--at", "x=3"]) == 0
        out = capsys.readouterr().out
        assert "forward pass" in out and "backward pass" in out
        assert "15.0" in out  # the value
        assert "8.0" in out  # the gradient

    def test_trace_writes_a_json_file(self, tmp_path):
        target = tmp_path / "t.json"
        assert cli.main(["trace", "tanh(x)", "--at", "x=0.5", "--json", str(target)]) == 0
        payload = json.loads(target.read_text())
        assert payload["expression_text"] == "tanh(x)"
        assert payload["num_nodes"] == 1

    def test_trace_reports_a_bad_expression_without_a_traceback(self, capsys):
        assert cli.main(["trace", "foo(x)"]) == 2
        assert "Unknown function" in capsys.readouterr().err

    def test_trace_reports_a_malformed_at_argument(self, capsys):
        assert cli.main(["trace", "x", "--at", "nonsense"]) == 2
        assert "name=value" in capsys.readouterr().err

    def test_generate_fast_writes_an_index_and_every_listed_file(self, tmp_path, capsys):
        assert cli.main(["generate", "--out", str(tmp_path), "--fast"]) == 0
        index = json.loads((tmp_path / "index.json").read_text())
        assert index["traces"], "the index must not be empty"
        for entry in index["traces"]:
            assert (tmp_path / entry["file"]).exists(), entry["file"]
            payload = json.loads((tmp_path / entry["file"]).read_text())
            assert payload["schema_version"] == index["schema_version"]

    def test_generated_expression_traces_carry_real_gradients(self, tmp_path):
        cli.main(["generate", "--out", str(tmp_path), "--fast"])
        quadratic = json.loads((tmp_path / "quadratic.json").read_text())
        assert quadratic["output"]["value"]["values"] == [15.0]
        assert quadratic["leaf_gradients"][0]["gradient"]["values"] == [8.0]


class TestPresets:
    def test_every_expression_preset_executes(self):
        traces = presets.build_expression_traces()
        assert len(traces) == len(presets.EXPRESSION_PRESETS)
        for slug, trace in traces.items():
            assert trace["backward_steps"], f"{slug} produced no backward steps"
            assert trace["leaf_gradients"], f"{slug} produced no gradients"
            json.dumps(trace)

    def test_the_broadcast_preset_teaches_what_it_claims(self):
        trace = presets.build_expression_traces()["broadcast"]
        bias = next(g for g in trace["leaf_gradients"] if g["name"] == "b")
        assert bias["gradient"]["values"] == [4.0, 4.0, 4.0]

    def test_the_relu_preset_shows_a_half_dead_gradient(self):
        trace = presets.build_expression_traces()["relu-gate"]
        gradient = trace["leaf_gradients"][0]["gradient"]["values"]
        assert gradient[:2] == [0.0, 0.0] and all(v > 0 for v in gradient[2:])

    def test_a_training_step_preset_really_updates_parameters(self):
        trace = presets.build_training_step_trace()
        for update in trace["parameter_updates"]:
            assert update["before"]["values"] != update["after"]["values"]


@pytest.fixture
def running_server(tmp_path):
    """A server bound to an ephemeral port, serving a minimal app directory."""
    (tmp_path / "index.html").write_text("<!doctype html><title>t</title>")
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "index.json").write_text('{"traces": []}')
    (tmp_path / "secret.txt").write_text("should never be served")

    handler = type("H", (server._Handler,), {"directory_root": tmp_path, "allow_expressions": True})
    httpd = server._Server(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base, tmp_path
    finally:
        httpd.shutdown()
        httpd.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read()


def post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestServer:
    def test_serves_the_index(self, running_server):
        base, _ = running_server
        status, body = get(f"{base}/")
        assert status == 200 and b"<!doctype html>" in body

    def test_reports_its_capabilities_and_limits(self, running_server):
        base, _ = running_server
        status, body = get(f"{base}/api/capabilities")
        payload = json.loads(body)
        assert status == 200 and payload["allow_expressions"] is True
        assert payload["limits"]["max_nodes"] > 0

    def test_refuses_an_extension_outside_the_allowlist(self, running_server):
        base, _ = running_server
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(f"{base}/secret.txt")
        assert exc.value.code == 404

    @pytest.mark.parametrize(
        "attempt",
        [
            "/../../../etc/passwd",
            "/..%2f..%2fetc%2fpasswd",
            "/traces/../../etc/passwd",
            "//etc/passwd",
        ],
    )
    def test_path_traversal_is_refused(self, running_server, attempt):
        base, _ = running_server
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(f"{base}{attempt}")
        assert exc.value.code == 404

    def test_evaluates_an_allowed_expression(self, running_server):
        base, _ = running_server
        status, payload = post(f"{base}/api/trace", {"expression": "x^2 + 2*x", "values": {"x": 3}})
        assert status == 200
        assert payload["output"]["value"]["values"] == [15.0]
        assert payload["leaf_gradients"][0]["gradient"]["values"] == [8.0]

    @pytest.mark.parametrize(
        "hostile",
        [
            "__import__('os').system('ls')",
            "open('/etc/passwd').read()",
            "eval('1+1')",
            "x.__class__.__bases__",
            "globals()",
        ],
    )
    def test_hostile_expressions_are_refused_with_400(self, running_server, hostile):
        base, _ = running_server
        status, payload = post(f"{base}/api/trace", {"expression": hostile, "values": {}})
        assert status == 400 and "error" in payload

    def test_oversized_body_is_refused(self, running_server):
        base, _ = running_server
        status, payload = post(
            f"{base}/api/trace",
            {"expression": "x", "values": {"pad": [0.0] * 20000}},
        )
        assert status in (400, 413), payload

    def test_malformed_json_is_refused(self, running_server):
        base, _ = running_server
        request = urllib.request.Request(
            f"{base}/api/trace",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=5)
        assert exc.value.code == 400

    def test_unknown_post_route_is_404(self, running_server):
        base, _ = running_server
        status, _ = post(f"{base}/api/anything", {})
        assert status == 404

    def test_expression_endpoint_is_off_by_default(self, tmp_path):
        """Opt-in means opt-in: without the flag there is no execution surface."""
        (tmp_path / "index.html").write_text("<!doctype html>")
        handler = type(
            "H", (server._Handler,), {"directory_root": tmp_path, "allow_expressions": False}
        )
        httpd = server._Server(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            status, payload = post(f"{base}/api/trace", {"expression": "x", "values": {}})
            assert status == 403 and "--allow-expressions" in payload["hint"]
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_serve_refuses_a_directory_with_no_index(self, tmp_path, capsys):
        assert server.serve(tmp_path, port=0, open_browser=False) == 1
        assert "index.html not found" in capsys.readouterr().out
