import asyncio
import gzip
import http.server
import threading
import unittest

import httpx

from voicemate.agent.tools.base import ToolError
from voicemate.agent.tools.netguard import check_url, safe_get


class _Handler(http.server.BaseHTTPRequestHandler):
    """/ok → 200, /to-localhost → redirect to localhost, /loop → redirect to itself."""

    def do_GET(self):  # noqa: N802
        port = self.server.server_address[1]
        if self.path == "/gzip":
            body = gzip.compress(b"compressed fine")
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/ok":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"fine")
            return
        self.send_response(302)
        target = f"http://localhost:{port}/ok" if self.path == "/to-localhost" else "/loop"
        self.send_header("Location", target)
        self.end_headers()

    def log_message(self, *args):
        pass


class TestCheckUrl(unittest.IsolatedAsyncioTestCase):
    async def test_local_and_private_targets_are_blocked(self):
        for url in (
            "http://127.0.0.1:11434/api/tags",
            "http://localhost:8080/",
            "http://10.0.0.1/",
            "http://192.168.1.1/admin",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://0.0.0.0/",
        ):
            with self.subTest(url=url), self.assertRaises(ToolError):
                await check_url(url)

    async def test_non_http_schemes_and_missing_hosts_are_blocked(self):
        for url in ("file:///etc/passwd", "ftp://example.org/x", "http:///nohost", "javascript:1"):
            with self.subTest(url=url), self.assertRaises(ToolError):
                await check_url(url)

    async def test_public_address_passes_and_allowlist_is_honoured(self):
        await check_url("https://1.1.1.1/")
        await check_url("http://127.0.0.1:9/", allowed_hosts=frozenset({"127.0.0.1"}))


class TestSafeGet(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.http = httpx.AsyncClient()
        self.allowed = frozenset({"127.0.0.1"})

    async def asyncTearDown(self):
        await self.http.aclose()
        await asyncio.to_thread(self.server.shutdown)  # blocks ~0.5 s; keep the loop free
        self.server.server_close()

    async def test_plain_get(self):
        response = await safe_get(self.http, f"{self.base}/ok", self.allowed)
        self.assertEqual(response.text, "fine")

    async def test_redirect_to_a_local_host_is_blocked(self):
        with self.assertRaises(ToolError):
            await safe_get(self.http, f"{self.base}/to-localhost", self.allowed)

    async def test_gzip_body_is_decoded_once(self):
        response = await safe_get(self.http, f"{self.base}/gzip", self.allowed)
        self.assertEqual(response.text, "compressed fine")

    async def test_oversized_body_is_refused(self):
        with self.assertRaises(ToolError):
            await safe_get(self.http, f"{self.base}/ok", self.allowed, max_bytes=2)

    async def test_redirect_loop_stops(self):
        with self.assertRaises(ToolError):
            await safe_get(self.http, f"{self.base}/loop", self.allowed, max_redirects=3)


if __name__ == "__main__":
    unittest.main()
