"""Outbound URL guard against server-side request forgery.

The model reads untrusted web content, so a page could instruct it to fetch
``http://localhost:11434/...`` (Ollama), the voicemate UI, a router admin page or a
link-local metadata address. Every URL a tool fetches on the model's behalf must therefore
resolve to public addresses only, and redirects are followed manually so each hop is checked
again. (DNS rebinding between the check and the connection remains a residual risk.)
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx

from voicemate.agent.tools.base import ToolError

#: Maximum redirects followed by :func:`safe_get`.
MAX_REDIRECTS: int = 5

#: Largest response body :func:`safe_get` downloads (bytes).
MAX_BODY_BYTES: int = 20_000_000

_STREAM_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


async def check_url(url: str, allowed_hosts: frozenset[str] = frozenset()) -> None:
    """Reject URLs that are not http(s) or that resolve to a non-public address.

    Args:
        url: The URL to check.
        allowed_hosts: Host names exempt from the address check (tests, explicit config).

    Raises:
        ToolError: If the URL is not allowed.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ToolError("Only http(s) URLs with a host name can be fetched")
    host = parts.hostname
    if host in allowed_hosts:
        return
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ToolError(f"Cannot resolve {host}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global or address.is_multicast:
            raise ToolError(f"Blocked {host}: it points to a local or private network address")


async def safe_get(
    http: httpx.AsyncClient,
    url: str,
    allowed_hosts: frozenset[str] = frozenset(),
    max_redirects: int = MAX_REDIRECTS,
    max_bytes: int = MAX_BODY_BYTES,
) -> httpx.Response:
    """GET ``url``, checking the target and every redirect hop with :func:`check_url`.

    The body is streamed and the download stops at ``max_bytes``.

    Raises:
        ToolError: On a blocked hop, too many redirects or an oversized body.
        httpx.HTTPError: On network or HTTP status errors.
    """
    for _ in range(max_redirects + 1):
        await check_url(url, allowed_hosts)
        async with http.stream("GET", url, follow_redirects=False) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers["location"])
                continue
            response.raise_for_status()
            body = bytearray()
            async for block in response.aiter_bytes():
                body += block
                if len(body) > max_bytes:
                    raise ToolError(f"Response larger than {max_bytes // 1_000_000} MB")
            # The body is already decompressed, so encoding/length headers no longer apply.
            headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() not in _STREAM_HEADERS
            }
            return httpx.Response(
                response.status_code, headers=headers, content=bytes(body), request=response.request
            )
    raise ToolError("Too many redirects")
