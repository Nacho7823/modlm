"""Pytest fixtures for MCP tests."""

from __future__ import annotations

import asyncio
import logging
import socket
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from llmlib.mcp import LocalMCPClient, HTTPMCPClient

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MCP_TEMPLATE_PATH = BASE_DIR / "mcpTemplate" / "mcp_server.py"

NOTEPAD_STDIO_COMMAND = [
    sys.executable,
    str(MCP_TEMPLATE_PATH),
    "--transport",
    "stdio",
]
NOTEPAD_HTTP_PORT = 19876
NOTEPAD_HTTP_URL = f"http://localhost:{NOTEPAD_HTTP_PORT}/mcp"


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open on a host."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


async def wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    """Wait for a port to become available."""
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open(host, port, timeout=1.0):
            return True
        await asyncio.sleep(0.5)
    return False


@pytest.fixture
def notepad_stdio_command() -> list[str]:
    """Command to start the notepad MCP server in stdio mode."""
    return NOTEPAD_STDIO_COMMAND


@pytest.fixture
def notepad_http_port() -> int:
    """Port for the notepad MCP server in HTTP mode."""
    return NOTEPAD_HTTP_PORT


@pytest.fixture
def notepad_http_url() -> str:
    """URL for the notepad MCP server in HTTP mode."""
    return NOTEPAD_HTTP_URL


@pytest_asyncio.fixture
async def notepad_client(
    notepad_stdio_command: list[str],
) -> AsyncGenerator[LocalMCPClient, None]:
    """Create and connect a LocalMCPClient to the notepad server."""
    client = LocalMCPClient(
        name="notepad-stdio",
        command=notepad_stdio_command,
    )
    try:
        await client.connect(connect_timeout=30)
        yield client
    finally:
        await client.disconnect()


@pytest_asyncio.fixture
async def notepad_http_client(
    notepad_http_url: str,
    notepad_http_port: int,
) -> AsyncGenerator[HTTPMCPClient, None]:
    """Create, connect, and manage HTTP notepad server lifecycle."""
    proc = None
    client = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(MCP_TEMPLATE_PATH),
            "--transport",
            "http",
            "--port",
            str(notepad_http_port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(BASE_DIR),
        )

        logger.info(
            f"Started notepad HTTP server on port {notepad_http_port}, PID: {proc.pid}"
        )

        port_ready = await wait_for_port("localhost", notepad_http_port, timeout=15)
        if not port_ready:
            stderr_output = ""
            if proc.stderr:
                try:
                    stderr_data = await asyncio.wait_for(
                        proc.stderr.read(4096), timeout=1
                    )
                    stderr_output = stderr_data.decode() if stderr_data else ""
                except Exception:
                    pass
            raise RuntimeError(
                f"Notepad server not ready on port {notepad_http_port}. "
                f"stderr: {stderr_output}"
            )

        client = HTTPMCPClient(
            name="notepad-http",
            url=notepad_http_url,
        )
        await client.connect(connect_timeout=30)
        yield client
    finally:
        if client is not None and client.is_connected:
            await client.disconnect()
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
            except Exception:
                pass
            logger.info(f"Stopped notepad HTTP server")


@pytest_asyncio.fixture
async def notepad_client_no_connect() -> LocalMCPClient:
    """Create a LocalMCPClient without connecting."""
    return LocalMCPClient(
        name="test-notepad",
        command=["echo", "test"],
    )


@pytest_asyncio.fixture
async def notepad_http_client_no_connect() -> HTTPMCPClient:
    """Create an HTTPMCPClient without connecting."""
    return HTTPMCPClient(
        name="test-notepad-http",
        url=NOTEPAD_HTTP_URL,
    )


@pytest.fixture
def anyio_backend() -> str:
    """Backend for async tests."""
    return "asyncio"
