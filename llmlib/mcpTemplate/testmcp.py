#!/usr/bin/env python
"""
Test script for MCP server (remote mode).
Usage: python testmcp.py
"""

import asyncio
import sys
import os
import platform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


def get_python_command():
    """Get the appropriate Python command for the current platform."""
    if platform.system() == "Windows":
        return "python"
    return "python3"


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Test MCP server")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["http", "stdio"],
        default="http",
        help="Transport type: http or stdio (default: http)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host for HTTP transport (default: localhost)",
    )
    return parser.parse_args()


async def test_http(host: str, port: int):
    """Test MCP server with streamable-http transport."""
    print("=" * 50)
    print(f"Testing MCP server with streamable-http transport")
    print(f"URL: http://{host}:{port}/mcp")
    print("=" * 50)

    async with streamable_http_client(f"http://{host}:{port}/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            print("\n--- List Tools ---")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            print("\n--- List Resources ---")
            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"  - {resource.uri}")


async def test_stdio():
    """Test MCP server with stdio transport."""
    print("=" * 50)
    print("Testing MCP server with stdio transport")
    print("=" * 50)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(script_dir, "mcp_server.py")
    python_cmd = get_python_command()

    server_params = StdioServerParameters(
        command=python_cmd,
        args=[server_script],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n--- List Tools ---")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            print("\n--- List Resources ---")
            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"  - {resource.uri}")


async def main():
    args = parse_args()

    if args.transport == "stdio":
        await test_stdio()
    else:
        print("Testing MCP server")
        await test_http(args.host, args.port)


if __name__ == "__main__":
    asyncio.run(main())
