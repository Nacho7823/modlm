import os
import json
import argparse
import sys
from mcp.server.fastmcp import FastMCP

parser = argparse.ArgumentParser(description="MCP Server for notepad")
parser.add_argument(
    "--transport",
    choices=["http", "sse", "stdio"],
    default="stdio",
    help="Transport type: http, sse, or stdio (default: stdio)",
)
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host to bind to (default: 0.0.0.0)",
)
parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="Port to bind to (default: 8000)",
)
args = parser.parse_args()

mcp = FastMCP("notepad", host=args.host, port=args.port)

# Almacenamiento simple en memoria para las notas
notes = {}


@mcp.tool()
def create_note(name: str, content: str) -> str:
    """
    Crea una nueva nota.

    Args:
        name: El nombre de la nota.
        content: El contenido de la nota.
    """
    notes[name] = content
    return f"Nota '{name}' creada con éxito."


@mcp.tool()
def read_note(name: str) -> str:
    """
    Lee el contenido de una nota existente.

    Args:
        name: El nombre de la nota.
    """
    return notes.get(name, f"Error: La nota '{name}' no existe.")


@mcp.tool()
def list_notes() -> list:
    """
    Lista los nombres de todas las notas guardadas.
    """
    return list(notes.keys())


@mcp.resource("notes://all")
def get_all_notes() -> str:
    """
    Obtiene todas las notas en formato JSON.
    """
    return json.dumps(notes, indent=2)


if __name__ == "__main__":
    print(f"[notepad] Starting MCP server...", file=sys.stderr)
    print(f"[notepad] Transport: {args.transport}", file=sys.stderr)

    if args.transport == "stdio":
        print(f"[notepad] Using stdio transport", file=sys.stderr)
        mcp.run(transport="stdio")
    elif args.transport == "http":
        print(f"[notepad] HTTP server on {args.host}:{args.port}", file=sys.stderr)
        mcp.run(transport="streamable-http")
    elif args.transport == "sse":
        print(f"[notepad] SSE server on {args.host}:{args.port}", file=sys.stderr)
        mcp.run(transport="sse")
