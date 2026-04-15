"""Entry point for llmapp."""

import sys
from llmapp.app import ChatApp


def main() -> None:
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
