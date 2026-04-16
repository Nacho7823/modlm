import logging
import sys
from llmapp.app import ChatApp


def main() -> None:
    # Configure logging to file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("llmapp.log", encoding="utf-8"),
        ],
    )
    
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
