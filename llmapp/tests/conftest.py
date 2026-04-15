"""Test configuration for llmapp."""

import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

sys.path.insert(0, str(project_root))
