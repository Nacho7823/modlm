#!/bin/bash
cd /mnt/d/projects/modlm/backend
source venv/bin/activate
python -c "from app.services.mcp_manager import mcp_manager; methods = [m for m in dir(mcp_manager) if not m.startswith('_')]; print(methods)"