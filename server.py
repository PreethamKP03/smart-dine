"""
Root entry point to launch the SmartDine Real-Time Serving Web Application.
Run:
    python server.py
"""

import sys
import os

# Add SmartDine folder to python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SMARTDINE_DIR = os.path.join(ROOT_DIR, "SmartDine")
if SMARTDINE_DIR not in sys.path:
    sys.path.insert(0, SMARTDINE_DIR)

import uvicorn
from SmartDine.src.serving import app

if __name__ == "__main__":
    port = 8000
    print(f"\n============================================================")
    print(f"  SmartDine Real-Time Serving Engine Active")
    print(f"  Interactive Web UI : http://localhost:{port}")
    print(f"  Swagger API Docs   : http://localhost:{port}/docs")
    print(f"============================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
