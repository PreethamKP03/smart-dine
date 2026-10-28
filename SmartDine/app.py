"""Compatibility launcher; UI source is the outer app.py."""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
runpy.run_path(str(root / 'app.py'), run_name='__main__')
