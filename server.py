"""Optional local API and dashboard: python server.py."""
import uvicorn
from SmartDine.src.serving import app

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')
