"""Real local server checks plus programmatic Streamlit widget interactions.

This does not claim a human/browser visual review. Each child is stopped in
finally, and its output is retained in local ignored logs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from SmartDine.src.schema import write_json


def request(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=10) as response:
        body = response.read().decode()
        return response.status, body


def check_server(command, url, label, extra=None):
    log = ROOT / 'docs' / (label + '.log')
    with log.open('w', encoding='utf-8') as output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f'{label} exited before health check; inspect local log.')
                try:
                    status, body = request(url)
                    break
                except (OSError, TimeoutError):
                    time.sleep(.5)
            else:
                raise TimeoutError(label + ' startup timed out')
            result = {'health_status':status, 'health_response':body}
            if extra: result.update(extra())
            return result
        finally:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=5)


def main():
    results = {}
    def api_extra():
        status, body = request('http://127.0.0.1:8000/api/recommend',
                               {'members':[{'preferred_cuisines':['Italian'], 'budget':800}], 'top_n':3})
        result = json.loads(body)
        assert status==200 and len(result['recommendations'])==3
        _, empty = request('http://127.0.0.1:8000/api/recommend',
                            {'members':[{}], 'filter_location':'not a real neighborhood'})
        assert not json.loads(empty)['recommendations']
        assert request('http://127.0.0.1:8000/static/app.js')[0] == 200
        assert request('http://127.0.0.1:8000/static/index.css')[0] == 200
        return {'recommend_status':status, 'recommendations':len(result['recommendations']),
                'empty_filter_verified':True, 'dashboard_assets_status':200,
                'dashboard_status':request('http://127.0.0.1:8000/')[0]}
    results['api'] = check_server([sys.executable, 'server.py'], 'http://127.0.0.1:8000/api/health', 'api', api_extra)
    results['streamlit'] = check_server([sys.executable, '-m', 'streamlit', 'run', 'app.py',
                                        '--server.headless=true', '--server.address=127.0.0.1',
                                        '--server.port=8501', '--browser.gatherUsageStats=false'],
                                        'http://127.0.0.1:8501/_stcore/health', 'streamlit')
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
    assert not at.exception and not at.error, str(at.exception)
    at.button[0].click().run()
    assert not at.exception and not at.error, str(at.exception)
    # Change preset after widgets have already been rendered (session regression).
    at.sidebar.selectbox[0].select('Custom Group').run()
    assert not at.exception and not at.error, str(at.exception)
    at.number_input(key='group_size').set_value(2).run()
    assert not at.exception and not at.error, str(at.exception)
    at.button[0].click().run()
    assert not at.exception and not at.error, str(at.exception)
    results['streamlit_widget_test'] = {'initial_render':True, 'recommend_click':True,
                                      'preset_change':True, 'member_count_change':True,
                                      'browser_visual_review':False}
    write_json(ROOT/'docs/app-smoke-results.json', results)
    print(json.dumps(results, indent=2))


if __name__=='__main__':
    main()
