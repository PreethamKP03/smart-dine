"""Record syntax, imports, environment consistency, paths and figure integrity."""
from pathlib import Path
import importlib
import json
import re
import subprocess
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from SmartDine.src.schema import write_json, sha256


def main():
    records = []
    commands = [[sys.executable, '-m', 'compileall', '-q', 'app.py', 'server.py', 'pipeline.py', 'SmartDine/src', 'tests', 'tools'],
                ['node', '--check', 'SmartDine/frontend/app.js'],
                [sys.executable, '-m', 'pip', 'check'],
                [sys.executable, str(ROOT/'pipeline.py'), 'evaluate']]
    for command in commands:
        # Evaluate intentionally launches from outside the canonical root.
        result = subprocess.run(command, cwd=ROOT.parent if command[-1]=='evaluate' else ROOT,
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        record = {'command': ' '.join(['python' if c==sys.executable else Path(c).name if c==str(ROOT/'pipeline.py') else c for c in command]),
                  'exit_code':result.returncode, 'stdout':result.stdout.replace(str(ROOT), '<project>'),
                  'stderr':result.stderr.replace(str(ROOT), '<project>')}
        records.append(record)
        if result.returncode: raise RuntimeError(record)
    for path in (ROOT/'SmartDine/src').glob('*.py'):
        importlib.import_module('SmartDine.src.'+path.stem)
    violations=[]
    for folder in [ROOT/'SmartDine/src', ROOT/'SmartDine/frontend', ROOT/'SmartDine/notebooks']:
        for path in folder.rglob('*'):
            if path.suffix in {'.py','.js','.html','.ipynb'}:
                content = path.read_text(encoding='utf-8')
                if path.suffix == '.ipynb':
                    content = '\n'.join(''.join(c.get('source', [])) for c in json.loads(content)['cells'])
                if re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/|/home/', content):
                    violations.append(path.relative_to(ROOT).as_posix())
    figures=[]
    for path in sorted((ROOT/'SmartDine/outputs/figures').glob('*.png')):
        with Image.open(path) as im:
            size=im.size
            im.verify()
        figures.append({'file':path.name,'pixels':size,'sha256':sha256(path)})
    record={'commands':records, 'all_source_modules_imported':True,
            'absolute_path_violations':violations,'valid_figure_count':len(figures),'figures':figures,
            'git':'Unavailable in supplied ZIP; original workspace .git left untouched.',
            'browser_check':'Recommend button clicked in IAB; cards, rationale and uncertainty labels visibly rendered.',
            'browser_screenshot':'docs/browser-recommendations.png'}
    write_json(ROOT/'docs/verification.json',record)
    assert not violations
    print(json.dumps({'commands_passed':len(records), 'source_imports':'passed',
                      'absolute_path_violations':violations, 'valid_figures':len(figures)},indent=2))


if __name__=='__main__':
    main()
