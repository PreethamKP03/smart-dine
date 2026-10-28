"""Build a small explicit-allowlist submission archive; never include private/local data."""
from pathlib import Path
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from SmartDine.src.schema import sha256, write_json


def package():
    names = ['app.py','server.py','pipeline.py','requirements.txt','README.md','PROJECT_AUDIT.md',
             'pytest.ini','.gitignore','SmartDine/__init__.py','SmartDine/app.py',
             'SmartDine/README.md','SmartDine/requirements.txt','SmartDine/.gitignore',
             'SmartDine/data/processed/restaurant_features.csv',
             'SmartDine/data/processed/group_members.csv',
             'SmartDine/data/processed/split_manifest.json',
             'SmartDine/data/processed/group_restaurant_features.manifest.json',
             'SmartDine/models/smartdine_best_model.joblib',
             'SmartDine/models/feature_columns.txt','SmartDine/models/model_metadata.json',
             'SmartDine/outputs/README.md','SmartDine/outputs/model_comparison.csv',
             'SmartDine/outputs/model_evaluation_metrics.csv','SmartDine/outputs/feature_coefficients.csv',
             'SmartDine/outputs/scenarios.json','SmartDine/outputs/historical_replay.json']
    for folder, extensions in [('SmartDine/src',{'.py'}),('SmartDine/frontend',{'.html','.css','.js'}),
            ('SmartDine/notebooks',{'.ipynb'}),('SmartDine/outputs/figures',{'.png','.json'}),
            ('tests',{'.py','.cjs'}),('tools',{'.py'}),('docs',{'.md','.json','.xml','.png'})]:
        names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).iterdir()
                     if p.is_file() and p.suffix in extensions and p.name!='submission_manifest.json')
    records=[]
    for name in sorted(set(names)):
        path=ROOT/name
        if not path.is_file(): raise FileNotFoundError(name)
        if path.stat().st_size >= 50*1024*1024: raise ValueError('Unexpected large submission file: '+name)
        records.append({'path':name,'bytes':path.stat().st_size,'sha256':sha256(path)})
    manifest={'files':records,'file_count':len(records),
              'excluded':'raw/cleaned/matrix/train/test CSVs, environment, backups, caches, logs, legacy output images/profiles',
              'raw_pipeline_verified':False}
    destination=ROOT.parent/'smart-dine-submission.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for record in records: archive.write(ROOT/record['path'],'smart-dine-main/'+record['path'])
        archive.writestr('smart-dine-main/SUBMISSION_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(records)+1
    write_json(ROOT/'docs/submission_manifest.json',{**manifest,'archive_name':destination.name,
               'archive_bytes':destination.stat().st_size,'archive_sha256':sha256(destination),
               'crc_check':'passed'})
    print(f'Packaged {len(records)+1} files; {destination.stat().st_size:,} bytes; CRC check passed.')
    return destination


if __name__=='__main__':
    package()
