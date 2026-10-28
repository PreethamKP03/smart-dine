"""Run explicit SmartDine stages from any working directory."""
import argparse
import importlib
import sys
from pathlib import Path

STAGES = {
    'validate': ('preprocessing', 'validate_raw'),
    'clean': ('preprocessing', 'clean_dataset'),
    'features': ('feature_engineering', 'generate_features'),
    'groups': ('generate_group_data', 'generate_groups'),
    'matrix': ('build_group_restaurant_features', 'build_matrix'),
    'split': ('prepare_ml_dataset', 'prepare_dataset'),
    'train': ('train_model', 'train_and_compare_models'),
    'evaluate': ('evaluation', 'evaluate_system'),
    'eda': ('eda', 'generate_eda'),
    'scenarios': ('recommend', 'run_scenarios'),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stages', nargs='+', choices=[*STAGES, 'all'])
    args = parser.parse_args(argv)
    stages = list(STAGES) if 'all' in args.stages else args.stages
    for stage in stages:
        print(f'STAGE {stage}', flush=True)
        try:
            module, function = STAGES[stage]
            getattr(importlib.import_module('SmartDine.src.' + module), function)()
        except FileNotFoundError as exc:
            # Avoid personal filesystem paths in recorded logs.
            print(f'BLOCKED {stage}: required input missing. {exc.strerror or str(exc) if not exc.filename else Path(exc.filename).name}', file=sys.stderr)
            return 2
        except (ValueError, KeyError, TypeError) as exc:
            print(f'FAILED {stage}: {exc}', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
