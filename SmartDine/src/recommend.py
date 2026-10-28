"""Representative scenarios from actual saved synthetic profiles."""
import pandas as pd
from .paths import DATA_DIR, OUTPUT_DIR
from .features import members_from_frame
from .group_preferences import GroupPreferences
from .recommendation import SmartDineRecommender
from .schema import write_json


def run_scenarios():
    engine = SmartDineRecommender()
    groups = pd.read_csv(DATA_DIR / 'group_members.csv')
    results = {}
    for gid, frame in list(groups.groupby('group_id'))[:5]:
        result = engine.recommend(GroupPreferences(members_from_frame(frame)), top_n=5)
        results[str(gid)] = result
        print(f'{gid}: {len(result["recommendations"])} results; {result["total_candidates_evaluated"]} eligible.')
    write_json(OUTPUT_DIR / 'scenarios.json', results)
    return results


if __name__ == '__main__':
    run_scenarios()
