"""Labeled EDA of the supplied processed catalogue, not a claim about raw listings."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .paths import DATA_DIR, FIGURE_DIR
from .schema import sha256, write_json


def generate_eda(input_path=None, output_dir=FIGURE_DIR):
    path = Path(input_path or DATA_DIR / 'restaurant_features.csv')
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError('EDA requires a nonempty processed catalogue.')
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    def save(name, title, xlabel, ylabel, draw):
        fig, ax = plt.subplots(figsize=(9, 6))
        draw(ax)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout(); fig.savefig(dest / name, dpi=150); plt.close(fig)
        files.append(name)
    save('missing_values.png', 'Processed catalogue: unknown values', 'Missing fields', 'Restaurant count',
         lambda ax: frame.isna().sum().plot.bar(ax=ax))
    for column, title, xlabel in [('rating', 'Observed ratings only', 'Rating out of 5'),
                                  ('cost_for_two', 'Observed costs (full range)', 'INR for two people'),
                                  ('votes', 'Observed votes (log count axis)', 'Votes')]:
        save(column + '_distribution.png', title, xlabel, 'Restaurant count',
             lambda ax, c=column: ax.hist(pd.to_numeric(frame[c], errors='coerce').dropna(), bins=40, log=c=='votes'))
    for column, title in [('location', 'Top 15 listed neighborhoods'), ('rest_type', 'Top 15 restaurant type combinations')]:
        save(column + '_counts.png', title, 'Restaurants', column,
             lambda ax, c=column: frame[c].fillna('Unknown').value_counts().head(15).sort_values().plot.barh(ax=ax))
    cuisines = frame.cuisines.dropna().str.split(',').explode().str.strip().value_counts().head(15)
    save('cuisines.png', 'Top 15 cuisine labels (multi-label counts)', 'Restaurant-label count', 'Cuisine',
         lambda ax: cuisines.sort_values().plot.barh(ax=ax))
    numeric = frame[['rating', 'cost_for_two', 'votes', 'cuisine_count']].corr()
    def correlation(ax):
        im = ax.imshow(numeric, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(len(numeric)), numeric.columns, rotation=35)
        ax.set_yticks(range(len(numeric)), numeric.columns)
        ax.figure.colorbar(im, ax=ax, label='Pearson correlation (pairwise complete)')
    save('correlations.png', 'Processed numeric correlations', '', '', correlation)
    groups_path = DATA_DIR / 'group_members.csv'
    if groups_path.exists():
        groups = pd.read_csv(groups_path)
        save('synthetic_budgets.png', 'Synthetic member budgets; not survey responses', 'INR for two', 'Synthetic member count',
             lambda ax: ax.hist(groups.budget, bins=20))
    write_json(dest / 'eda_manifest.json', {'input_sha256': sha256(path), 'rows': len(frame), 'figures': files,
               'figure_count': len(files), 'scope': 'processed supplied snapshot; raw cleaning not verified'})
    print(f'Generated {len(files)} EDA figures.')
    return files


if __name__ == '__main__':
    generate_eda()
