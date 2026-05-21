# Script Entry Points

`scripts/run_pipeline.py` is the preferred public workflow entry point.
Subfolders keep granular commands available for reruns, debugging, and
inspection.

- `data/`: reference and source data download commands.
- `build/`: grid, lookup, feature, and model-table construction commands.
- `model/`: model training, prediction, evaluation, and validation variants.
- `qa/`: table inspection, summaries, and relationship diagnostics.
- `plots/`: maps and diagnostic figures.
- `tools/`: specialized utilities that are not part of the main workflow.
- `dev/`: local development scripts; ignored by Git.

Run scripts from the repository root after installing the package in editable
mode:

```bash
pip install -e .
python scripts/run_pipeline.py --stage all
```

Grouped plotting commands are available through:

```bash
python scripts/plots/plot_all_maps.py --group context
python scripts/plots/plot_all_maps.py --group environmental predictions
python scripts/plots/plot_all_maps.py --group weekly
python scripts/plots/plot_all_maps.py --group gear
python scripts/plots/plot_all_maps.py --group videos
```
