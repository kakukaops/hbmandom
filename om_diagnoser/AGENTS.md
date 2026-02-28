# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-based Python pipeline for optical module fault simulation, labeling, training, and inference.
- Core scripts: `simulator.py`, `data_preprocessor.py`, `auto_labeler.py`, `om_fault_predictor.py`, `predict_faults.py`
- Orchestration: `run_pipeline.sh`
- Configuration: `config/info.yaml`, `config/rules.yaml`, `config/hyper_parameters.yaml`
- Generated artifacts (created at runtime): `data/`, `metadata/`, `models/`, `reports/`, `plots/`, `predictions/`

Keep new modules focused and colocated with the stage they extend (preprocess/train/predict).

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install dependencies.
- `python data_preprocessor.py --simulation --period_days 30 --num_modules 10 --fault_ratio 0.2`: generate simulated raw/features data.
- `python auto_labeler.py`: apply rules from `config/rules.yaml` and write labeled data.
- `python om_fault_predictor.py --target rx_los`: train and evaluate one fault-type model.
- `python predict_faults.py --target rx_los --example`: run a quick prediction smoke check.
- `./run_pipeline.sh [input_csv] [target]`: run labeling -> feature generation -> training end-to-end.

## Coding Style & Naming Conventions
Use Python 3.8+ style already present in the codebase:
- 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Prefer explicit `argparse` CLI options and type hints on non-trivial functions.
- Keep YAML keys lowercase and descriptive (example: `predict_window_days`, `label_column`).
- No formatter/linter is currently enforced; match existing style and keep diffs tight.

## Testing Guidelines
There is no dedicated `tests/` suite or coverage gate yet. Minimum validation for code changes:
- Run the affected CLI path(s).
- For pipeline changes, run `./run_pipeline.sh` or equivalent stage commands.
- For model/prediction changes, confirm new files in `models/` and `reports/model_evaluation_report.json`, then run `predict_faults.py`.

If you add reusable logic, include `pytest` tests in a new `tests/` directory and document fixtures/data assumptions.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commits, sometimes with scoped prefixes (for example, `refactor(om_diagnoser): ...`).
- Recommended format: `type(scope): brief summary` (example: `refactor(preprocessor): simplify target column generation`).
- PRs should include: purpose, changed configs/paths, commands run for validation, and key output artifacts or metrics impacted.
- Link related issues/tasks and avoid committing large generated files unless explicitly needed for review.
