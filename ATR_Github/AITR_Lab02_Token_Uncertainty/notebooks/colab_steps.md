# Google Colab steps
1. Create a new Colab notebook and enable a GPU runtime if available.
2. Upload or clone this repository into the Colab working directory.
3. Install dependencies: `!pip install -r requirements.txt`
4. Smoke test: `!python scripts/run_experiment.py --n_pairs 50 --model_name distilgpt2 --output_dir results/smoke_test`
5. Inspect `results/smoke_test/metrics.json` and `predictions.csv`. Check finite scores and nonzero scored-token counts.
6. If successful, run: `!python scripts/run_experiment.py --n_pairs 500 --model_name distilgpt2 --output_dir results/main`
7. Save the results folder and record exact command/runtime details in `baseline_validation.md`.
8. Copy actual plots to `paper/figures/` and compile the manuscript only after replacing TODOs.
