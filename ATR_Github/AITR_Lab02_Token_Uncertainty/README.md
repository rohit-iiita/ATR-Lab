# Token-Level Uncertainty as a Predictor of Hallucinations in LLMs

This is a reproducible experimental scaffold, **not evidence that the experiment has already run**. Execute the code, inspect outputs, and replace all TODO placeholders in `paper/main.tex` before submission.

## Research question
On HaluEval QA, can mean token negative log-likelihood (NLL) and maximum token NLL distinguish benchmark-labelled hallucinated answers from paired reference answers when both are scored conditionally on the same question and knowledge context?

## Design
- Dataset: HaluEval QA, downloaded through Hugging Face Datasets.
- Scorer: `distilgpt2`, a lightweight causal LM that exposes token logits. It is not assumed to be the answer-generating model.
- Predictors: mean response-token NLL and maximum response-token NLL.
- Splitting: source rows split before candidate expansion, so each question's reference/hallucinated pair stays in one split.
- Metrics: AUROC, AUPRC, F1, precision, recall. F1 threshold selected on validation only.
- Deeper analysis: false positives/negatives, paired score differences, response length.
- NLL is a model-likelihood signal, not a probability that an answer is true.

## Google Colab quick start
```bash
!pip install -r requirements.txt
!python scripts/run_experiment.py --n_pairs 50 --model_name distilgpt2 --output_dir results/smoke_test
!python scripts/run_experiment.py --n_pairs 500 --model_name distilgpt2 --output_dir results/main
```
Run the 50-pair smoke test first. Inspect output CSV/JSON and record actual versions, hardware, runtime and errors in `baseline_validation.md`. Do not report any result until the code has actually produced it.

## Project files
- `src/score_halueval.py`: data loading, token scoring, metrics and error analysis.
- `scripts/run_experiment.py`: command-line entry point.
- `paper/main.tex`, `paper/references.bib`: manuscript draft and references.
- `research_decisions.md`: review-to-experiment reasoning.
- `baseline_validation.md`: fill after execution.
- `ai_assistance_log.md`: update honestly to reflect actual AI use and human verification.
- `data/README.md`: dataset source and license-check reminder.

## Limitations
One benchmark, one scorer and a small sample do not establish generalization. Token NLL is affected by tokenization, lexical rarity, response length and the scorer's training distribution. HaluEval labels are benchmark annotations, not universal ground truth. Report negative or inconclusive findings honestly.
