# Research Decisions Log

## Project
Token-Level Uncertainty as a Predictor of Hallucinations in Large Language Models

## Decision 1 — Narrow the broad review question
**Decision:** Compare mean token NLL and maximum token NLL as sequence-level signals for ranking HaluEval QA hallucinated answers above paired reference answers.

**Reason:** The literature review discusses many candidate signals (entropy, probability margin, semantic entropy, hidden states, calibration and claim-conditioned probability). Mean and maximum NLL are simpler, comparable baselines for a short reproducible assignment.

**Evidence from review:** Sections 3 and 9 discuss token NLL and sequence-level aggregation; Sections 7–8 discuss lexical confounding and uncertainty versus truth.

**Status:** INFERRED experimental direction; not a claim of novelty.

## Decision 2 — Dataset
Use the QA configuration of HaluEval through Hugging Face Datasets. Each source row pairs a reference and a hallucinated answer under the same context. Verify the live dataset card, schema, license and revision before final submission.

## Decision 3 — Scoring model
Use `distilgpt2` as a lightweight open-weight causal language-model scorer. It exposes logits and is feasible for Colab. It is not necessarily the model that generated the benchmark answers; this is cross-model likelihood scoring, not a replication of the original generation process.

## Decision 4 — Predictors
- Mean response-token NLL: average negative log probability of response tokens given prompt and preceding answer tokens.
- Maximum response-token NLL: maximum negative log probability among response tokens.
- Prompt tokens are excluded. NLL is not a truth probability.

## Decision 5 — Evaluation protocol
Split source rows into train/validation/test before expanding each row into paired candidates. Use the same prompt for both answers. Report AUROC and AUPRC; select the F1 threshold on validation only and apply it unchanged to test. Save per-example scores and logs. Analyze false positives/negatives and response length.

## Decision 6 — Claim boundary
Conclusions apply only to this scorer, dataset configuration, sample and protocol. They do not establish causal interpretation, token localization, cross-model calibration or broad generalization.

## First executable milestone
Load the dataset, verify columns, score 50 source rows, inspect output CSVs/metrics and record actual hardware/runtime/errors in `baseline_validation.md`.
