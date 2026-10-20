# One-Page Research Gap and Decision Brief

**Review topic:** Token-Level Uncertainty as a Predictor of Hallucinations in Large Language Models.

**Review-derived motivation:** The literature review discusses token negative log-likelihood (NLL), entropy, claim-conditioned probability, semantic entropy, calibration and token/claim-level aggregation. It also identifies lexical rarity, tokenization and confidently wrong outputs as reasons that token uncertainty cannot be treated as truth. The review therefore motivates a controlled comparison of simple sequence-level aggregations.

**Candidate gap (qualified, not a novelty claim):** A practical baseline comparison of mean versus maximum response-token NLL under one fixed scoring model and a paired benchmark can help establish how these two simple aggregation rules behave before adding more complex predictors. This project does not claim that the comparison has never been performed; a focused literature search is required before claiming novelty.

**Research question:** On the HaluEval QA paired-answer benchmark, do mean token NLL and maximum token NLL distinguish benchmark-labelled hallucinated answers from reference answers when scored by the same causal language model?

**Hypotheses:** H1: hallucinated answers will tend to receive higher mean response-token NLL. H2: maximum token NLL will offer a different or complementary ranking signal.

**Dataset/source:** `pminervini/HaluEval`, QA configuration, expected fields `knowledge`, `question`, `right_answer`, `hallucinated_answer`. Verify the live dataset card, actual loaded schema, access date, revision and license. Upstream project and paper: https://github.com/RUCAIBox/HaluEval and https://doi.org/10.18653/v1/2023.emnlp-main.397.

**Baseline/scorer:** `distilgpt2`, scoring each answer conditionally on the same knowledge/question prompt. It is not assumed to be the model that generated the dataset answers; this is cross-model likelihood scoring.

**Predictors:** Mean response-token NLL and maximum response-token NLL. Prompt tokens excluded.

**Protocol:** Select source rows with seed 42; split rows 60%/20%/20% into train/validation/test before creating paired candidates. Use the validation split to choose an F1 threshold; freeze it for test. Report AUROC, AUPRC, F1, precision, recall, confusion matrix, paired score differences and false-positive/false-negative examples.

**Deeper evaluation:** Analyze false positives/negatives and response-length association. Treat this as descriptive; do not infer causation.

**Compute and risk:** Start with 50 pairs in Colab; only scale to 500 after verifying output and runtime. Risks include memory/time, tokenizer effects, long prompts, scorer/dataset-generation mismatch, benchmark-label limitations and rare-word confounding.

**First executable milestone:** Load and inspect 50 source rows, score both candidates, confirm finite scores and positive token counts, save CSV/JSON, and complete the baseline validation report from actual execution. No metrics are asserted before this run.
