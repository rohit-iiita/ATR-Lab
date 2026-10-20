# Assignment completion map

## Phase I — From review to research question
Use `research_decisions.md`. The selected question is deliberately narrower than the review's broad research question. Verify at least 3–5 references in the original papers and record checked passages/DOIs.

## Phase II — Gap verification and feasibility
Motivating gap: the review identifies lexical confounding and the need to compare uncertainty aggregation at meaningful units. This experiment tests two sequence-level aggregations. This is not a verified novelty claim. Search recent literature for direct comparisons before using words such as “novel,” “first,” or “state of the art.”

## Phase III — Baseline and implementation
Run the 50-pair smoke test, inspect the schema and token scoring, then run the 500-pair experiment if resources permit. Complete `baseline_validation.md` from actual execution evidence.

## Phase IV — Experimental evidence
Use `results/main/metrics.json`, `predictions.csv`, distribution plots, `length_vs_mean_nll.png`, and `test_error_analysis.csv`. Report actual metrics, failure cases, limitations and any negative findings. Never paste placeholder values.

## Phase V — Manuscript, peer review and reproducibility
Complete `paper/main.tex` and `paper/references.bib`; copy actual plots into `paper/figures/`; compile the 6–10-page manuscript using the instructor's required template if one exists. Use `paper/peer_review_checklist.md` to critically review and revise. The GitHub repository must contain code, instructions, environment details and actual outputs.

## Phase VI — AI provenance
Update `ai_assistance_log.md` to record actual AI use, human decisions, tests and modifications. Do not check any box that is not true.

## Final manual checks
- Confirm deadline/template/group policy on Google Classroom.
- Verify dataset terms and every citation.
- Run code from a clean runtime or document why this was not possible.
- Ensure paper results exactly match saved CSV/JSON.
- Commit no secrets, restricted data, fabricated results or unverified claims.
- Five-minute demo: question (30s), data/scoring (60s), code execution (60s), actual results/errors (90s), limitations/reproducibility (60s).
