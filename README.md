# Lead Intelligence System

A small Python CLI that reads a CSV of inbound leads, scores them against a transparent rubric, uses the Groq LLM API to classify leads in batches, drafts personalized outreach for qualified leads, and writes an actionable JSON report.

## Run it

1. Install Python 3.10+.
2. In this folder, run `python -m venv .venv`.
3. Activate it:
   - Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`.
5. Copy `.env.example` to `.env` and set `GROQ_API_KEY` to your own key. Never commit `.env`.
6. Run: `python main.py --input leads.csv --output output_report.json`.
7. Optional: `python main.py --input leads.csv --output demo_report.json --demo` tests the full pipeline without calling an API.

The default batch size is 10, so 100 leads require about 10 LLM calls rather than one call per lead. You can change it with `--batch-size 5` or `--batch-size 20` (maximum 50). If a batch fails after retries, the program uses a transparent rules-based fallback, marks those records for human review, and continues processing.

## Qualification rubric

The score is 1–10, based on four weighted factors: company-size fit (30%), industry fit (30%), source/intent signal (20%), and recency of last interaction (20%). Scores 8–10 are qualified when supported by the supplied data; scores 1–4 are rejected only when evidence is clear; borderline, incomplete, or conflicting cases are sent to human review. The weights and target profile are assumptions for this sample B2B SaaS business and should be calibrated with the actual sales team.

## Output

`output_report.json` contains aggregate stats, a ranked priority queue, flagged/review leads, rejected leads, all lead results, reasoning, factor scores, analysis method, and up to five outreach examples.

## Known limitations

- This sample has no real product description, ICP, pricing, budget, or timeline fields. The rubric uses industry, company size, source, and recency as proxies; it must be adapted to the real product.
- LLM output can be wrong. Check reasoning and human-review cases before contacting leads.
- The fallback is intentionally simple and is not equivalent to LLM judgment.
- The tool drafts messages only; it does not send email or update a CRM.
- The date parser accepts common formats but ambiguous dates should be normalized to `YYYY-MM-DD`.
