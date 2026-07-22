# Contributing

Thank you for improving the Pricing Recommendation Engine. Changes should keep
the domain independent from FastAPI, LightGBM, MLflow, and storage details.

## Development workflow

1. Create a focused branch from `main`.
2. Install the locked environment with `uv sync --locked --extra dev`.
3. Add or update tests for every behavior change.
4. Run the release-quality gate:

   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src
   uv run pytest --cov=pricing_engine --cov-report=term-missing
   uv build --no-sources
   docker compose config --quiet
   ```

5. Open a pull request that explains the problem, decision, evidence, and
   operational or model-risk impact.

Do not commit raw customer data, credentials, `.env`, model binaries, MLflow
state, or generated artifacts. Synthetic fixtures must be clearly labeled.

## Architecture and model changes

- Preserve point-in-time correctness and chronological evaluation.
- Reuse the versioned feature factory in training and serving.
- Record consequential architectural decisions in `docs/adr/`.
- Never promote a model automatically from a training job.
- Do not claim causal revenue uplift without controlled experimental evidence.
- Keep API v1 changes backward-compatible; additions are preferred over field
  removals or semantic changes.

## Commit and review guidance

Use concise imperative commit messages. Keep refactors separate from behavioral
changes when practical. Reviewers should be able to reproduce every metric or
demo result from committed code and synthetic or approved data.
