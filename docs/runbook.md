# Operations Runbook

This runbook covers local development and the Compose integration environment.
It is not a cloud deployment guide. Commands assume execution from the
repository root.

## Prerequisites

- Python 3.11, 3.12, or 3.13;
- `uv` 0.9.15 or another version capable of reading the committed lockfile;
- Docker Desktop with Compose v2 for the integration workflow; and
- enough memory for LightGBM, SHAP, MLflow, PostgreSQL, and MinIO.

## Locked local setup

PowerShell:

```powershell
Copy-Item .env.example .env
uv sync --locked --extra dev --python 3.12
uv run pricing-engine generate-demo-data `
  --output data/demo/observations.parquet `
  --properties 12 `
  --decision-days 180 `
  --seed 42
uv run pricing-engine train `
  --input data/demo/observations.parquet `
  --output artifacts/local-model
```

The demo generator is deterministic for a fixed seed. With the default 400
central trees and 100 trees per quantile tail, this produces 2,160 rows;
post-purge train/calibration/test counts are 1,416/218/253 and purge counts are
130/143. The final snapshot reports MAE 0.0214351, RMSE 0.0299894, R²
0.9308273, simultaneous coverage 0.9169960, width 0.2014993, and
revenue-at-historical-price WAPE 0.0284972. Its metrics are synthetic
integration evidence, not proof of real-market quality, causal elasticity, or
policy uplift.

The corresponding policy diagnostics are mean response 0.1944230, flat rate
0.0039526, upper/lower/any boundary rates
0.0553360/0.0000000/0.0553360, and mean selected price change -0.0063241.
`holiday/no` has 168 rows, coverage 0.93452, MAE 0.01880, and WAPE 0.02457;
`holiday/yes` has 85 rows, coverage 0.88235, MAE 0.02664, and WAPE 0.03537.

The CLI reconfigures stdout and stderr to UTF-8 (with backslash replacement for
unencodable third-party output), including on Windows consoles. Keep redirected
JSON/log files UTF-8; a legacy terminal font may still fail to render some
Unicode glyphs even though the emitted bytes are valid.

Set the local bundle in `.env`:

```dotenv
PRICING_MODEL_URI=artifacts/local-model
```

Start the development API:

```powershell
uv run uvicorn pricing_engine.interfaces.api.app:create_app `
  --factory --reload --port 8000
```

Validate process, stable-profile, and model state:

```powershell
Invoke-RestMethod http://localhost:8000/health/live
Invoke-RestMethod http://localhost:8000/health/ready
```

`/health/live` can return 200 without a model. `/health/ready` returns 200 when
the artifact-free `MARKET_EVIDENCE_STATISTICAL_V1` profile is available. Inspect
`model_loaded` and `model_version` separately. Check the model-backed path with:

```powershell
Invoke-RestMethod `
  'http://localhost:8000/health/ready?profile=PERFORMANCE_AWARE_EXPERIMENTAL'
```

That explicit check returns 503 until a configured compatible bundle is loaded
and warmed. Existing monitors that treated default readiness as a model check
must migrate to this profile query.

## Release-quality gate

```powershell
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv pip check
uv run pricing-engine validate `
  --input tests/fixtures/statistical/plusbnb-consumer.synthetic.json
uv run pricing-engine recommend `
  --input tests/fixtures/statistical/plusbnb-consumer.synthetic.json
uv run pricing-engine export-openapi --output docs/openapi.json
git diff --exit-code -- docs/openapi.json
uv run pytest --cov=pricing_engine --cov-report=term-missing --cov-report=xml
uv build --no-sources
docker compose config --quiet
docker compose --profile performance-experimental config --quiet
```

CI repeats these checks across supported Python versions, validates the stable
synthetic consumer contract and OpenAPI snapshot, clean-installs the built
wheel, imports production entry points, and then runs a true Docker E2E:
it generates demo data inside the API image, retrains/registers a configurable
model namespace, explicitly promotes it, starts the champion API, submits a
recommendation, and asserts the proxied artifact location, bundle checksum,
tenant/currency binding, alias, and untrusted-host rejection.

## Compose integration platform

The default Compose model starts only the API, which is enough for the stable
statistical profile:

```powershell
docker compose up --build --detach api
```

For the separate model/MLflow integration, review `.env`, then validate and
start the explicit experimental stack:

```powershell
docker compose --profile performance-experimental config --quiet
docker compose --profile performance-experimental `
  --env-file .env up --build --detach
docker compose --profile performance-experimental ps
```

Services:

| Service | URL | Expected purpose |
| --- | --- | --- |
| API | `http://localhost:8000` | Recommendation serving and health |
| MLflow | `http://localhost:5000` | Experiments and model registry |
| MinIO S3 API | `http://localhost:9000` | MLflow artifacts |
| MinIO console | `http://localhost:9001` | Local artifact inspection |
| PostgreSQL | Compose network only | MLflow metadata |

Pinned images and credentials in Compose are for local integration only. Use
managed secrets, TLS, private networking, backups, and maintained services in a
real environment.

With no `PRICING_MODEL_URI`, the API is live and ready for
`MARKET_EVIDENCE_STATISTICAL_V1`. The explicit experimental readiness query
returns 503, allowing orchestration to distinguish stable availability from
model-backed serving capability.

## Register and promote a model through MLflow

Host commands use `PRICING_MLFLOW_TRACKING_URI=http://localhost:5000`; artifact
uploads and downloads go through that MLflow endpoint rather than direct MinIO.

Create a gated candidate:

```powershell
uv run pricing-engine retrain `
  --input data/demo/observations.parquet `
  --experiment-name pricing-demand `
  --registered-model-name pricing-demand
```

The command validates point-in-time data, partitions chronologically, purges
labels whose `outcome_available_date` crosses the next boundary (plus any
configured embargo), trains, group-calibrates, tests, fingerprints the dataset,
applies promotion gates, and registers a candidate. Each stay receives equal
aggregate metric weight and interval coverage is simultaneous over all of its
snapshots. If a compatible champion exists, the pipeline loads it and evaluates
both models on the candidate's exact same immutable current holdout before
enforcing non-regression gates. It never changes the `champion` alias.

Registration also checks the ordered semantic feature-contract hash and the
registered-model namespace's immutable single-tenant/currency binding. A stale
feature contract or a mismatched tenant/currency must use a compatible retrain
and, where scope differs, a separately governed registered-model name.

The fit gives every snapshot weight `1 / snapshot_count_for_stay`, and passes
that weight to all three boosters (100-tree lower tail, 400-tree central model,
100-tree upper tail). Promotion gates evaluate predictive and economic policy
metrics both in aggregate and by required lead-time, holiday, event, and
location slices; `holiday/yes` and `holiday/no` are both mandatory.

Review the run and model version in MLflow. Promote only after reviewing
aggregate and slice metrics, interval behavior, monotonicity, drift, and known
limitations:

```powershell
uv run pricing-engine promote `
  --version <MODEL_VERSION> `
  --approved-by "Miguel Angel Sierra Hayer" `
  --registered-model-name pricing-demand
```

Promotion accepts only a `FINISHED` candidate run. It rechecks gates and
requires the SHA-256 recorded in the run parameters and model-version tags to
match. It also rejects a candidate when the `champion` alias no longer points
to the registry version used for its benchmark; retrain/re-evaluate instead of
promoting stale comparison evidence.

The SHA-256 is calculated over the serialized bundle. Registry loading treats
the model-version checksum as authoritative and verifies the download before
joblib deserialization. The MLflow PyFunc carries a checksum sidecar and checks
it before its own `joblib.load` path. Never bypass an integrity failure.

Configure the API to load the approved alias:

```dotenv
PRICING_MODEL_URI=models:/pricing-demand@champion
```

Recreate only the API and confirm experimental model state:

```powershell
docker compose --profile performance-experimental `
  up --detach --force-recreate api
Invoke-RestMethod `
  'http://localhost:8000/health/ready?profile=PERFORMANCE_AWARE_EXPERIMENTAL'
```

The Compose API overrides `PRICING_MLFLOW_TRACKING_URI` with the internal
`http://mlflow:5000` address. The API downloads through MLflow's artifact proxy;
only MLflow accesses the internal MinIO endpoint and credentials. The expected
experimental health response contains `model_loaded=true` and the loaded model
version before operators issue an experimental recommendation. A 200 default
readiness status alone only proves availability of the stable statistical
profile.

In `PRICING_ENVIRONMENT=production`, configuration fails closed unless
`PRICING_MODEL_URI` has exactly the governed alias form
`models:/<registered-model-name>@champion`, a single serving tenant is bound,
and production identity/host settings are present. Local bundle paths and
specific-version URIs are for development or diagnostics, not production
serving.

### Artifact proxy compatibility

Compose starts MLflow with `--serve-artifacts` and
`--artifacts-destination s3://mlflow/`. Host retraining and the API talk only to
MLflow; they do not receive MinIO credentials. The hard-coded Compose
credentials and any local `.env` secrets are local-only examples and must not
be used as production secret management.

MLflow does not rewrite the artifact root of an existing experiment. If
`retrain` reports that an old experiment points directly to `s3://`, `gs://`,
`wasbs://`, or `abfss://`, do not give object-store credentials to the host or
API to bypass the check. Migrate the experiment, or create a newly proxied
experiment and a newly governed registered-model namespace, for example:

```powershell
uv run pricing-engine retrain `
  --input data/demo/observations.parquet `
  --experiment-name pricing-demand-proxied-v2 `
  --registered-model-name pricing-demand-proxied-v2
```

Promote and configure the same new registered-model name. Its namespace will be
immutably bound to the first candidate's tenant and currency; a different
tenant or currency requires another new name and independently trained model.

MLflow deliberately exempts `/health` from host filtering so container probes
can reach it. This exemption does not extend to MLflow API endpoints: an
untrusted `Host` is expected to receive 403, and CI verifies that behavior. The
same Docker E2E verifies that the experiment artifact location starts with
`mlflow-artifacts:`, rather than merely assuming proxy configuration is active.

## Monitoring

Operational endpoints:

```text
GET /health/live
GET /health/ready
GET /metrics
```

Compare recent features with an approved baseline:

```powershell
uv run pricing-engine drift `
  --reference baseline.parquet `
  --current recent.parquet `
  --threshold 0.20
```

The drift output distinguishes `ok`, `drifted`, `insufficient_data`,
`constant_reference`, and contract-related states; PSI is null when it cannot
be supported. Drift initiates review and never authorizes automatic deployment.
Production monitoring must also join delayed outcomes and track calibration,
occupancy error, revenue diagnostics, low-confidence volume, latency, errors,
and human overrides by market and other critical slices.

Revenue WAPE is measured at historical observed prices. Treat it as a
predictive diagnostic only—not causal policy uplift or off-policy evaluation.
Before automated rollout, require real-data validation, multi-year rolling
backtests across market regimes, and controlled experiments or defensible OPE.

Inspect container state and logs:

```powershell
docker compose ps
docker compose logs --tail 200 api
docker compose logs --tail 200 mlflow
```

## Serving capacity and timeout behavior

The recommendation endpoint has two independent deadlines. Body receipt uses
`PRICING_REQUEST_BODY_TIMEOUT_SECONDS` (default five seconds); if the request
stream is not received in time, the API returns 408 before inference. Body size
is independently capped by `PRICING_MAX_REQUEST_BODY_BYTES`, whose default and
contractual ceiling are 1,048,576 bytes. A lower value is an explicit
operational override, appears as `effective_request_body_limit_bytes` in
capabilities, and may reject a larger contract-valid statistical document.

The container starts one Uvicorn worker. Each process loads its own model and
SHAP state and owns its own recommendation semaphore, whose default capacity is
four. `PRICING_REQUEST_TIMEOUT_SECONDS` is the separate soft inference/capacity
deadline (default five seconds):

- expiry while waiting for a semaphore slot returns 503 with `Retry-After: 1`;
- expiry after synchronous inference starts returns 504; and
- that timeout is soft: it does not kill native synchronous work, and the slot
  remains occupied until the work really completes.

Tune `PRICING_RECOMMENDATION_MAX_CONCURRENCY` and both timeout settings with
slow-upload and inference load tests. Monitor 408, 503, and 504 separately.
Adding Uvicorn workers
duplicates memory, SHAP state, and the concurrency allowance. Prefer deliberate
horizontal replicas when needed, with tenant-correct routing, replica-level
readiness, bounded aggregate concurrency, and a rolling recreation after alias
changes; models are loaded at startup, not hot-reloaded.

## Rollback

1. Identify a last verified version whose lifecycle is `superseded`.
2. Ensure no other promotion or rollback is running for the registered-model
   namespace. The implementation has compensating repair but no distributed
   lock or compare-and-swap, so alias changes require one serialized writer.
3. Restore it with a named incident approver and an auditable 10–500 character
   reason:

   ```powershell
   uv run pricing-engine rollback `
     --version <MODEL_VERSION> `
     --approved-by "Miguel Angel Sierra Hayer" `
     --reason "Restore the last verified version during incident response." `
     --registered-model-name pricing-demand
   ```

4. Recreate the API container.
5. Confirm readiness and execute a known recommendation smoke test.
6. Record the affected version, requests, time window, reason, and remediation.

Never edit a registered artifact in place.

## Shutdown and local data removal

Stop containers while preserving PostgreSQL and MinIO volumes:

```powershell
docker compose down --remove-orphans
```

`docker compose down --volumes` permanently deletes the local registry database
and artifact store. Use it only when that loss is intentional.

## Validation boundary

The repository's automated checks validate code, packaging, contracts, and
container builds. A complete live Compose recommendation requires Docker,
candidate registration, explicit promotion, and a configured model alias. Do
not claim that end-to-end deployment evidence until those steps have completed
successfully in the target environment. The coherent synthetic demo is not
causal proof, and local `.env`/Compose credentials are not a production secret
management design.
