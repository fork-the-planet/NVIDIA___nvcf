# `nvcf-cli` flag reference

## Global flags (every subcommand)

| Flag | Purpose | Default |
|---|---|---|
| `--config FILE` | Config file path | `$HOME/.nvcf-cli.yaml` |
| `--debug` | Verbose HTTP logging on stderr | `false` |
| `--json` | JSONL events on stderr | auto (off in TTY, on under `--non-interactive`) |

## `self-hosted` parent (persistent across subcommands)

| Flag | Purpose | Default |
|---|---|---|
| `--stack=…` | Bundle source: local path, git URL, or `oci://` URL | embedded OCI URL pinned by CLI version |
| `--env=local\|prd\|…` | Helmfile environment name | `local` for dev builds, `prd` for releases |
| `--non-interactive` | Disable all stdin prompts | `false` |
| `--token=$JWT` | Admin JWT, overrides stored session | — |
| `--no-apply` | `install` only — emit YAML only, don't kubectl apply | `false` |
| `--output=text\|json` | Legacy alias for `--json` (deprecated, removed in next major) | `text` |
| `--plain` | Force plain streaming output | auto-detect |
| `--wait DURATION` | `check` only; block until pass | — |
| `--control-plane-context CTX` | kubectl context for control plane (REQ-20) | current context |
| `--compute-plane-context CTX` | kubectl context for compute plane (REQ-20) | current context |
| `--icms-url URL` | Public ICMS URL; required when contexts differ | derived from `base_http_url` |
| `--local-only` | `check --pre` only; skip all kubectl contact | `false` |

## `up`-specific

| Flag | Purpose | Default |
|---|---|---|
| `--cluster-name NAME` | ICMS cluster row identifier (required) | — |
| `--nca-id ID` | NCA account ID | `nvcf-default` |
| `--region REGION` | Cluster region | `us-west-1` |
| `--plan-only` | Dry-run; emit phase plan + ETAs without changes | `false` |

## `status`-specific

| Flag | Purpose | Default |
|---|---|---|
| `--cluster-name NAME` | Limit to one compute plane | — (control + all clusters) |
| `--watch` | Live re-render | `false` |
| `--watch-interval DUR` | `--watch` cadence | `5s` |
| `--component NAME` | Filter components panel | all |
| `--show-events DUR` | Recent events window | `5m` |
| `--no-events` | Skip events panel | `false` |

## `cluster register`-specific

| Flag | Purpose | Default |
|---|---|---|
| `--name NAME` | Cluster name (required) | — |
| `--nca-id ID` | NCA ID (required) | — |
| `--region REGION` | Region (required, non-empty) | — |
| `--icms-url URL` | ICMS endpoint | from config |
| `--ignore-existing` | Match-or-create instead of fail-on-exists | `false` |
| `--identity-source psat\|spire` | Identity source | `psat` |

## `function create`-specific

`--input-file FILE` is recommended for repeatable function definitions. For ad-hoc creates:

| Flag | Purpose | Default |
|---|---|---|
| `--name NAME` | Function name | - |
| `--image IMAGE` | Container image | - |
| `--inference-url URI` | Container inference endpoint | - |
| `--inference-port PORT` | Container inference port | - |
| `--function-type DEFAULT\|STREAMING\|LLM` | Function type | `DEFAULT` |
| `--helm-chart CHART` | Helm chart URL or OCI reference. Can be used with `--function-type=LLM` for chart-packaged LLM workloads. | - |
| `--helm-chart-service NAME` | Kubernetes Service name exposed by the chart. Required when `--helm-chart` is set. | - |
| `--models NAME:VERSION:URI` | Standard model artifact; repeatable | - |
| `--llm-model SPEC` | LLM model config; format `name=<model>,uris=<uri>|<uri>,routingMethod=<round_robin|power_of_two|random>,tokenRateLimit=<limit>`; repeatable. Token limits use `<value>-<unit>` with `S`, `M`, `H`, `D`, or `W`, for example `1000-S`. Use input JSON for combined token limits because inline specs use commas as field separators. | - |

In JSON, LLM functions set `functionType: "LLM"` and model routing metadata under `models[].llmConfig`. `llmConfig.uris` declares the OpenAI-compatible upstream paths exposed by the model. Current supported paths are `/v1/chat/completions`, `/v1/responses`, and `/v1/embeddings`. `llmConfig.routingMethod` accepts `round_robin`, `power_of_two`, or `random`.
`llmConfig.tokenRateLimit` accepts one or more comma-separated positive integer token limits in `<value>-<unit>` format. Supported units are `S` (seconds), `M` (minutes), `H` (hours), `D` (days), and `W` (weeks). Use distinct units when combining limits, for example `1000-S,5000-M,100000-H,500000-D,1000000-W` in input JSON.

Helm chart packaging is independent of `functionType`. For a Helm-chart backed LLM function, set `functionType: "LLM"`, `helmChart`, `helmChartServiceName`, and `models[].llmConfig` in the same create request. `inferencePort` must be the Kubernetes Service port exposed by `helmChartServiceName`.

LLM invocation requests use `model: "<function-id>/<model-name>"`. The function ID selects the NVCF function, and the model name is forwarded upstream. Chat completions and Responses API requests can use `x-multi-turn-session-id` for session stickiness; embeddings requests do not.

## `function update`-specific

| Flag | Purpose | Default |
|---|---|---|
| `--tags TAG[,TAG]` | Replace function tags | - |
| `--llm-model-update SPEC` | LLM model update; format `name=<model>,routingMethod=<round_robin|power_of_two|random>,tokenRateLimit=<limit>`; repeatable. Token limit example: `1000-S`. Use input JSON for combined token limits. | - |

In JSON, `function update` accepts `modelUpdates[]` entries with `modelName` and `llmConfig.routingMethod` and/or `llmConfig.tokenRateLimit`. `uris` are create-time model metadata and are not part of model updates.

## `function deploy create`-specific

`--input-file FILE` is recommended (full JSON spec). For ad-hoc one-shot use:

| Flag | Purpose | Default |
|---|---|---|
| `--function-id ID` | Function ID (required) | — |
| `--gpu NAME` | GPU family | `H100` |
| `--instance-type TYPE` | SKU | `NCP.GPU.H100_1x` |
| `--min-instances N` | Min replicas | `1` |
| `--max-instances N` | Max replicas | `1` |
| `--max-request-concurrency N` | Concurrency | `10` |
| `--clusters NAME[,NAME]` | Target specific compute clusters | (any matching SKU) |
