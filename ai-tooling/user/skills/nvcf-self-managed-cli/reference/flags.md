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
