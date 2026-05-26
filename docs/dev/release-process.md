# Release Process for Monorepo-Native Services

This page covers how a service in the NVCF umbrella monorepo gets a
new release: how versions are computed, what triggers a release, what
gets published (OCI image, Helm chart), and how to add a new service
to the release pipeline.

Pre-cutover services kept their own repo with manual chart-version /
image-version bumps. Once a service moves to the umbrella, it joins
this shared auto-versioning machinery.

## Summary

| Step | What | Where |
| --- | --- | --- |
| 1. Author commit | Conventional Commits format. The commit `type` decides whether (and how) the next release version bumps | Local |
| 2. Open MR | Pre-merge gates run: per-service Bazel build, image-push build-only, license, sonarqube, etc. | MR pipeline |
| 3. Merge to main | Default-branch pipeline runs: `compute-next-release-version-<svc>` -> `<svc>-bazel` -> `<svc>-image-push` -> (if helm) `helm-package-<svc>` + `helm-push-<svc>-*` -> `semantic-release-<svc>` -> `<svc>-slack-notify` | Default-branch pipeline |
| 4. Tag created | `semantic-release-<svc>` creates the git tag `<service-name>-v<X.Y.Z>` and a GitLab Release | End of step 3 |
| 5. Tag pipeline | A new pipeline runs against the tag ref. Same publish jobs re-run; NGC pushes are idempotent (helm-push-ngc defaults to `ngc-duplicate: skip`) | Tag pipeline |

No human `git tag` step. Steps 3-5 are automatic once the MR merges.

## Versioning

Each service has its own version line. The git tag format is:

```
<service-name>-v<X.Y.Z>
```

e.g. `nvcf-unbound-v0.7.18`, `nvcf-ratelimiter-v1.13.0`,
`nvcf-grpc-proxy-v0.4.2`.

The prefix is required because all services share one repo's tag
namespace: a bare `v0.7.18` from nvcf-unbound would collide with the
next `v0.7.18` from ratelimiter. `semantic-release-monorepo` uses
this prefix to know which commits belong to which service when
computing the next version (it scopes by subtree path).

The version itself is computed by `semantic-release` from
Conventional Commits since the previous tag. The release rules per
service (see `tools/generate-subproject-ci/main.go` for the
`.releaserc.json` template) currently bump:

- `feat:` -> minor
- `fix:` / `chore:` / `refactor:` / `style:` / `docs:` / `ci:` / `perf:` -> patch

This is more aggressive than vanilla semantic-release defaults
(`feat` / `fix` / `perf` only); the umbrella opts in to releasing on
chore + docs + ci so internal-only refactors still get a fresh image
and chart that match the latest commit. If you do not want a commit
to trigger a release, use a commit subject that does not match any
of these types (e.g. `wip:` or no type at all).

The version stamps every release artifact in lockstep:

- OCI image: `nvcr.io/<org>/<image>:<X.Y.Z>` (also `latest`,
  `<short-sha>`)
- Helm chart: `Chart.yaml`'s `version` and `appVersion` both stamp
  to `<X.Y.Z>` at package time (the in-tree `Chart.yaml` carries a
  placeholder; the packaged `.tgz` gets the real semver)
- Git tag and GitLab Release: `<service-name>-v<X.Y.Z>`

## Pipeline lanes per service

Every service that opts into the release pipeline emits this fixed
set of jobs in the generated YAML
(`tools/ci/generated-release-jobs.yml`):

### `compute-next-release-version-<svc>` (stage: init)

Runs `semantic-release --dry-run --no-ci` to compute the next
version. Writes a `NEXT_VERSION` dotenv artifact that later jobs
consume. Runs on:

- MR pipelines whose change paths touch the service
- Default-branch pipeline with the same change paths
- Tag pipelines (`$CI_COMMIT_TAG =~ /^<service-name>-v/`); the
  script falls back to `${CI_COMMIT_TAG#<service-name>-v}` when
  there is no new release-worthy commit

### `<svc>-bazel` (stage: Prerequisites)

`bazel test //...` plus `bazel build :image_index` (and chart
target if applicable). Runs on MR + default-branch + scheduled +
web pipelines.

### `<svc>-image-push` (stage: Prerequisites)

Runs `bazel build :image_push_*` on MR pipelines (validates the auth
scope and the base image fetch end-to-end without pushing) and
`bazel run :image_push_*` on default-branch and tag pipelines. The
`docker_auth_path` scoping in `tools/ci/subproject-validations.yaml`
keeps the push token off the public base-image pull (NVCF-10337).

### `helm-package-<svc>` (stage: publish)

Optional, only emitted when the service declares a `release.helm`
block. Wraps `cds/cds-components/helm-package@0.16.6` with
`chart-version: ${NEXT_VERSION}`. The packaged `.tgz` carries the
semver from `compute-next-release-version-<svc>` regardless of what
`Chart.yaml` in git says.

### `helm-push-<svc>-<target>` (stage: publish)

Optional, one job per entry in `release.helm.push_targets`. Wraps
`cds/cds-components/helm-push-ngc@0.16.6` to push the packaged chart
to the named NGC org. Default `ngc-duplicate: skip` makes re-pushing
the same version a no-op, so the tag pipeline's duplicate publish
is safe.

### `semantic-release-<svc>` (stage: publish)

Runs `semantic-release --no-ci` (real run, not dry). Creates the
git tag and GitLab Release if there is a release-worthy commit. The
`resource_group: semantic-release-notes` setting serializes per-
service runs across the project, avoiding the
`refs/notes/semantic-release` push race that hits when multiple
services release on one pipeline.

### `<svc>-slack-notify` (stage: Snapshot)

Optional, only when the service declares a `slack_channel`. Posts
to the configured Slack channel via backstage-helper after the tag
pipeline completes.

### `<svc>-sonarqube-analysis` (stage: Prerequisites)

Optional, only when the service declares a `sonarqube_project_key`.
Runs sonar-scanner against the subtree on MR + default-branch
pipelines.

## End-to-end flow (worked example: nvcf-unbound)

A developer opens an MR with a commit
`fix(nvcf-unbound): handle empty A record edge case`.

**MR pipeline** runs:

- `nvcf-unbound-bazel`: builds and tests
- `compute-next-release-version-nvcf-unbound`: dry-run says the next
  version would be `0.7.19` (patch bump from `fix:`)
- `nvcf-unbound-image-push`: `bazel build :image_push_devops` etc.
  validates the auth setup and base image fetch but does not push
- `nvcf-unbound-sonarqube-analysis`: scans the subtree

The MR gets reviewed and merged.

**Default-branch pipeline** (commit on main) runs:

1. `compute-next-release-version-nvcf-unbound` recomputes
   `NEXT_VERSION=0.7.19`
2. `nvcf-unbound-bazel` builds + tests
3. `nvcf-unbound-image-push` pushes
   `nvcr.io/nv-ngc-devops/nvcf-unbound:0.7.19` and
   `nvcr.io/0651155215864979/ncp-dev/nvcf-unbound:0.7.19`
4. `helm-package-nvcf-unbound` packages `nvcf-unbound-0.7.19.tgz`
5. `helm-push-nvcf-unbound-{ncp-dev,nv-ngc-devops,nvcf-internal}`
   push the chart to all three NGC orgs
6. `semantic-release-nvcf-unbound` creates the git tag
   `nvcf-unbound-v0.7.19` and a matching GitLab Release
7. `nvcf-unbound-slack-notify` posts to `#nv-nvcf-cicd`

**Tag pipeline** (triggered by step 6) runs jobs 1-5 again. NGC
pushes for image and helm are no-ops (the version already exists
from the default-branch pipeline). `semantic-release-nvcf-unbound`
does not re-run (it has no rule for tag pipelines).

## Adding a new service to the release pipeline

To register a new service for auto-versioned releases:

### 1. Add a `release:` block in `tools/ci/subproject-validations.yaml`

Image-only example (no Helm chart):

```yaml
- id: <service-id>
  path: src/<plane>/<service-id>
  change_paths:
    - src/<plane>/<service-id>/**/*
  release:
    service_name: <service-name>
    image_push_targets:
      - name: devops
        bazel_target: //nvidia-internal:image_push_devops
        auth:
          type: ci_var
          ci_var: NGC_DEVOPS_API_KEY
        docker_auth_path: nvcr.io/nv-ngc-devops/<service-name>
      - name: ncp_dev
        bazel_target: //nvidia-internal:image_push_ncp_dev
        auth:
          type: ci_var
          ci_var: NGC_DEVOPS_API_KEY
        docker_auth_path: nvcr.io/0651155215864979/ncp-dev/<service-name>
    slack_channel: C08S6KLCEJH
    sonarqube_project_key: <project-key>   # optional
```

Image + Helm example (see nvcf-unbound for the working version):

```yaml
- id: <service-id>
  path: src/<plane>/<service-id>
  change_paths:
    - src/<plane>/<service-id>/**/*
  release:
    service_name: <service-name>
    image_push_targets:
      # ... same as image-only ...
    helm:
      chart_path: deploy           # relative to subtree root
      push_targets:
        - name: ncp-dev
          ngc_path: 0651155215864979/ncp-dev
          ngc_key_var: NGC_DEVOPS_API_KEY
        - name: nv-ngc-devops
          ngc_path: nv-ngc-devops
          ngc_key_var: NGC_DEVOPS_API_KEY
        - name: nvcf-internal
          ngc_path: "0544956542906249"
          ngc_key_var: NGC_PUSH_SERVICE_KEY_NVCF_INTERNAL
    slack_channel: C08S6KLCEJH
```

### 2. Regenerate the pipeline YAML

```bash
go run -C tools/generate-subproject-ci . \
  --config ../ci/subproject-validations.yaml \
  --release-output ../ci/generated-release-jobs.yml
```

Commit the regenerated `tools/ci/generated-release-jobs.yml` along
with the validations change. The `check-release-pipeline-generated`
CI job verifies the file is in sync.

### 3. Add a `<svc>-bazel` job in the umbrella `.gitlab-ci.yml`

Mirror an existing entry (e.g. `nats-auth-callout-bazel`). Sets
`SUBTREE` and the `changes:` filter.

### 4. Push an anchor tag at main HEAD

If the service had previous releases from its upstream repo, push an
anchor tag matching the last upstream version so `semantic-release`
picks up from a recognized starting point:

```bash
git tag -a <service-name>-v<last-upstream-version> -m "anchor: <svc> cutover continuity"
git push origin <service-name>-v<last-upstream-version> -o ci.skip
```

The `-o ci.skip` prevents the anchor tag from firing a publish
pipeline; the existing artifacts at that version stay untouched.

If the service is genuinely new (no prior releases), skip this step.
The first release will be `1.0.0` (semantic-release default).

### What the first release after cutover looks like

The cutover MR's commit type decides whether the first
umbrella-managed release bumps minor or patch:

- `feat(<svc>): cut over to monorepo-native` -> `feat:` -> minor
  bump. Example: anchor `<svc>-v0.5.8` plus a `feat:` cutover
  commit produces `<svc>-v0.6.0`, not `<svc>-v0.5.9`. The patch
  line resets to 0.
- `chore(<svc>): cut over to monorepo-native` -> `chore:` -> patch
  bump. Same anchor produces `<svc>-v0.5.9`.

Both are defensible. The cutover changes how an artifact is built
and released (new pipeline, new image registry token scoping,
chart version derivation), so calling it `feat:` carries real
release-flow signal for operators reading the changelog. If you
care more about keeping the patch line continuous with the
upstream's last release, use `chore:`.

The first six native cutovers in the umbrella all used `feat:`
(grpc-proxy, ratelimiter, nats-auth-callout, function-autoscaler,
http-invocation, nvcf-unbound), so each landed on the next minor
above the anchor. Future cutovers should follow the same convention
unless you have a specific reason to prefer patch continuity.

## Troubleshooting

### Tag pipeline fails YAML validation with "X needs Y but Y is not in any previous stage"

Some release job has a hard `needs:` on a job whose rules do not
include `$CI_COMMIT_TAG`. Either add the tag rule to the depended
job's rules or mark the `needs:` as `optional: true`. Fixed for the
helm-package case in NVCF-10337 follow-up.

### Slack notification says "tag: unknown"

The `git describe` fallback used to fail on shallow CI clones. The
template now prefers `${CI_COMMIT_TAG#<service-name>-v}` first, with
`git describe` as a fallback. If you still see this, the tag pipeline
fired without a matching `<service-name>-v*` tag.

### Two slack notifications per release

Two pipelines (default-branch + tag) both fired the slack-notify
rule. The template now scopes slack-notify to `$CI_COMMIT_TAG` only
so it fires once. If you see two, the rule must have an unintended
`$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH` entry.

### "Cannot lock ref refs/notes/semantic-release"

Multiple `semantic-release-<svc>` jobs raced on the shared notes
ref. The template sets `resource_group: semantic-release-notes` so
GitLab serializes them. Retrying the failed job usually succeeds
since the ref has settled by then.

### Image push 403 on the base image fetch

The push token's scope leaked to the public base-image pull. Make
sure `docker_auth_path` on the affected target is set to the
push destination's full path (e.g.
`nvcr.io/nv-ngc-devops/<image>`), not just `nvcr.io`. Without the
scope, `rules_oci` applies the scoped push token to every nvcr.io
URL it pulls during the build, including the public distroless
base, and 403s. Fixed in NVCF-10337 / !294.

## Related references

- `tools/ci/subproject-validations.yaml`: source of truth for which
  services have a release block and how they are configured.
- `tools/generate-subproject-ci/main.go`: the generator that emits
  the YAML in `tools/ci/generated-release-jobs.yml`.
- `tools/ci/generated-release-jobs.yml`: generated output the
  umbrella `.gitlab-ci.yml` includes; do not hand-edit.
- `BAZEL.md` at the repo root: Bazel-build-related conventions.
- `deploy/stacks/self-managed/.gitlab-ci.yml`: the self-managed
  stack's release flow (helmfile-based bundle, not individual chart
  push). Different shape from service releases; same semantic-release
  driver.
