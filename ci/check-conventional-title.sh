#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate that a merge request / pull request title is a Conventional Commit.
#
# Rationale: nvcf/nvcf releases run on semantic-release. When an MR/PR is
# squash-merged, the squash commit subject is derived from the title. If the
# title is not a Conventional Commit (for example a raw branch name like
# "Feat/bump cassandra migration"), semantic-release cannot parse a release
# type and silently skips the version bump. This check rejects such titles
# before merge so a release is never skipped by an unparseable subject.
#
# Policy: <type>[optional (scope)][optional !]: <subject>
#   - type is lowercase and one of the NVCF Conventional Commit types
#   - a scope is optional (scope is not currently mandated)
#   - a "!" breaking-change marker is allowed
#   - the subject after ": " must be non-empty
#
# Input: the title is read from $TITLE, or from the first argument. On GitLab
# the caller passes $CI_MERGE_REQUEST_TITLE; on GitHub the workflow passes the
# pull request title via the TITLE env var (never inline, to avoid injection).
set -euo pipefail

title="${TITLE:-${1:-}}"

# Strip a leading Draft:/WIP: marker (GitLab/GitHub work-in-progress prefixes)
# and surrounding whitespace before validating the real title.
title="$(printf '%s' "${title}" | sed -E 's/^[[:space:]]*(\[?[Dd]raft\]?|WIP|wip)[:[:space:]]+//; s/^[[:space:]]+//; s/[[:space:]]+$//')"

if [ -z "${title}" ]; then
  echo "ERROR: no title provided to check-conventional-title.sh (set \$TITLE or pass an argument)." >&2
  exit 1
fi

# NVCF Conventional Commit types (see AGENTS.md "Commit Messages").
types='feat|fix|perf|docs|build|test|refactor|ci|chore|style|revert'

# <type>(<optional scope>)<optional !>: <subject starting with a non-space char>
# The subject must begin with a non-whitespace character so a colon followed by
# only spaces (e.g. "fix:  ") is rejected as an empty subject.
pattern="^(${types})(\([a-z0-9][a-z0-9._/-]*\))?!?: [^[:space:]].*"

if printf '%s' "${title}" | grep -qE "${pattern}"; then
  echo "OK: title is a Conventional Commit: ${title}"
  exit 0
fi

cat >&2 <<EOF
ERROR: the merge request / pull request title is not a Conventional Commit.

  Title checked: ${title}

Required format:
  <type>(<optional scope>): <subject>

  - <type> must be lowercase and one of:
      ${types//|/, }
  - a scope in parentheses is optional, e.g. fix(cassandra):
  - a "!" may follow for a breaking change, e.g. feat(api)!:
  - the subject after ": " must not be empty

Examples:
  fix(cassandra): bump migration image to 0.11.0
  feat(worker): add NATS auth violation surfacing
  chore(ci): pin bazel-ci image tag

Why this is enforced:
  On squash-merge the title becomes the commit subject that semantic-release
  reads. A non-conventional subject (for example a raw branch name such as
  "Feat/bump cassandra migration") cannot be parsed, so the release version is
  silently skipped. Fixing the title keeps releases from being missed.
EOF
exit 1
