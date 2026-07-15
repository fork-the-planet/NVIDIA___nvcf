#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fail a pull request that does not reference an associated issue.
#
# A PR passes if its title or body (with HTML comments stripped, so the PR
# template's own guidance/examples do not count) contains any of:
#   - a GitHub issue reference: "#123", "Closes/Fixes/Resolves/Relates to #123",
#     or "NVIDIA/nvcf#123"
#   - a tracker key: NVCF-1234, NVCFCLUST-1234, NVCT-1234, NVBUG-1234, NVBUGS-1234
#   - the escape hatch "NO-REF" (use only when no issue genuinely applies)
#
# Inputs (from the workflow, passed via env to avoid shell injection from the
# attacker-controlled PR body):
#   PR_TITLE, PR_BODY
set -euo pipefail

# Strip HTML comments so the template's guidance and examples (which live inside
# <!-- --> and mention NO-REF / example keys) are not mistaken for a real ref.
visible="$(
  PR_TITLE="${PR_TITLE:-}" PR_BODY="${PR_BODY:-}" python3 - <<'PY'
import os, re
text = os.environ.get("PR_TITLE", "") + "\n" + os.environ.get("PR_BODY", "")
print(re.sub(r"<!--.*?-->", "", text, flags=re.S))
PY
)"

# Escape hatch: explicit NO-REF written by the author.
if grep -qE '(^|[^A-Za-z0-9])NO-REF([^A-Za-z0-9]|$)' <<<"$visible"; then
  echo "check-pr-issue: NO-REF present; issue requirement waived."
  exit 0
fi

# Accepted issue references. A bare "#N" is intentionally NOT accepted on its
# own (it false-matches prose like "#1 priority" or "#3 steps"); an explicit
# action keyword, a cross-repo reference, or a tracker key is required.
#   - action keyword + optional owner/repo + issue: "Closes #123",
#     "Fixes NVIDIA/nvcf#123", "Resolves #123", "Relates to #123"
#   - explicit cross-repo GitHub issue: "NVIDIA/nvcf#123"
#   - tracker key: NVCF-1234, NVCFCLUST-1234, NVCT-1234, NVBUG-1234, NVBUGS-1234
if grep -qiE '(close[sd]?|fix(e[sd])?|resolve[sd]?|relate[sd]?[[:space:]]+to)[[:space:]:]+([A-Za-z0-9._/-]+)?#[0-9]+' <<<"$visible" \
  || grep -qE '\bNVIDIA/nvcf#[0-9]+' <<<"$visible" \
  || grep -qE '\b(NVCF|NVCFCLUST|NVCT|NVBUG|NVBUGS)-[0-9]+\b' <<<"$visible"; then
  echo "check-pr-issue: PR references an associated issue."
  exit 0
fi

cat >&2 <<'MSG'
ERROR: this pull request does not reference an associated issue.

Add one of the following to the PR description:
  - a GitHub issue, e.g. "Closes #123" or "Fixes NVIDIA/nvcf#123"
  - a tracker key, e.g. "NVCF-1234"
  - or "NO-REF" if no issue genuinely applies (use sparingly, per AGENTS.md)
MSG
exit 1
