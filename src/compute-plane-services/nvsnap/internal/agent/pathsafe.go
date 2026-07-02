/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package agent

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// resolveWithinRoot validates a caller-supplied relative path against a
// serving root and returns the fully symlink-resolved absolute path that
// is safe to open. It defends the agent's file-serving endpoints
// (/v1/checkpoints/{id}/file, /v1/captures/{hash}/file) against symlink
// traversal (nvsnap#92): a symlink committed inside a checkpoint/capture
// tree must not let a peer read files outside the tree.
//
// The earlier lexical-only check was insufficient on two counts:
//   - a bare strings.HasPrefix(target, root) lets "<root>-evil" pass, and
//   - os.Stat/os.Open follow symlinks, so a link inside the tree pointing
//     at /etc/passwd would be served despite the lexical check passing.
//
// Defense: reject "..", then resolve symlinks on BOTH the root and the
// target with filepath.EvalSymlinks (which also confirms the target
// exists), and require the resolved target to equal the resolved root or
// sit under it with a path-separator boundary. A symlink escaping the
// tree resolves to a path outside realRoot and is rejected.
//
// Returns the resolved absolute path on success. Errors are intentionally
// generic (callers map them to 400/404) so we don't leak filesystem layout.
func resolveWithinRoot(root, relPath string) (string, error) {
	cleaned := filepath.Clean("/" + relPath) // anchor so "../" can't climb above /
	if cleaned == "/" {
		return "", fmt.Errorf("empty path")
	}
	target := filepath.Join(root, cleaned)

	realRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", fmt.Errorf("resolve root: %w", err)
	}
	realTarget, err := filepath.EvalSymlinks(target)
	if err != nil {
		return "", fmt.Errorf("resolve target: %w", err)
	}

	if realTarget != realRoot && !strings.HasPrefix(realTarget, realRoot+string(os.PathSeparator)) {
		return "", fmt.Errorf("path escapes serving root")
	}
	return realTarget, nil
}
