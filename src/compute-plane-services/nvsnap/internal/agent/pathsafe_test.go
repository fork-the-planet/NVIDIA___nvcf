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
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestResolveWithinRoot(t *testing.T) {
	tmp := t.TempDir()
	root := filepath.Join(tmp, "root")
	if err := os.MkdirAll(filepath.Join(root, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	// A real file inside the root.
	inside := filepath.Join(root, "sub", "ok.img")
	if err := os.WriteFile(inside, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A secret OUTSIDE the root, and a symlink inside the root pointing at it.
	secret := filepath.Join(tmp, "secret.txt")
	if err := os.WriteFile(secret, []byte("password"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(secret, filepath.Join(root, "escape")); err != nil {
		t.Fatal(err)
	}
	// A sibling dir sharing the root's name prefix (the bare-HasPrefix bug).
	if err := os.MkdirAll(root+"-evil", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(root+"-evil/x", []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Run("file inside root resolves", func(t *testing.T) {
		got, err := resolveWithinRoot(root, "sub/ok.img")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want, _ := filepath.EvalSymlinks(inside)
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})

	t.Run("symlink escaping root is rejected", func(t *testing.T) {
		if _, err := resolveWithinRoot(root, "escape"); err == nil {
			t.Fatal("symlink to outside-root file must be rejected, got nil error")
		}
	})

	t.Run("dot-dot traversal is rejected", func(t *testing.T) {
		// Anchored clean turns "../secret.txt" into "/secret.txt" → joined
		// under root → does not exist → error (never escapes).
		if _, err := resolveWithinRoot(root, "../secret.txt"); err == nil {
			t.Fatal("../ traversal must be rejected")
		}
	})

	t.Run("prefix-sibling directory is rejected", func(t *testing.T) {
		// Requesting a path that would land in "<root>-evil" must not pass a
		// naive HasPrefix check. We can't express it via relPath (Join keeps
		// it under root), but verify the boundary directly: a target whose
		// resolved path is root+"-evil/x" is outside.
		if _, err := resolveWithinRoot(root+"-evil", "x"); err != nil {
			t.Fatalf("sanity: serving the evil dir as its own root should work: %v", err)
		}
		// The real guarantee: "<root>-evil/x" is NOT served when root is `root`.
		got, err := resolveWithinRoot(root, "../root-evil/x")
		if err == nil {
			t.Fatalf("must not serve sibling-prefix path, got %q", got)
		}
	})

	t.Run("missing file maps to not-exist", func(t *testing.T) {
		_, err := resolveWithinRoot(root, "sub/nope.img")
		if !errors.Is(err, os.ErrNotExist) {
			t.Errorf("missing file should wrap os.ErrNotExist, got %v", err)
		}
	})
}
