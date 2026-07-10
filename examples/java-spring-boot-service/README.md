# java-spring-boot-service

Reference Spring Boot service built with Bazel. It exists to exercise and
demonstrate the Java-on-Bazel foundation in `//rules/java` and the Maven wiring
in the root `MODULE.bazel`. Copy it as the starting point for a new Java service.

## Layout

```
BUILD.bazel                         nvcf_java_library + nvcf_spring_boot_image + JUnit 5 test
src/main/java/com/nvidia/nvcf/example/
  ExampleApplication.java           @SpringBootApplication entry point
  HelloController.java              trivial @RestController
src/test/java/.../HelloControllerTest.java
```

## Build and test

```
bazel test //examples/java-spring-boot-service:hello_controller_test
bazel build //examples/java-spring-boot-service:image_index
```

`nvcf_spring_boot_image` generates the same target set as the Go image macro:

- `:image` host-arch image
- `:image_index` multi-arch index (amd64 + arm64)
- `:image_load` load into a local Docker daemon
- `:image.tar` tarball
- `:image_push` when the macro is given `registry = ...`

Run it locally:

```
bazel run //examples/java-spring-boot-service:image_load
docker run --rm -p 8080:8080 examples/java-spring-boot-service:latest
curl localhost:8080/hello
```

## Creating a new Java service from this template

1. Copy this directory to `src/<plane>/<your-service>` (or another loadable
   path; anything under a directory listed in `.bazelignore` is skipped).
2. Rename the package and `main_class`.
3. Add the Maven coordinates your code imports to `maven.install` in the root
   `MODULE.bazel`, then re-pin: `bazel run @maven//:pin`. List the artifacts you
   import directly in `deps` so the strict-deps header compiler resolves them.
4. Add a subproject entry in `tools/ci/subproject-validations.yaml` (model it on
   `java-example-service`) so the build and test run in CI.
5. To publish images, pass `registry = ...` to `nvcf_spring_boot_image` and add a
   `release:` block to the subproject entry (model it on `grpc-proxy`). The
   release machinery is image-format agnostic: it stamps and pushes the
   `:image_index` the same way it does for Go and Rust services.

## Using nv-boot or other internal dependencies

This example depends only on public Maven Central artifacts so it builds
anywhere, including the public GitHub mirror. A real internal service that
depends on nv-boot adds the internal Artifactory Maven virtual repository to the
`repositories` list in `maven.install` and lists the nv-boot coordinates in
`artifacts`. See `//rules/java/README.md` for that wiring.
