# Deploy and invoke a container function

Standard four-step flow: function CREATE → DEPLOY → API key (for invoke scope) → INVOKE.

## 1. Function CREATE

Define the function in a JSON file. Minimum:

```json
{
  "name": "echo-test",
  "containerImage": "nvcr.io/0651155215864979/ncp-dev/load_tester_supreme:0.0.8",
  "inferenceUrl": "/echo",
  "inferencePort": 8000,
  "description": "smoke test echo function",
  "functionType": "DEFAULT",
  "apiBodyFormat": "CUSTOM",
  "health": {
    "protocol": "HTTP",
    "uri": "/health",
    "port": 8000,
    "timeout": "PT30S",
    "expectedStatusCode": 200
  }
}
```

```sh
nvcf-cli function create --input-file=create-fn.json
# → Function ID: <fn_id>
# → Version ID: <ver_id>
```

Save the IDs — they're needed for deploy + invoke.

## 2. Function DEPLOY

```json
{
  "functionId": "<fn_id>",
  "versionId": "<ver_id>",
  "deploymentSpecifications": [{
    "gpu": "H100",
    "instanceType": "NCP.GPU.H100_1x",
    "minInstances": 1,
    "maxInstances": 1,
    "maxRequestConcurrency": 10
  }]
}
```

Note: `gpu` is the GPU **family** (e.g. `H100`), `instanceType` is the SKU (e.g. `NCP.GPU.H100_1x`). Mismatch returns ICMS 400 "Invalid GPU".

```sh
nvcf-cli function deploy create --input-file=deploy-fn.json
```

This command **blocks** until the deployment reaches ACTIVE (default 900s timeout). When it returns rc=0, a function pod is running in the `nvcf-backend` namespace on the compute-plane cluster.

## 3. Mint an API key with `invoke_function` scope

`nvcf-cli init`'s admin token does NOT carry the `invoke_function` scope — invoke would 403 with "missing requested authorities". Mint an API key:

```sh
nvcf-cli api-key generate --description="echo-test invoke" --expires-in=1h
# → API Key: nvapi-nvcf-stg-...
```

The key is automatically saved to the CLI state file and used by the next `function invoke` call. **Don't echo the key into chat / logs**.

## 4. Function INVOKE

```json
{
  "functionId": "<fn_id>",
  "versionId": "<ver_id>",
  "requestBody": {"message": "hello"}
}
```

```sh
nvcf-cli function invoke --input-file=invoke-fn.json
# → Function invocation completed!
# → Status: fulfilled
# → Request ID: ...
```

If status is `errored`, query ICMS for the deployment's pod logs (kubectl on the compute-plane cluster) and surface to the user.

## 5. Cleanup

When the user is done with the smoke test:

```sh
nvcf-cli function delete --function-id=<fn_id> --version-id=<ver_id>
```

**Confirm with the user before deleting.** This removes both the deployment and the function record.

## Notes

- Deploy times can vary widely depending on NATS stream-init latency on cold-cluster runs (1-7 minutes observed in dev-VM tests). Be patient; don't kill the `function deploy create` command early.
- For different GPU families, change both `gpu` and `instanceType` together. Common SKUs: `NCP.GPU.H100_1x`, `NCP.GPU.H100_2x`, `NCP.GPU.A100_1x`.
- The function's container image must be pullable from the compute-plane cluster's image-credential-helper config (check `kubectl describe pod -n nvcf-backend` if pods land in `ImagePullBackOff`).
