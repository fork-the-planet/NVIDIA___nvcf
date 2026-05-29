# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import uvicorn
import traceback
from typing import Literal
from pydantic import BaseModel
from fastapi import FastAPI, status, HTTPException

NCCL_TEST_PATH = "/opt/nccl-tests/build/all_reduce_perf"
NVBANDWIDTH_PATH = "./nvbandwidth/nvbandwidth"

app = FastAPI()


def check_gpu_availability() -> str:
    """
    Check GPU availability using nvidia-smi.
    
    Returns:
        str: The nvidia-smi output

    Raises:
        HTTPException: If nvidia-smi fails or GPUs are not available
    """
    print("Checking GPU availability with nvidia-smi...")
    try:
        gpu_info = subprocess.check_output("nvidia-smi", shell=True, text=True)
        print(f"nvidia-smi output:\n{gpu_info}")
        print("nvidia-smi executed successfully, GPU is available and drivers are installed correctly.")
        return gpu_info
    except subprocess.CalledProcessError as e:
        error_msg = f"nvidia-smi failed: {str(e)}\nOutput: {e.output if hasattr(e, 'output') else 'N/A'}"
        print(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )


class HealthCheck(BaseModel):
    status: str = "OK"


@app.get("/health", tags=["healthcheck"], summary="Perform a Health Check",
         response_description="Return HTTP Status Code 200 (OK)", status_code=status.HTTP_200_OK,
         response_model=HealthCheck)
def get_health() -> HealthCheck:
    return HealthCheck(status="OK")


class TestParameters(BaseModel):
    np: int = 0
    b: str = "8"
    e: str = "128M"
    f: str = "2"
    g: str = "1"
    n: str = "20"
    npernode: int = 1
    mnnvl: bool = False
    debug: bool = False
    cluster_type: Literal["ncp-mlx5", "aws-gb200", "aws-gb300"]
@app.post("/nccl-test")
def nccl_test(tp: TestParameters) -> dict:
    try:
        # Check GPU availability
        check_gpu_availability()

        # Build the command
        # ex: /opt/amazon/openmpi/bin/mpirun --allow-run-as-root --debug-devel -bind-to none -mca plm_rsh_agent ssh_helper --mca pml ^cm,ucx --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0,veth_def_agent -x LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/nvidia/lib:/opt/amazon/ofi-nccl/lib/aarch64-linux-gnu -x PATH=$PATH:/opt/amazon/efa/bin:/usr/bin -x FI_PROVIDER=efa -x FI_EFA_USE_DEVICE_RDMA=1 -x FI_EFA_FORK_SAFE=1 -x NCCL_DEBUG=INFO -x NCCL_MNNVL_ENABLE=1 -np 16 -npernode 4 --hostfile $HOSTFILE -- /opt/nccl-tests/build/all_reduce_perf -n 20 -b 1K -e 16G -f 2 -g 1
        if tp.np > 0:
            env_flags = ""
            if tp.debug:
                env_flags += "-x NCCL_DEBUG=INFO "
            env_flags += f"-x NCCL_MNNVL_ENABLE={'1' if tp.mnnvl else '0'} "

            nccl_args = f"{NCCL_TEST_PATH} -n {tp.n} -b {tp.b} -e {tp.e} -f {tp.f} -g {tp.g}"
            hostfile_args = f"-np {tp.np} -npernode {tp.npernode} --hostfile $HOSTFILE"

            if tp.cluster_type == "aws-gb200":
                mpirun = "/opt/amazon/openmpi/bin/mpirun"
                ld_path = "/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/nvidia/lib:/opt/amazon/ofi-nccl/lib/aarch64-linux-gnu"
                path_extra = "/opt/amazon/efa/bin:/usr/bin"
                efa_flags = "-x FI_PROVIDER=efa -x FI_EFA_USE_DEVICE_RDMA=1 -x FI_EFA_FORK_SAFE=1 "
                command = (f"{mpirun} --allow-run-as-root --debug-devel -bind-to none "
                           f"-mca plm_rsh_agent ssh_helper "
                           f"--mca pml ^cm,ucx --mca btl tcp,self "
                           f"--mca btl_tcp_if_exclude lo,docker0,veth_def_agent "
                           f"-x LD_LIBRARY_PATH={ld_path} "
                           f"-x PATH={path_extra} "
                           f"{efa_flags}{env_flags}"
                           f"{hostfile_args} -- {nccl_args}")
            elif tp.cluster_type == "aws-gb300":
                command = (f"unset NCCL_NET_PLUGIN && unset NCCL_TUNER_PLUGIN && "
                           f"/usr/bin/env "
                           f"-u OMPI_MCA_btl_tcp_if_include "
                           f"-u OMPI_MCA_btl_tcp_if_exclude "
                           f"-u OMPI_MCA_oob_tcp_if_include "
                           f"-u OMPI_MCA_oob_tcp_if_exclude "
                           f"/opt/amazon/openmpi/bin/mpirun "
                           f"--allow-run-as-root "
                           f"--prefix /opt/amazon/openmpi "
                           f"-np {tp.np} "
                           f"--hostfile $HOSTFILE "
                           f"-N {tp.npernode} "
                           f"--bind-to none "
                           f"--mca plm_rsh_args \"-o StrictHostKeyChecking=no -o ConnectionAttempts=10\" "
                           f"--mca orte_keep_fqdn_hostnames true "
                           f"--mca pml ob1 "
                           f"--mca btl tcp,self "
                           f"--mca btl_tcp_if_include eth0 "
                           f"--mca oob tcp "
                           f"--mca oob_tcp_if_include eth0 "
                           f"-x PATH "
                           f"-x LD_LIBRARY_PATH "
                           f"{env_flags}"
                           f"-x NCCL_DEBUG_SUBSYS "
                           f"-x NCCL_SOCKET_IFNAME "
                           f"-x NCCL_IB_GID_INDEX "
                           f"-x NCCL_NVLS_ENABLE=1 "
                           f"-x NCCL_CUMEM_ENABLE=1 "
                           f"-x NCCL_NET_GDR_C2C=1 "
                           f"{NCCL_TEST_PATH} "
                           f"-b {tp.b} "
                           f"-e {tp.e} "
                           f"-f {tp.f} "
                           f"-n {tp.n} "
                           f"-g {tp.g} "
                           f"-N 10")
            elif tp.cluster_type == "ncp-mlx5":
                command = (f"mpirun --allow-run-as-root "
                           f"--bind-to none "
                           f"--map-by slot "
                           f"--mca plm_rsh_agent ssh_helper "
                           f"--mca routed direct "
                           f"--mca plm_rsh_no_tree_spawn 1 "
                           f"--mca pml ob1 "
                           f"--mca btl tcp,self "
                           f"--mca coll ^hcoll "
                           f"-x LD_LIBRARY_PATH -x PATH "
                           f"-x NCCL_NET_GDR_LEVEL=PHB "
                           f"-x NCCL_IB_DISABLE=0 "
                           f"-x NCCL_NVLS_DISABLE=1 "
                           f"-x NCCL_IB_GID_INDEX=3 "
                           f"{env_flags}"
                           f"{hostfile_args} -- {nccl_args}")
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported cluster_type: '{tp.cluster_type}'. Must be 'aws-gb200', 'aws-gb300', or 'ncp-mlx5'."
                )
        else:
            command = f"{NCCL_TEST_PATH} -n {tp.n} -b {tp.b} -e {tp.e} -f {tp.f} -g {tp.g}"

        print(f"Executing command: {command}")
                
        # Execute the test
        try:
            output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
            print(f"Command succeeded. Output:\n{output}")
            return {
                "status": "success",
                "output": output,
                "command": command,
                "parameters": tp.dict()
            }
        except subprocess.CalledProcessError as e:
            error_output = e.output if hasattr(e, 'output') else str(e)
            error_msg = f"Command failed with exit code {e.returncode}"
            print(f"{error_msg}\nOutput:\n{error_output}")
            return {
                "status": "failed",
                "error": error_msg,
                "output": error_output,
                "command": command,
                "parameters": tp.dict(),
                "exit_code": e.returncode
            }
            
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )


class BandwidthTestParameters(BaseModel):
    bufferSize: int = 512  # Buffer size in MiB
    testcase: str = None  # Specific testcase to run (optional)
    testcasePrefix: str = None  # Testcase prefix to run (optional)
    testSamples: int = 3  # Number of iterations
    useMean: bool = False  # Use mean instead of median
    skipVerification: bool = False  # Skip data verification
    disableAffinity: bool = False  # Disable CPU affinity
    json: bool = True  # Return JSON output
    multinode: bool = False  # Run multinode tests (requires MPI)
    np: int = 0  # Number of MPI processes (for multinode)
    verbose: bool = False  # Verbose output


@app.post("/bandwidth-test")
def bandwidth_test(params: BandwidthTestParameters) -> dict:
    try:
        # Check GPU availability
        check_gpu_availability()
        
        # Build the nvbandwidth command
        base_command = NVBANDWIDTH_PATH
        
        # Add buffer size
        command_args = [f"-b {params.bufferSize}"]
        
        # Add test samples
        command_args.append(f"-i {params.testSamples}")
        
        # Add optional flags
        if params.useMean:
            command_args.append("-m")
        if params.skipVerification:
            command_args.append("-s")
        if params.disableAffinity:
            command_args.append("-d")
        if params.json:
            command_args.append("-j")
        if params.verbose:
            command_args.append("-v")
        # Add testcase selection
        if params.testcase:
            command_args.append(f"-t {params.testcase}")
        elif params.testcasePrefix:
            command_args.append(f"-p {params.testcasePrefix}")
        
        # Construct the final command
        if params.multinode and params.np > 0:
            # Run with MPI for multinode tests
            command = (f"mpirun --allow-run-as-root -n {params.np} -mca plm_rsh_agent ssh_helper --hostfile $HOSTFILE -npernode 1 --debug-devel -- {base_command} "
                      f"{' '.join(command_args)}")
        else:
            command = f"{base_command} {' '.join(command_args)}"
        
        print(f"Executing bandwidth test command: {command}")
        
        # Execute the test
        try:
            output = subprocess.check_output(
                command, 
                shell=True, 
                text=True, 
                stderr=subprocess.STDOUT,
                timeout=300  # 5 minute timeout
            )
            print(f"Bandwidth test succeeded. Output:\n{output}")
            
            # If JSON output is requested, try to parse it
            result = {
                "status": "success",
                "output": output,
                "command": command,
                "parameters": params.dict()
            }
            
            if params.json:
                try:
                    import json
                    # Try to extract JSON from output
                    json_start = output.find('{')
                    json_end = output.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_data = json.loads(output[json_start:json_end])
                        result["bandwidth_results"] = json_data
                except Exception as json_err:
                    print(f"Could not parse JSON output: {json_err}")
                    # Keep the raw output in the response
            
            return result
            
        except subprocess.TimeoutExpired:
            error_msg = "Bandwidth test timed out after 5 minutes"
            print(error_msg)
            return {
                "status": "timeout",
                "error": error_msg,
                "command": command,
                "parameters": params.dict()
            }
        except subprocess.CalledProcessError as e:
            error_output = e.output if hasattr(e, 'output') else str(e)
            error_msg = f"Bandwidth test failed with exit code {e.returncode}"
            print(f"{error_msg}\nOutput:\n{error_output}")
            return {
                "status": "failed",
                "error": error_msg,
                "output": error_output,
                "command": command,
                "parameters": params.dict(),
                "exit_code": e.returncode
            }
            
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=int(os.getenv('WORKER_COUNT', 1)))
