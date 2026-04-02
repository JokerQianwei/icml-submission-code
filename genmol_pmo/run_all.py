# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


ORACLES = [
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo",
]


def parse_csv_list(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_oracles(raw):
    if raw is None:
        return list(ORACLES)
    parsed = parse_csv_list(raw)
    invalid = [item for item in parsed if item not in ORACLES]
    if invalid:
        raise ValueError(
            f"Unknown oracle(s): {invalid}. Supported oracles: {', '.join(ORACLES)}"
        )
    return parsed


def get_repo_root():
    return Path(__file__).resolve().parents[3]


def default_output_dir(repo_root):
    return repo_root / "scripts/exps/pmo/main/genmol/results"


def expected_outputs(output_dir, oracle, seed):
    return (
        output_dir / f"results_GenMol_{oracle}_{seed}.yaml",
        output_dir / f"{oracle}_{seed}.csv",
    )


def build_command(args, oracle):
    command = [
        sys.executable,
        "scripts/exps/pmo/run.py",
        "-o",
        oracle,
        "-s",
        str(args.seed),
        "--max_oracle_calls",
        str(args.max_oracle_calls),
        "--freq_log",
        str(args.freq_log),
        "--n_jobs",
        str(args.n_jobs),
    ]
    if args.config is not None:
        command.extend(["-c", args.config])
    if args.output_dir is not None:
        command.extend(["--output_dir", args.output_dir])
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_concurrent", type=int, default=8)
    parser.add_argument("--oracles", type=str, default=None)
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--freq_log", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("-c", "--config", type=str, default="hparams.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--poll_interval", type=float, default=5.0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if args.max_concurrent <= 0:
        raise ValueError("--max_concurrent must be a positive integer.")
    if args.poll_interval <= 0:
        raise ValueError("--poll_interval must be positive.")

    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise ValueError("No GPU is specified. Use --gpus, e.g., --gpus 0,1,2,3,4,5,6,7")

    oracles = parse_oracles(args.oracles)

    repo_root = get_repo_root()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else default_output_dir(repo_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir).resolve() if args.log_dir is not None else output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = deque()
    for oracle in oracles:
        yaml_path, csv_path = expected_outputs(output_dir, oracle, args.seed)
        if args.skip_existing and yaml_path.exists() and csv_path.exists():
            print(f"[skip] {oracle}: found {yaml_path.name} and {csv_path.name}")
            continue
        tasks.append(oracle)

    if not tasks:
        print("No task left to run.")
        return 0

    max_concurrent = min(args.max_concurrent, len(gpus), len(tasks))
    worker_gpus = gpus[:max_concurrent]
    print(
        f"Start PMO runs: {len(tasks)} task(s), seed={args.seed}, "
        f"max_concurrent={max_concurrent}, gpus={worker_gpus}"
    )

    running = {}
    results = []
    all_start = time.time()

    while tasks or running:
        for gpu in worker_gpus:
            if gpu in running or not tasks:
                continue

            oracle = tasks.popleft()
            command = build_command(args, oracle)
            log_path = log_dir / f"run_{oracle}_seed{args.seed}.log"
            log_handle = open(log_path, "w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = {
                "oracle": oracle,
                "process": process,
                "start_time": time.time(),
                "log_path": log_path,
                "log_handle": log_handle,
            }
            print(
                f"[launch] oracle={oracle} seed={args.seed} gpu={gpu} pid={process.pid} "
                f"log={log_path}"
            )

        finished_gpus = []
        for gpu, job in running.items():
            return_code = job["process"].poll()
            if return_code is None:
                continue

            elapsed = time.time() - job["start_time"]
            job["log_handle"].close()
            finished_gpus.append(gpu)
            results.append(
                {
                    "oracle": job["oracle"],
                    "gpu": gpu,
                    "return_code": return_code,
                    "elapsed_sec": elapsed,
                    "log_path": str(job["log_path"]),
                }
            )
            status = "ok" if return_code == 0 else "failed"
            print(
                f"[done] oracle={job['oracle']} gpu={gpu} status={status} "
                f"rc={return_code} elapsed={elapsed/60.0:.1f} min"
            )

        for gpu in finished_gpus:
            running.pop(gpu, None)

        if tasks or running:
            time.sleep(args.poll_interval)

    total_minutes = (time.time() - all_start) / 60.0
    failed = [item for item in results if item["return_code"] != 0]
    print(f"Finished all scheduled PMO runs in {total_minutes:.1f} min.")

    if failed:
        print("Failed tasks:")
        for item in failed:
            print(
                f"  oracle={item['oracle']} gpu={item['gpu']} rc={item['return_code']} "
                f"log={item['log_path']}"
            )
        return 1

    print("All tasks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
