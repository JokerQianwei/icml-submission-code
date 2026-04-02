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
import csv
from pathlib import Path

import yaml

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


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    auc_sum = 0.0
    prev = 0.0
    called = 0
    ordered_results = sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False)
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = sorted(temp_result, key=lambda kv: kv[1][0], reverse=True)[:top_n]
        top_n_now = sum(item[1][0] for item in temp_result) / len(temp_result)
        auc_sum += freq_log * (top_n_now + prev) / 2.0
        prev = top_n_now
        called = idx
    temp_result = sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True)[:top_n]
    top_n_now = sum(item[1][0] for item in temp_result) / len(temp_result)
    auc_sum += (len(buffer) - called) * (top_n_now + prev) / 2.0
    if finish and len(buffer) < max_oracle_calls:
        auc_sum += (max_oracle_calls - len(buffer)) * top_n_now
    return auc_sum / max_oracle_calls


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


def default_output_dir():
    return Path(__file__).resolve().parents[0] / "main/genmol/results"


def find_yaml_file(output_dir, oracle, seed):
    canonical = output_dir / f"results_GenMol_{oracle}_{seed}.yaml"
    if canonical.exists():
        return canonical
    # Some TDC oracles append a suffix (e.g., "_current"), so we match fallback names.
    candidates = sorted(output_dir.glob(f"results_GenMol_{oracle}*_{seed}.yaml"))
    if candidates:
        return candidates[0]
    return canonical


def compute_metrics_from_yaml(result_file, max_oracle_calls, freq_log):
    with open(result_file, "r", encoding="utf-8") as handle:
        result_dict = yaml.safe_load(handle) or {}

    if not isinstance(result_dict, dict):
        raise ValueError(f"Unexpected YAML format in {result_file}")
    if not result_dict:
        raise ValueError(f"Empty result file: {result_file}")

    rows = []
    for smiles, values in result_dict.items():
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            raise ValueError(
                f"Unexpected score/call format for {smiles} in {result_file}: {values}"
            )
        score = float(values[0])
        call_idx = int(values[1])
        rows.append((smiles, score, call_idx))

    rows.sort(key=lambda x: x[2])
    rows_budget = [row for row in rows if row[2] <= max_oracle_calls]

    if not rows_budget:
        raise ValueError(
            f"No molecule with call_idx <= {max_oracle_calls} in {result_file}"
        )

    top_scores = sorted([row[1] for row in rows_budget], reverse=True)[:10]
    final_top10 = float(sum(top_scores) / len(top_scores))
    final_top1 = float(top_scores[0])

    mol_dict = {
        smiles: [float(score), int(call_idx)] for smiles, score, call_idx in rows_budget
    }
    auc_top1 = float(top_auc(mol_dict, 1, True, freq_log, max_oracle_calls))
    auc_top10 = float(top_auc(mol_dict, 10, True, freq_log, max_oracle_calls))

    return {
        "n_molecules_total": int(len(rows)),
        "n_molecules_budget": int(len(rows_budget)),
        "auc_top1": auc_top1,
        "auc_top10": auc_top10,
        "final_top1": final_top1,
        "final_top10": final_top10,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracles", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--freq_log", type=int, default=100)
    parser.add_argument("--skip_missing", action="store_true")
    args = parser.parse_args()

    oracles = parse_oracles(args.oracles)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else default_output_dir().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for oracle in oracles:
        yaml_path = find_yaml_file(output_dir, oracle, args.seed)
        csv_path = output_dir / f"{oracle}_{args.seed}.csv"
        if not yaml_path.exists():
            missing.append(str(yaml_path))
            continue

        metrics = compute_metrics_from_yaml(
            yaml_path,
            max_oracle_calls=args.max_oracle_calls,
            freq_log=args.freq_log,
        )
        rows.append(
            {
                "oracle": oracle,
                "seed": args.seed,
                "auc_top1": metrics["auc_top1"],
                "auc_top10": metrics["auc_top10"],
                "final_top1": metrics["final_top1"],
                "final_top10": metrics["final_top10"],
                "n_molecules_total": metrics["n_molecules_total"],
                "n_molecules_budget": metrics["n_molecules_budget"],
                "yaml_file": str(yaml_path),
                "csv_file": str(csv_path),
            }
        )

    if missing and not args.skip_missing:
        missing_list = "\n".join(missing)
        raise FileNotFoundError(
            f"Missing {len(missing)} result file(s):\n{missing_list}\n"
            "Use --skip_missing to aggregate available tasks only."
        )

    if not rows:
        raise RuntimeError("No valid result file found for aggregation.")

    rows.sort(key=lambda item: item["oracle"])
    metrics_file = output_dir / f"pmo_metrics_seed{args.seed}.csv"
    with open(metrics_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "oracle",
                "seed",
                "auc_top1",
                "auc_top10",
                "final_top1",
                "final_top10",
                "n_molecules_total",
                "n_molecules_budget",
                "yaml_file",
                "csv_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    sum_auc_top1 = float(sum(item["auc_top1"] for item in rows))
    sum_auc_top10 = float(sum(item["auc_top10"] for item in rows))
    sum_final_top1 = float(sum(item["final_top1"] for item in rows))
    sum_final_top10 = float(sum(item["final_top10"] for item in rows))
    n_tasks = len(rows)
    summary_row = {
        "seed": args.seed,
        "n_tasks": int(n_tasks),
        "sum_auc_top1": sum_auc_top1,
        "sum_auc_top10": sum_auc_top10,
        "sum_final_top1": sum_final_top1,
        "sum_final_top10": sum_final_top10,
        "mean_auc_top1": float(sum_auc_top1 / n_tasks),
        "mean_auc_top10": float(sum_auc_top10 / n_tasks),
        "mean_final_top1": float(sum_final_top1 / n_tasks),
        "mean_final_top10": float(sum_final_top10 / n_tasks),
        "max_oracle_calls": args.max_oracle_calls,
        "freq_log": args.freq_log,
        "missing_tasks": int(len(missing)),
    }

    summary_file = output_dir / f"pmo_summary_seed{args.seed}.csv"
    with open(summary_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Wrote per-task metrics to: {metrics_file}")
    print(f"Wrote summary metrics to: {summary_file}")
    print(
        f"n_tasks={summary_row['n_tasks']} | "
        f"sum_auc_top1={summary_row['sum_auc_top1']:.3f} | "
        f"sum_auc_top10={summary_row['sum_auc_top10']:.3f} | "
        f"sum_final_top1={summary_row['sum_final_top1']:.3f} | "
        f"sum_final_top10={summary_row['sum_final_top10']:.3f}"
    )
    if missing:
        print(f"Skipped {len(missing)} missing task(s).")


if __name__ == "__main__":
    main()
