import json
from collections import defaultdict
from pathlib import Path

def load_results(path="ablations/results_full_dynamic_critic.json"):
    with open(path, "r") as f:
        return json.load(f)["results"]

import re

def get_subprob(sample):
    return sample.get("agents", {}).get("subproblem", {}).get("output", "")

def parse_subproblems(raw):
    # raw = sample.get("agents", {}).get("subproblem", {}).get("output", "")

    # Remove triple backticks and language identifiers
    clean = re.sub(r"^```(json)?", "", raw.strip(), flags=re.IGNORECASE).strip("` \n")

    try:
        parsed = json.loads(clean)
        subproblems = parsed.get("subproblems", [])
    except Exception:
        return []

    # Normalize and filter subproblem clause types
    allowed_clauses = {"orderby", "groupby", "join", "union", "limit", "having", "intersect", "except"}

    clauses = []
    for sub in subproblems:
        clause = sub.get("clause", "").lower().replace(" ", "")
        if clause in allowed_clauses:
            clauses.append(clause)

    return clauses


def group_by_subproblem(samples):
    buckets = defaultdict(list)
    for sample in samples:
        raw = get_subprob(sample)
        clauses = parse_subproblems(raw)
        for clause in set(clauses):
            buckets[clause].append({
                "exec_match": sample.get("exec_match", False),
                "content": sample
            })
    return buckets

def write_subproblem_files(buckets, output_dir="subproblem_analysis"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary = {}

    for clause, items in buckets.items():
        false_samples = [x["content"] for x in items if not x["exec_match"]]
        true_samples = [x["content"] for x in items if x["exec_match"]]
        total = len(items)
        false_count = len(false_samples)
        false_pct = round(false_count / total * 100, 2)

        output = {
            "subproblem": clause,
            "total_samples": total,
            "exec_match_false_count": false_count,
            "exec_match_false_pct": false_pct,
            "false_samples": false_samples,
            "true_samples": true_samples
        }

        with open(f"{output_dir}/{clause}.json", "w") as f:
            json.dump(output, f, indent=2)

        summary[clause] = {
            "total": total,
            "false": false_count,
            "false_pct": false_pct
        }

    return dict(sorted(summary.items(), key=lambda x: -x[1]["false_pct"]))

def main():
    print("🔍 Loading results.json...")
    results = load_results("results.json")

    print("📊 Grouping samples by subproblem...")
    buckets = group_by_subproblem(results)

    print(f"📁 Writing {len(buckets)} subproblem .json files...")
    stats = write_subproblem_files(buckets)

    print("\n✅ Summary of subproblem exec_match failure rates:")
    for sub, stat in stats.items():
        print(f" - {sub:10s}: {stat['false_pct']:>5.1f}% failures ({stat['false']}/{stat['total']})")

if __name__ == "__main__":
    main()
