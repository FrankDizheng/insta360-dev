"""
Content moderation evaluation: run Qwen3.5-27B on labeled TW dataset.
Compares model predictions against human-annotated labels.

Usage:
  python eval_moderation.py --csv data.csv --prompt prompt.md [--limit N] [--workers N]
"""

import argparse
import asyncio
import csv
import json
import re
import time
from pathlib import Path

import httpx

_API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen3.5-27b-opus"
MAX_TOKENS = 1024
TEMPERATURE = 0.1


def set_api_url(url: str):
    global _API_URL
    _API_URL = url


def load_prompt_template(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if "{{input_text}}" not in text:
        raise ValueError("Prompt template must contain {{input_text}} placeholder")
    return text


def load_csv(path: str, limit: int = 0) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            if len(row) < 4:
                continue
            content = row[2].strip()
            try:
                label = int(row[3].strip())
            except ValueError:
                continue
            if not content:
                continue
            rows.append({
                "index": i,
                "region": row[0],
                "hit_word": row[1],
                "content": content,
                "ground_truth": label,  # 0=pass, 1=reject
                "flash_result": row[6] if len(row) > 6 else "",
                "flash_duration": row[7] if len(row) > 7 else "",
            })
    return rows


def parse_model_output(text: str) -> dict:
    """Extract label and reason from model output."""
    text = text.strip()
    # Try to find JSON in the output
    json_matches = re.findall(r'\{[^{}]*"label"\s*:\s*"[^"]*"[^{}]*\}', text)
    if json_matches:
        last_json = json_matches[-1]
        try:
            result = json.loads(last_json)
            label_str = result.get("label", "").strip().lower()
            predicted = 1 if label_str == "reject" else 0
            return {"predicted": predicted, "reason": result.get("reason", ""), "raw": text}
        except json.JSONDecodeError:
            pass

    lower = text.lower()
    if '"label": "reject"' in lower or '"label":"reject"' in lower:
        return {"predicted": 1, "reason": "", "raw": text}
    elif '"label": "pass"' in lower or '"label":"pass"' in lower:
        return {"predicted": 0, "reason": "", "raw": text}

    return {"predicted": -1, "reason": "PARSE_ERROR", "raw": text}


async def evaluate_single(
    client: httpx.AsyncClient,
    prompt_template: str,
    row: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    content = row["content"]
    prompt = prompt_template.replace("{{input_text}}", content)

    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }

    async with semaphore:
        t0 = time.time()
        try:
            resp = await client.post(_API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            output_text = data["choices"][0]["message"]["content"]
            duration = time.time() - t0
            result = parse_model_output(output_text)
            result["duration"] = round(duration, 2)
            result["index"] = row["index"]
            result["content"] = content
            result["ground_truth"] = row["ground_truth"]
            result["hit_word"] = row["hit_word"]
            usage = data.get("usage", {})
            result["prompt_tokens"] = usage.get("prompt_tokens", 0)
            result["completion_tokens"] = usage.get("completion_tokens", 0)
            return result
        except Exception as e:
            return {
                "index": row["index"],
                "content": content,
                "ground_truth": row["ground_truth"],
                "hit_word": row["hit_word"],
                "predicted": -1,
                "reason": f"ERROR: {e}",
                "raw": "",
                "duration": round(time.time() - t0, 2),
            }


async def run_eval(prompt_template: str, rows: list[dict], workers: int):
    semaphore = asyncio.Semaphore(workers)
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = [evaluate_single(client, prompt_template, r, semaphore) for r in rows]
        results = []
        done = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            done += 1
            gt = "reject" if result["ground_truth"] == 1 else "pass"
            pd = "reject" if result["predicted"] == 1 else ("pass" if result["predicted"] == 0 else "ERROR")
            match = "OK" if result["ground_truth"] == result["predicted"] else "MISS"
            if done % 20 == 0 or done == total:
                print(f"[{done}/{total}] {match} gt={gt} pred={pd} dur={result.get('duration', '?')}s | {result['content'][:40]}")
    return results


def compute_metrics(results: list[dict]):
    valid = [r for r in results if r["predicted"] != -1]
    errors = [r for r in results if r["predicted"] == -1]

    tp = sum(1 for r in valid if r["ground_truth"] == 1 and r["predicted"] == 1)
    tn = sum(1 for r in valid if r["ground_truth"] == 0 and r["predicted"] == 0)
    fp = sum(1 for r in valid if r["ground_truth"] == 0 and r["predicted"] == 1)
    fn = sum(1 for r in valid if r["ground_truth"] == 1 and r["predicted"] == 0)

    total = len(valid)
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    actual_pos = tp + fn
    actual_neg = tn + fp

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples:  {len(results)}")
    print(f"Valid results:  {total}")
    print(f"Parse errors:   {len(errors)}")
    print(f"")
    print(f"Ground truth:   {actual_pos} reject / {actual_neg} pass")
    print(f"Predicted:      {tp+fp} reject / {tn+fn} pass")
    print(f"")
    print(f"  TP (correct reject): {tp}")
    print(f"  TN (correct pass):   {tn}")
    print(f"  FP (false reject):   {fp}  (pass -> reject)")
    print(f"  FN (false pass):     {fn}  (reject -> pass)")
    print(f"")
    print(f"  Accuracy:   {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Precision:  {precision:.4f}  (of predicted rejects, how many correct)")
    print(f"  Recall:     {recall:.4f}  (of actual rejects, how many caught)")
    print(f"  F1 Score:   {f1:.4f}")
    print("=" * 60)

    durations = [r["duration"] for r in valid if "duration" in r]
    if durations:
        avg_dur = sum(durations) / len(durations)
        total_dur = sum(durations)
        print(f"  Avg latency:   {avg_dur:.2f}s per sample")
        print(f"  Total time:    {total_dur:.1f}s ({total_dur/60:.1f}min)")
        comp_tokens = [r.get("completion_tokens", 0) for r in valid]
        if any(comp_tokens):
            avg_tokens = sum(comp_tokens) / len(comp_tokens)
            print(f"  Avg output tokens: {avg_tokens:.0f}")
    print("=" * 60)

    # Per hit_word breakdown
    hit_words = {}
    for r in valid:
        hw = r.get("hit_word", "unknown")
        if hw not in hit_words:
            hit_words[hw] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        if r["ground_truth"] == 1 and r["predicted"] == 1:
            hit_words[hw]["tp"] += 1
        elif r["ground_truth"] == 0 and r["predicted"] == 0:
            hit_words[hw]["tn"] += 1
        elif r["ground_truth"] == 0 and r["predicted"] == 1:
            hit_words[hw]["fp"] += 1
        elif r["ground_truth"] == 1 and r["predicted"] == 0:
            hit_words[hw]["fn"] += 1

    print("\nPer hit_word breakdown:")
    print(f"{'hit_word':<16} {'total':>5} {'acc':>6} {'prec':>6} {'recall':>6} {'f1':>6} | {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}")
    print("-" * 85)
    for hw in sorted(hit_words.keys()):
        m = hit_words[hw]
        t = m["tp"] + m["tn"] + m["fp"] + m["fn"]
        acc = (m["tp"] + m["tn"]) / t if t else 0
        p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0
        r = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else 0
        f = 2 * p * r / (p + r) if (p + r) else 0
        print(f"{hw:<16} {t:>5} {acc:>6.1%} {p:>6.1%} {r:>6.1%} {f:>6.1%} | {m['tp']:>4} {m['tn']:>4} {m['fp']:>4} {m['fn']:>4}")

    # Show mismatched cases
    mismatches = [r for r in valid if r["ground_truth"] != r["predicted"]]
    if mismatches:
        print(f"\nMismatched cases ({len(mismatches)} total, showing first 30):")
        print("-" * 85)
        for r in mismatches[:30]:
            gt = "reject" if r["ground_truth"] == 1 else "pass"
            pd = "reject" if r["predicted"] == 1 else "pass"
            print(f"  [{r['hit_word']:<10}] GT={gt:<6} PRED={pd:<6} | {r['content'][:60]}")

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "errors": len(errors),
    }


def save_results(results: list[dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "hit_word", "content", "ground_truth", "predicted", "reason", "duration", "prompt_tokens", "completion_tokens"])
        for r in sorted(results, key=lambda x: x["index"]):
            gt_str = "reject" if r["ground_truth"] == 1 else "pass"
            pd_str = "reject" if r["predicted"] == 1 else ("pass" if r["predicted"] == 0 else "ERROR")
            writer.writerow([
                r["index"], r.get("hit_word", ""), r["content"],
                gt_str, pd_str, r.get("reason", ""),
                r.get("duration", ""), r.get("prompt_tokens", ""), r.get("completion_tokens", ""),
            ])
    print(f"\nDetailed results saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate content moderation with Qwen3.5-27B")
    parser.add_argument("--csv", required=True, help="Path to labeled CSV")
    parser.add_argument("--prompt", required=True, help="Path to prompt template (v2.md)")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to evaluate (0=all)")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent requests")
    parser.add_argument("--output", default="eval_results.csv", help="Output CSV path")
    parser.add_argument("--api-url", default=_API_URL, help="vLLM API URL")
    args = parser.parse_args()

    set_api_url(args.api_url)

    print("Loading prompt template...")
    prompt_template = load_prompt_template(args.prompt)

    print("Loading test data...")
    rows = load_csv(args.csv, args.limit)
    print(f"Loaded {len(rows)} samples")

    pos = sum(1 for r in rows if r["ground_truth"] == 1)
    neg = sum(1 for r in rows if r["ground_truth"] == 0)
    print(f"Distribution: {pos} reject / {neg} pass")
    print(f"Concurrent workers: {args.workers}")
    print(f"Starting evaluation...\n")

    t0 = time.time()
    results = await run_eval(prompt_template, rows, args.workers)
    wall_time = time.time() - t0

    metrics = compute_metrics(results)
    print(f"\n  Wall clock time: {wall_time:.1f}s ({wall_time/60:.1f}min)")
    print(f"  Throughput: {len(results)/wall_time:.1f} samples/sec")

    save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
