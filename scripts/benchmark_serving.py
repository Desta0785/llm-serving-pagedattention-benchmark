#!/usr/bin/env python3
"""Simple vLLM OpenAI-compatible serving benchmark.

This script sends concurrent completion requests to a running vLLM server and
writes latency / throughput metrics to a CSV file.
"""

import argparse
import asyncio
import csv
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp


DEFAULT_INPUT_TOKENS = [128, 512, 1024, 2048]
DEFAULT_OUTPUT_TOKENS = [64]
DEFAULT_CONCURRENCY = [1, 2, 4, 8]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def make_prompt(target_tokens: int) -> str:
    """Create an approximate prompt with target_tokens whitespace-separated words.

    This is not tokenizer-exact, but it is stable enough for a local benchmark
    matrix. For tokenizer-exact prompts, use the model tokenizer in a future
    refinement.
    """
    words = [f"token{i % 100}" for i in range(target_tokens)]
    return " ".join(words)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[index]


async def send_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    start = time.perf_counter()
    async with session.post(url, json=payload, timeout=timeout_s) as response:
        text = await response.text()
        latency_s = time.perf_counter() - start

        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {text[:500]}")

        data = await response.json()
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", max_tokens)
        total_tokens = usage.get("total_tokens", 0)

        return {
            "latency_s": latency_s,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


async def run_case(
    base_url: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    num_requests: int,
    timeout_s: int,
) -> dict[str, Any]:
    prompt = make_prompt(input_tokens)
    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        pending = set()
        results: list[dict[str, Any]] = []
        started = 0
        case_start = time.perf_counter()

        while started < num_requests or pending:
            while started < num_requests and len(pending) < concurrency:
                task = asyncio.create_task(
                    send_completion(
                        session=session,
                        base_url=base_url,
                        model=model,
                        prompt=prompt,
                        max_tokens=output_tokens,
                        timeout_s=timeout_s,
                    )
                )
                pending.add(task)
                started += 1

            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                results.append(task.result())

        total_time_s = time.perf_counter() - case_start

    latencies = [item["latency_s"] for item in results]
    generated_tokens = sum(int(item["completion_tokens"]) for item in results)

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "total_time_s": total_time_s,
        "throughput_req_s": num_requests / total_time_s,
        "throughput_tok_s": generated_tokens / total_time_s,
        "p50_latency_s": statistics.median(latencies),
        "p95_latency_s": percentile(latencies, 95),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="results/serving_benchmark.csv")
    parser.add_argument("--input-tokens", default=",".join(map(str, DEFAULT_INPUT_TOKENS)))
    parser.add_argument("--output-tokens", default=",".join(map(str, DEFAULT_OUTPUT_TOKENS)))
    parser.add_argument("--concurrency", default=",".join(map(str, DEFAULT_CONCURRENCY)))
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--timeout-s", type=int, default=300)
    args = parser.parse_args()

    input_tokens_list = parse_int_list(args.input_tokens)
    output_tokens_list = parse_int_list(args.output_tokens)
    concurrency_list = parse_int_list(args.concurrency)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "input_tokens",
        "output_tokens",
        "concurrency",
        "num_requests",
        "total_time_s",
        "throughput_req_s",
        "throughput_tok_s",
        "p50_latency_s",
        "p95_latency_s",
        "min_latency_s",
        "max_latency_s",
    ]

    rows: list[dict[str, Any]] = []
    for input_tokens in input_tokens_list:
        for output_tokens in output_tokens_list:
            for concurrency in concurrency_list:
                print(
                    f"Running input={input_tokens}, output={output_tokens}, "
                    f"concurrency={concurrency}, requests={args.num_requests}",
                    flush=True,
                )
                row = await run_case(
                    base_url=args.base_url,
                    model=args.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    concurrency=concurrency,
                    num_requests=args.num_requests,
                    timeout_s=args.timeout_s,
                )
                rows.append(row)
                print(
                    f"  throughput={row['throughput_tok_s']:.2f} tok/s, "
                    f"p50={row['p50_latency_s']:.3f}s, "
                    f"p95={row['p95_latency_s']:.3f}s",
                    flush=True,
                )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
