# vLLM / PagedAttention Serving Benchmark

This project benchmarks vLLM serving performance under different input context lengths and concurrency levels. The goal is to understand how KV cache memory, PagedAttention, and request concurrency affect LLM serving throughput and latency.

## Why This Project

LLM serving scalability is not only about model size. In production-style serving, performance is strongly affected by:

- KV cache memory usage
- batching and scheduling
- context length
- concurrent requests
- GPU memory pressure

vLLM uses PagedAttention to manage KV cache memory more efficiently, which helps improve serving throughput and scalability.

## Environment

- GPU: NVIDIA GeForce RTX 4080, 16GB VRAM
- Driver: 596.21
- CUDA reported by `nvidia-smi`: 13.2
- Runtime: Docker
- Serving engine: vLLM OpenAI-compatible server
- Docker image: `vllm/vllm-openai:latest`
- Model: `Qwen/Qwen3-0.6B`

## Repository Structure

```text
.
├── README.md
├── docs/
│   └── serving_benchmark.md
├── scripts/
│   ├── benchmark_serving.py
│   └── plot_results.py
├── results/
│   └── serving_benchmark.csv
└── figures/
    ├── throughput_vs_concurrency.png
    └── latency_vs_context.png
```

## Setup

### 1. Start vLLM server

```bash
sudo docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --gpu-memory-utilization 0.80
```

### 2. Validate the server

```bash
curl http://localhost:8000/v1/models
```

### 3. Install Python dependencies

Using conda:

```bash
conda create -n vllm-bench python=3.11 -y
conda activate vllm-bench
conda install -c conda-forge aiohttp matplotlib -y
```

## Run Benchmark

```bash
python scripts/benchmark_serving.py \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-0.6B \
  --input-tokens 128,512,1024,2048 \
  --output-tokens 64 \
  --concurrency 1,2,4,8 \
  --num-requests 32 \
  --output results/serving_benchmark.csv
```

## Plot Results

```bash
python scripts/plot_results.py \
  --input results/serving_benchmark.csv \
  --output-dir figures
```

## Benchmark Matrix

| Parameter | Values |
|---|---|
| Input context length | 128, 512, 1024, 2048 approximate tokens |
| Output length | 64 tokens |
| Concurrency | 1, 2, 4, 8 |
| Requests per case | 32 |
| Total cases | 16 |

## Results

### Throughput vs Concurrency

![Throughput vs concurrency](figures/throughput_vs_concurrency.png)

### P95 Latency vs Context Length

![P95 latency vs context length](figures/latency_vs_context.png)

## Key Observations

- Throughput increases as concurrency increases.
- Longer input context reduces throughput.
- Longer input context increases p95 latency, especially at higher concurrency.
- Even with a small 0.6B model, KV cache and request scheduling have visible performance impact.

Example result at concurrency 8:

| Input tokens | Throughput |
|---:|---:|
| 128 | 1769.0 output tok/s |
| 2048 | 888.5 output tok/s |

## Documentation

- Full benchmark report: [`docs/serving_benchmark.md`](docs/serving_benchmark.md)

## Limitations

- Synthetic prompts are approximate token lengths, not tokenizer-exact prompts.
- The benchmark uses one model, one GPU, and one fixed output length.
- Results represent this local RTX 4080 setup and should not be interpreted as universal vLLM performance numbers.

## Future Work

- Use tokenizer-exact prompt generation.
- Add more models and output lengths.
- Test higher concurrency levels until saturation or OOM.
- Compare vLLM with a non-PagedAttention baseline.
- Add time-to-first-token and inter-token latency measurements.
