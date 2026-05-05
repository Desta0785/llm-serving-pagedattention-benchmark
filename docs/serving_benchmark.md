# vLLM / PagedAttention Serving Benchmark

## Day 11: Serving Baseline

Goal: run a reproducible vLLM serving baseline before expanding to the full batch/context benchmark matrix.

### Environment

- GPU: NVIDIA GeForce RTX 4080, 16GB VRAM
- Driver: 596.21
- CUDA reported by `nvidia-smi`: 13.2
- Runtime: Docker
- vLLM image: `vllm/vllm-openai:latest`
- Model: `Qwen/Qwen3-0.6B`
- API port: `8000`

### Docker GPU Validation

The host GPU was visible through `nvidia-smi`. Docker GPU access required NVIDIA Container Toolkit.

After configuring the NVIDIA runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

GPU access inside Docker can be validated with:

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### vLLM Serving Command

The first launch failed because the default vLLM GPU memory utilization target was slightly higher than the free VRAM available on startup. The baseline therefore uses `--gpu-memory-utilization 0.80` for a stable local setup.

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

### Validation

Model endpoint:

```bash
curl http://localhost:8000/v1/models
```

Observed result:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-0.6B",
      "object": "model",
      "owned_by": "vllm",
      "root": "Qwen/Qwen3-0.6B",
      "max_model_len": 40960
    }
  ]
}
```

### Notes

- `--gpu-memory-utilization` controls the total GPU memory budget vLLM may use, not just the model weight size.
- The 0.6B model weights are small, but vLLM also reserves memory for KV cache, runtime buffers, CUDA graphs, and serving concurrency.
- Day 12 will expand this baseline into a benchmark matrix over batch size and context length.

### Day 11 Status

Completed: vLLM OpenAI-compatible serving baseline is running and reproducible.
