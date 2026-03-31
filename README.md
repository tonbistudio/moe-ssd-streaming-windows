# SSD Expert Streaming for Mixture of Experts Models on Windows

**Running a 32 GB AI model on 28 GB of memory — on a budget gaming PC.**

Inspired by [Daniel Isaac's (@danpacary)](https://twitter.com/danpacary) work on streaming MoE expert weights from SSD on Apple Silicon, this project recreates and validates the technique on a budget Windows PC — proving that consumer NVMe SSDs can extend your system's effective memory for large language model inference.

## The Result

A **32 GB model** (Qwen3-30B-A3B at Q8_0) running on a PC with only **28 GB of total memory** (12 GB VRAM + 16 GB RAM), generating text at **2.5–4.3 tokens/second** by streaming expert weights from an NVMe SSD in real-time.

## Hardware Used

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 3060 (12 GB VRAM) |
| RAM | 16 GB DDR4 |
| SSD | Sabrent Rocket 4.0 2TB NVMe |
| Effective SSD speed | 1.5 GB/s (bottlenecked by a PCIe 3.0 x2 adapter) |
| PC | Alienware Aurora R10 (Ryzen) |

## How It Works

Mixture of Experts (MoE) models have hundreds of "expert" sub-networks, but only a few activate per token. This means you don't need all the weights in memory at once.

**The three-tier memory hierarchy:**

1. **GPU VRAM (12 GB)** — Holds the shared layers: attention, routing, embeddings. These are needed for every token.
2. **System RAM (16 GB)** — Acts as a page cache for frequently-used expert weights.
3. **NVMe SSD (2 TB)** — Stores the full model file. Expert weights are memory-mapped and loaded on demand by the OS.

When the model's router selects an expert:
- If it's in RAM (page cache hit) → served instantly
- If not (page cache miss) → streamed from SSD at 1.5 GB/s

The OS page cache automatically manages which experts stay in RAM (hot) and which get evicted (cold), acting as an LRU cache for free.

## Results

| Configuration | Model Size | Speed | How It Works |
|--------------|-----------|-------|-------------|
| GPU only | 12 GB max | Can't run 30B models | Impossible |
| Q4 + CPU offload | 18 GB | 9.3 tok/s | Expert weights fit in RAM |
| Q8 + SSD streaming | 32 GB | 2.5–4.3 tok/s | Expert weights streamed from SSD |

The Q8 configuration is the true SSD streaming scenario — the 29 GB of expert weights physically cannot fit in 16 GB of RAM, so the OS must page them in from the SSD during inference.

## Quick Start

### Prerequisites

- Windows 10/11 with an NVIDIA GPU
- [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) (download the `win-cuda` release)
- A GGUF model file (see below)

### 1. Download llama.cpp

Grab the latest release from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases). Download `llama-*-bin-win-cuda-12.4-x64.zip` and `cudart-llama-bin-win-cuda-12.4-x64.zip`, extract both to the same folder.

### 2. Download a model

```bash
pip install huggingface-hub

# Q4 version (18 GB) — fits in RAM, faster
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen3-30B-A3B-GGUF', 'Qwen3-30B-A3B-Q4_K_M.gguf', local_dir='./models')"

# Q8 version (32 GB) — requires SSD streaming on 16 GB RAM systems
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen3-30B-A3B-GGUF', 'Qwen3-30B-A3B-Q8_0.gguf', local_dir='./models')"
```

### 3. Run with expert offloading

```bash
llama-cli.exe ^
  -m models\Qwen3-30B-A3B-Q8_0.gguf ^
  -ngl 99 ^
  -ot ".ffn_.*_exps.=CPU" ^
  -fa on ^
  -c 4096 ^
  -t 8 ^
  --conversation
```

**What the flags mean:**
- `-ngl 99` — Send all layers to GPU
- `-ot ".ffn_.*_exps.=CPU"` — Override: keep expert FFN weights on CPU (this is the key flag)
- `-fa on` — Enable Flash Attention (reduces VRAM usage)
- `-c 4096` — Context window size
- `-t 8` — CPU threads for expert computation

The `-ot` flag uses a regex pattern to match expert tensor names in the model file and routes them to CPU instead of GPU. Combined with mmap (memory-mapped file I/O, enabled by default), the OS loads expert weights from the SSD on demand.

### 4. Monitor SSD streaming (optional)

Open a second terminal while the model is running:

```powershell
powershell -ExecutionPolicy Bypass -File monitor\monitor_streaming.ps1
```

This shows real-time disk reads, page fault rates, free RAM, and estimated cache hit rate. You'll see disk I/O spike when cold experts are loaded and drop to zero when hot experts are served from cache.

## What's In This Repo

### `/benchmarks`

Windows NVMe throughput benchmarks written in C, testing different async I/O strategies:

| File | Approach |
|------|----------|
| `ssd_bench.c` | Multi-threaded ReadFile + OVERLAPPED |
| `ssd_bench_v2.c` | I/O Completion Ports (IOCP) with deep queue depth |
| `ssd_bench_v3.c` | Event-based batch submit with WaitForMultipleObjects |
| `hybrid_moe.c` | GPU VRAM + SSD latency simulator (uses CUDA Driver API) |

**To compile** (requires Visual Studio 2022):
```bash
# Open Developer Command Prompt, then:
cl /O2 ssd_bench.c /Fe:ssd_bench.exe
cl /O2 hybrid_moe.c /Fe:hybrid_moe.exe
```

The benchmarks create a 4 GB test file and measure throughput across different thread counts, queue depths, and block sizes. `hybrid_moe.c` additionally measures GPU PCIe copy bandwidth and simulates MoE expert loading latency.

### `/monitor`

`monitor_streaming.ps1` — PowerShell script that monitors disk I/O, page faults, free RAM, and cache hit rate in real-time during MoE inference. Color-coded output: green = cached, yellow = partial streaming, red = heavy SSD streaming.

### `/scripts`

Ready-to-use batch files for running Qwen3-30B-A3B with expert offloading:
- `run_qwen_moe.bat` — Q4_K_M (18 GB, experts in RAM)
- `run_qwen_moe_q8.bat` — Q8_0 (32 GB, SSD streaming)

Edit the paths in these files to match your llama.cpp and model locations.

### `/slides`

Presentation slides (`index.html`) and script (`script.md`) explaining the project. Open `index.html` in a browser, navigate with arrow keys. Press F for fullscreen.

## The SSD Bottleneck Discovery

During benchmarking, we discovered the SSD was running at 1.5 GB/s instead of the expected ~3.5 GB/s. Investigation revealed:

```
MaxLinkSpeed     = 4  (PCIe 4.0 — drive capability)
MaxLinkWidth     = 4  (x4 — drive capability)
CurrentLinkSpeed = 3  (PCIe 3.0 — adapter limit)
CurrentLinkWidth = 2  (x2 — adapter only wires 2 lanes!)
```

A cheap M.2-to-PCIe adapter card was only wiring 2 of 4 PCIe lanes. This halved the available bandwidth. With a proper x4 adapter or native M.2 slot, throughput would roughly double to ~3.5 GB/s, directly improving cold expert load times.

**How to check your own link width** (PowerShell):
```powershell
# Find your NVMe controller's PCI device ID
Get-PnpDevice | Where-Object { $_.FriendlyName -like '*NVM*' } | Format-List FriendlyName,InstanceId

# Check link speed/width (replace the InstanceId with yours)
Get-PnpDeviceProperty -InstanceId 'PCI\VEN_...' |
  Where-Object { $_.KeyName -like '*LinkSpeed*' -or $_.KeyName -like '*LinkWidth*' } |
  Format-List KeyName,Data
```

## Scaling Guide

**Will this work on your hardware?** The key requirements:

1. **Shared model layers must fit in VRAM.** For Qwen3-30B-A3B, that's ~3 GB. Any modern GPU with 6+ GB VRAM works.
2. **Expert weights are streamed via page cache.** More RAM = higher cache hit rate = faster. With 16 GB RAM you get partial caching. With 32 GB, the Q4 version fits entirely in RAM.
3. **SSD speed determines cold expert load time.** Faster NVMe = faster cold loads. Even a SATA SSD would work, just slower.

| Your RAM | Q4 (18 GB) | Q8 (32 GB) |
|----------|-----------|-----------|
| 16 GB | Experts mostly in RAM | SSD streaming required |
| 32 GB | Fully in RAM | Experts mostly in RAM |
| 64 GB | Fully in RAM | Fully in RAM |

## Acknowledgments

- **Daniel Isaac ([@danpacary](https://twitter.com/danpacary))** — Original MoE expert streaming technique and benchmarks on Apple Silicon
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — The inference engine that makes this possible with the `-ot` tensor override flag
- **[Qwen](https://huggingface.co/Qwen)** — The Qwen3-30B-A3B model

## License

MIT
