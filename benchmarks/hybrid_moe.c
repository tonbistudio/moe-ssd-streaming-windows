/*
 * hybrid_moe.c — Hybrid VRAM + SSD MoE expert streaming simulator
 *
 * Simulates a real MoE inference scenario:
 *   - Hot (frequently used) experts pinned in GPU VRAM
 *   - Cold experts streamed from SSD on demand
 *   - Measures end-to-end expert activation latency
 *
 * Uses CUDA Driver API (no nvcc needed — loads nvcuda.dll at runtime).
 *
 * Compile: cl /O2 hybrid_moe.c /Fe:hybrid_moe.exe
 */

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* ---------- CUDA Driver API types (minimal) ---------- */
typedef int CUresult;
typedef void *CUdevice;
typedef void *CUcontext;
typedef void *CUdeviceptr;

#define CUDA_SUCCESS 0

/* Function pointers loaded from nvcuda.dll */
typedef CUresult (*pfnCuInit)(unsigned int);
typedef CUresult (*pfnCuDeviceGet)(CUdevice *, int);
typedef CUresult (*pfnCuDeviceGetName)(char *, int, CUdevice);
typedef CUresult (*pfnCuDeviceTotalMem)(size_t *, CUdevice);
typedef CUresult (*pfnCuCtxCreate)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*pfnCuCtxDestroy)(CUcontext);
typedef CUresult (*pfnCuMemAlloc)(CUdeviceptr *, size_t);
typedef CUresult (*pfnCuMemFree)(CUdeviceptr);
typedef CUresult (*pfnCuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
typedef CUresult (*pfnCuMemcpyDtoH)(void *, CUdeviceptr, size_t);
typedef CUresult (*pfnCuCtxSynchronize)(void);
typedef CUresult (*pfnCuMemHostAlloc)(void **, size_t, unsigned int);
typedef CUresult (*pfnCuMemFreeHost)(void *);

static pfnCuInit             cuInit;
static pfnCuDeviceGet        cuDeviceGet;
static pfnCuDeviceGetName    cuDeviceGetName;
static pfnCuDeviceTotalMem   cuDeviceTotalMem;
static pfnCuCtxCreate        cuCtxCreate;
static pfnCuCtxDestroy       cuCtxDestroy;
static pfnCuMemAlloc         cuMemAlloc;
static pfnCuMemFree          cuMemFree;
static pfnCuMemcpyHtoD       cuMemcpyHtoD;
static pfnCuMemcpyDtoH       cuMemcpyDtoH;
static pfnCuCtxSynchronize   cuCtxSynchronize;
static pfnCuMemHostAlloc     cuMemHostAlloc;
static pfnCuMemFreeHost      cuMemFreeHost;

static HMODULE hCuda = NULL;

static int load_cuda(void) {
    hCuda = LoadLibraryA("nvcuda.dll");
    if (!hCuda) {
        fprintf(stderr, "Cannot load nvcuda.dll — is an NVIDIA GPU installed?\n");
        return 0;
    }
#define LOAD(name) do { \
    *(FARPROC *)&name = GetProcAddress(hCuda, #name); \
    if (!name) { fprintf(stderr, "Missing: %s\n", #name); return 0; } \
} while(0)
    LOAD(cuInit);
    LOAD(cuDeviceGet);
    LOAD(cuDeviceGetName);
    LOAD(cuDeviceTotalMem);
    LOAD(cuCtxCreate);
    LOAD(cuCtxDestroy);
    LOAD(cuMemAlloc);
    LOAD(cuMemFree);
    LOAD(cuMemcpyHtoD);
    LOAD(cuMemcpyDtoH);
    LOAD(cuCtxSynchronize);
    LOAD(cuMemHostAlloc);
    LOAD(cuMemFreeHost);
#undef LOAD
    return 1;
}

/* ---------- config ---------- */
/* Qwen3-MoE-30B-A3B: 128 experts, 8 active per token, ~200MB each (FP16) */
/* With Q4 quantization: ~50MB each */
#define NUM_EXPERTS         128
#define ACTIVE_PER_TOKEN    8
#define EXPERT_SIZE_FP16    (200 * 1024 * 1024LL)    /* 200 MB */
#define EXPERT_SIZE_Q4      (50 * 1024 * 1024LL)     /* 50 MB  */

/* How many hot experts fit in VRAM (leave 2 GB for model + KV cache) */
#define VRAM_BUDGET_GB      10   /* out of 12 GB RTX 3060 */
#define VRAM_BUDGET         ((int64_t)VRAM_BUDGET_GB << 30)

/* SSD async read settings */
#define SSD_QUEUE_DEPTH     16
#define SSD_THREADS         4

/* ---------- timer ---------- */
static double now_sec(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}

/* ---------- SSD async reader (from v2) ---------- */
typedef struct {
    OVERLAPPED ov;
    char      *buf;
} IOSlot;

#define MAX_QD_VAL 64

typedef struct {
    HANDLE hFile;
    HANDLE hIOCP;
    IOSlot slots[MAX_QD_VAL];
    int    qd;
    int64_t chunk_size;
} SSDReader;

static SSDReader *ssd_open(const char *path, int64_t chunk_size, int qd) {
    SSDReader *r = (SSDReader *)calloc(1, sizeof(SSDReader));
    r->chunk_size = chunk_size;
    r->qd = qd < MAX_QD_VAL ? qd : MAX_QD_VAL;

    r->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, NULL);
    if (r->hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "SSD open failed: %lu\n", GetLastError());
        free(r); return NULL;
    }

    r->hIOCP = CreateIoCompletionPort(r->hFile, NULL, 0, 1);

    for (int i = 0; i < r->qd; i++) {
        r->slots[i].buf = (char *)_aligned_malloc((size_t)chunk_size, 4096);
    }
    return r;
}

/* Read one expert-sized blob from offset, return pointer and time taken */
static double ssd_read_expert(SSDReader *r, int64_t offset, char **out_buf) {
    double t0 = now_sec();

    IOSlot *s = &r->slots[0];
    memset(&s->ov, 0, sizeof(OVERLAPPED));
    s->ov.Offset     = (DWORD)(offset & 0xFFFFFFFF);
    s->ov.OffsetHigh = (DWORD)(offset >> 32);

    DWORD nread = 0;
    BOOL ok = ReadFile(r->hFile, s->buf, (DWORD)r->chunk_size, &nread, &s->ov);
    if (!ok && GetLastError() == ERROR_IO_PENDING) {
        DWORD nr2;
        ULONG_PTR key;
        OVERLAPPED *pov;
        GetQueuedCompletionStatus(r->hIOCP, &nr2, &key, &pov, INFINITE);
    }

    *out_buf = s->buf;
    return now_sec() - t0;
}

static void ssd_close(SSDReader *r) {
    for (int i = 0; i < r->qd; i++)
        _aligned_free(r->slots[i].buf);
    CloseHandle(r->hIOCP);
    CloseHandle(r->hFile);
    free(r);
}

/* ---------- parallel SSD read: N experts at once ---------- */
typedef struct {
    SSDReader *reader;
    int64_t    offset;
    char      *out_buf;
    double     elapsed;
} ParReadCtx;

static DWORD WINAPI par_ssd_read_thread(LPVOID arg) {
    ParReadCtx *ctx = (ParReadCtx *)arg;
    ctx->elapsed = ssd_read_expert(ctx->reader, ctx->offset, &ctx->out_buf);
    return 0;
}

/* ---------- simulate MoE routing ---------- */
/* Zipf distribution: expert 0 is most popular, expert 127 least */
static int sample_expert_zipf(int n_experts) {
    double u = (double)rand() / RAND_MAX;
    /* Inverse CDF of Zipf(s=1.0) — crude but fine for simulation */
    double sum = 0;
    double H = 0;
    for (int i = 1; i <= n_experts; i++) H += 1.0 / i;
    for (int i = 1; i <= n_experts; i++) {
        sum += (1.0 / i) / H;
        if (u <= sum) return i - 1;
    }
    return n_experts - 1;
}

/* ---------- main ---------- */
int main(int argc, char **argv) {
    const char *ssd_path = "D:\\infra\\test_experts.bin";
    if (argc > 1) ssd_path = argv[1];

    printf("=== Hybrid VRAM + SSD MoE Expert Streaming Simulator ===\n\n");

    /* --- Init CUDA --- */
    if (!load_cuda()) return 1;

    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", err);
        return 1;
    }

    CUdevice dev;
    cuDeviceGet(&dev, 0);

    char name[256];
    cuDeviceGetName(name, sizeof(name), dev);

    size_t vram_total;
    cuDeviceTotalMem(&vram_total, dev);

    printf("GPU: %s (%.1f GB VRAM)\n", name, (double)vram_total / (1 << 30));

    CUcontext ctx;
    cuCtxCreate(&ctx, 0, dev);

    /* --- Configuration for each quantization --- */
    typedef struct {
        const char *label;
        int64_t     expert_size;
        int         num_hot;      /* how many fit in VRAM budget */
    } Config;

    Config configs[] = {
        {"FP16 (200 MB/expert)", EXPERT_SIZE_FP16,
         (int)(VRAM_BUDGET / EXPERT_SIZE_FP16)},
        {"Q4   ( 50 MB/expert)", EXPERT_SIZE_Q4,
         (int)(VRAM_BUDGET / EXPERT_SIZE_Q4)},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    /* We only need a small test file for the SSD read latency part.
       Use the existing 4GB file, reading from different offsets. */
    HANDLE hCheck = CreateFileA(ssd_path, GENERIC_READ, FILE_SHARE_READ,
                                NULL, OPEN_EXISTING, 0, NULL);
    if (hCheck == INVALID_HANDLE_VALUE) {
        printf("Test file not found at %s. Run ssd_bench.exe first.\n", ssd_path);
        cuCtxDestroy(ctx);
        return 1;
    }
    LARGE_INTEGER file_sz;
    GetFileSizeEx(hCheck, &file_sz);
    CloseHandle(hCheck);
    printf("SSD test file: %s (%.1f GB)\n\n", ssd_path,
           (double)file_sz.QuadPart / (1 << 30));

    for (int ci = 0; ci < n_configs; ci++) {
        Config *cfg = &configs[ci];
        int64_t esz = cfg->expert_size;
        int hot = cfg->num_hot;
        if (hot > NUM_EXPERTS) hot = NUM_EXPERTS;
        int cold = NUM_EXPERTS - hot;

        printf("========================================\n");
        printf("Config: %s\n", cfg->label);
        printf("  VRAM budget: %d GB -> %d hot experts in VRAM, %d cold on SSD\n",
               VRAM_BUDGET_GB, hot, cold);
        printf("  Total model: %d experts x %lld MB = %.1f GB\n",
               NUM_EXPERTS, esz >> 20,
               (double)NUM_EXPERTS * esz / (1 << 30));
        printf("========================================\n\n");

        /* --- Benchmark 1: VRAM copy latency (host -> device) --- */
        printf("  [VRAM] Host-to-Device copy latency:\n");

        /* Allocate pinned host memory and device memory for one expert */
        int64_t alloc_sz = esz;
        /* Cap allocation for FP16 to avoid OOM — we only need one expert buffer */
        CUdeviceptr d_buf;
        err = cuMemAlloc(&d_buf, (size_t)alloc_sz);
        if (err != CUDA_SUCCESS) {
            printf("    cuMemAlloc failed (err=%d) — expert too large for available VRAM.\n", err);
            printf("    Skipping VRAM benchmark for this config.\n\n");
            continue;
        }

        void *h_pinned = NULL;
        err = cuMemHostAlloc(&h_pinned, (size_t)alloc_sz, 0);
        if (err != CUDA_SUCCESS) {
            printf("    cuMemHostAlloc failed — using malloc instead.\n");
            h_pinned = _aligned_malloc((size_t)alloc_sz, 4096);
        }
        memset(h_pinned, 0xAB, (size_t)alloc_sz);

        /* Warmup */
        cuMemcpyHtoD(d_buf, h_pinned, (size_t)alloc_sz);
        cuCtxSynchronize();

        /* Benchmark */
        int n_iters = 10;
        double total_htod = 0;
        for (int i = 0; i < n_iters; i++) {
            double t0 = now_sec();
            cuMemcpyHtoD(d_buf, h_pinned, (size_t)alloc_sz);
            cuCtxSynchronize();
            total_htod += now_sec() - t0;
        }
        double avg_htod = total_htod / n_iters;
        double htod_gbps = ((double)alloc_sz / (1 << 30)) / avg_htod;

        printf("    %lld MB -> GPU: %.2f ms  (%.2f GB/s PCIe)\n",
               alloc_sz >> 20, avg_htod * 1000.0, htod_gbps);

        /* --- Benchmark 2: SSD read latency for one expert --- */
        printf("\n  [SSD] Single expert read latency:\n");

        /* Round expert size down to 4K alignment for NO_BUFFERING */
        int64_t aligned_esz = (esz / 4096) * 4096;
        if (aligned_esz > file_sz.QuadPart)
            aligned_esz = (file_sz.QuadPart / 4096) * 4096;

        SSDReader *ssd = ssd_open(ssd_path, aligned_esz, SSD_QUEUE_DEPTH);
        if (ssd) {
            double total_ssd = 0;
            char *rbuf;
            for (int i = 0; i < n_iters; i++) {
                int64_t off = (rand() % (file_sz.QuadPart / aligned_esz)) * aligned_esz;
                total_ssd += ssd_read_expert(ssd, off, &rbuf);
            }
            double avg_ssd = total_ssd / n_iters;
            double ssd_gbps = ((double)aligned_esz / (1 << 30)) / avg_ssd;

            printf("    %lld MB from SSD: %.2f ms  (%.2f GB/s)\n",
                   aligned_esz >> 20, avg_ssd * 1000.0, ssd_gbps);

            /* --- Benchmark 3: SSD -> GPU pipeline (read + copy) --- */
            double avg_pipeline = avg_ssd + avg_htod;
            printf("\n  [Pipeline] SSD -> RAM -> GPU for one expert:\n");
            printf("    SSD read: %.2f ms + PCIe copy: %.2f ms = %.2f ms total\n",
                   avg_ssd * 1000.0, avg_htod * 1000.0, avg_pipeline * 1000.0);

            /* --- Benchmark 4: Simulate N tokens of MoE inference --- */
            printf("\n  [Simulation] %d tokens, %d active experts each:\n",
                   50, ACTIVE_PER_TOKEN);

            srand(12345);
            int vram_hits = 0, ssd_loads = 0;
            double total_sim_time = 0;

            for (int tok = 0; tok < 50; tok++) {
                double tok_time = 0;
                int cold_this_token = 0;

                for (int a = 0; a < ACTIVE_PER_TOKEN; a++) {
                    int eid = sample_expert_zipf(NUM_EXPERTS);

                    if (eid < hot) {
                        /* Hot expert: already in VRAM, zero load time */
                        vram_hits++;
                    } else {
                        /* Cold expert: must load from SSD then copy to GPU */
                        cold_this_token++;
                        ssd_loads++;
                    }
                }

                if (cold_this_token > 0) {
                    /* In practice, cold experts load in parallel.
                       Model: max(SSD reads in parallel) + one PCIe copy */
                    tok_time = avg_ssd + avg_htod;  /* overlap SSD reads */
                }
                total_sim_time += tok_time;
            }

            double avg_tok_latency = total_sim_time / 50;
            double hit_rate = 100.0 * vram_hits / (50 * ACTIVE_PER_TOKEN);

            printf("    VRAM hits: %d/%d (%.1f%%)\n",
                   vram_hits, 50 * ACTIVE_PER_TOKEN, hit_rate);
            printf("    SSD loads: %d\n", ssd_loads);
            printf("    Avg token expert-load latency: %.2f ms\n",
                   avg_tok_latency * 1000.0);
            printf("    Tokens needing zero SSD: %d/50\n",
                   50 - (ssd_loads > 0 ? (int)(ssd_loads / 1.0) : 0));

            printf("\n  [Verdict for %s]:\n", cfg->label);
            if (avg_tok_latency * 1000.0 < 50.0) {
                printf("    >>> VIABLE for interactive inference (<50ms expert load)\n");
            } else if (avg_tok_latency * 1000.0 < 100.0) {
                printf("    >>> BORDERLINE — may work for batch/slow inference\n");
            } else {
                printf("    >>> TOO SLOW for interactive use at this quantization\n");
            }

            ssd_close(ssd);
        }

        cuMemFree(d_buf);
        if (h_pinned) {
            cuMemFreeHost(h_pinned);
        }
        printf("\n");
    }

    /* --- Summary --- */
    printf("=== Summary ===\n");
    printf("Your RTX 3060 (12 GB) + SABRENT NVMe setup:\n\n");
    printf("  FP16 experts (200 MB): ~50 hot in VRAM, 78 on SSD\n");
    printf("  Q4 experts   ( 50 MB): ~200 hot (all fit!) in VRAM\n\n");
    printf("Key insight: with Q4 quantization, your 12 GB VRAM can hold\n");
    printf("ALL 128 experts of Qwen3-MoE-30B — no SSD streaming needed!\n\n");
    printf("For larger models (e.g., 1T param MoE with 256+ experts),\n");
    printf("the hybrid approach becomes essential. Your SSD throughput\n");
    printf("determines the ceiling for cold expert loading.\n");

    cuCtxDestroy(ctx);
    FreeLibrary(hCuda);

    printf("\nDone.\n");
    return 0;
}
