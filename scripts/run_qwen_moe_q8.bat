@echo off
echo === Qwen3-30B-A3B Q8_0 — TRUE SSD STREAMING ===
echo.
echo Expert weights (29 GB) exceed your 16 GB RAM.
echo The OS will stream experts from SSD via page cache.
echo.
echo TIP: Open another terminal and run:
echo   powershell -ExecutionPolicy Bypass -File D:\infra\monitor_streaming.ps1
echo to watch the SSD streaming in real-time.
echo.
echo Starting model...
echo.

D:\infra\llama.cpp\bin\llama-cli.exe ^
  -m D:\infra\models\Qwen3-30B-A3B-Q8_0.gguf ^
  -ngl 99 ^
  -ot ".ffn_.*_exps.=CPU" ^
  -fa on ^
  -c 4096 ^
  -t 8 ^
  --no-warmup ^
  --conversation

pause
