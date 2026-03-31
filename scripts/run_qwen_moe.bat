@echo off
echo === Qwen3-30B-A3B MoE with Expert Offloading ===
echo.
echo GPU: Shared layers (attention, embeddings, routing) in VRAM
echo CPU: Expert FFN weights in system RAM
echo.

D:\infra\llama.cpp\bin\llama-cli.exe ^
  -m D:\infra\models\Qwen3-30B-A3B-Q4_K_M.gguf ^
  -ngl 99 ^
  -ot ".ffn_.*_exps.=CPU" ^
  -fa on ^
  -c 4096 ^
  -t 8 ^
  --no-warmup ^
  -p "You are a helpful assistant." ^
  --conversation

pause
