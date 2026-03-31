# monitor_streaming.ps1 — Real-time SSD streaming monitor
# Shows page faults, disk reads, and cache behavior during MoE inference
#
# Run in a separate terminal: powershell -ExecutionPolicy Bypass -File D:\infra\monitor_streaming.ps1

$host.UI.RawUI.WindowTitle = "MoE SSD Streaming Monitor"

Write-Host "=== MoE Expert SSD Streaming Monitor ===" -ForegroundColor Cyan
Write-Host "Watching for llama-cli.exe disk I/O and memory pressure...`n"

# Wait for llama-cli to start
Write-Host "Waiting for llama-cli.exe to start..." -ForegroundColor Yellow
while (-not (Get-Process -Name "llama-cli" -ErrorAction SilentlyContinue)) {
    Start-Sleep -Milliseconds 500
}
Write-Host "llama-cli.exe detected!`n" -ForegroundColor Green

$diskCounter = "\PhysicalDisk(_Total)\Disk Read Bytes/sec"
$faultCounter = "\Memory\Page Faults/sec"
$availCounter = "\Memory\Available MBytes"
$cacheCounter = "\Memory\Cache Bytes"

$prevRead = 0
$totalDiskRead = 0
$samples = 0
$cacheHits = 0
$cacheMisses = 0

Write-Host ("{0,-12} {1,-14} {2,-14} {3,-12} {4,-12} {5,-10}" -f "Time", "Disk Read/s", "Page Faults/s", "Free RAM MB", "Cache MB", "Est Hit%")
Write-Host ("{0,-12} {1,-14} {2,-14} {3,-12} {4,-12} {5,-10}" -f "----", "-----------", "-------------", "----------", "--------", "-------")

while (Get-Process -Name "llama-cli" -ErrorAction SilentlyContinue) {
    try {
        $diskRead = (Get-Counter $diskCounter -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue
        $pageFaults = (Get-Counter $faultCounter -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue
        $availMB = (Get-Counter $availCounter -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue
        $cacheMB = (Get-Counter $cacheCounter -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue / 1MB

        $diskReadMB = $diskRead / 1MB
        $totalDiskRead += $diskReadMB
        $samples++

        # Estimate cache hit rate: if disk reads are low but page faults are high,
        # those faults are being served from cache (soft faults)
        $hitRate = 0
        if ($pageFaults -gt 10) {
            # Rough: each hard fault reads ~4KB from disk
            $hardFaults = $diskRead / 4096
            $softFaults = [math]::Max(0, $pageFaults - $hardFaults)
            if ($pageFaults -gt 0) {
                $hitRate = [math]::Round(($softFaults / $pageFaults) * 100, 1)
            }
        }

        $time = Get-Date -Format "HH:mm:ss"

        $color = "White"
        if ($diskReadMB -gt 100) { $color = "Red" }       # heavy SSD streaming
        elseif ($diskReadMB -gt 10) { $color = "Yellow" }  # moderate
        else { $color = "Green" }                           # mostly cached

        Write-Host ("{0,-12} {1,-14} {2,-14} {3,-12} {4,-12} {5,-10}" -f `
            $time, `
            ("{0:N1} MB/s" -f $diskReadMB), `
            ("{0:N0}" -f $pageFaults), `
            ("{0:N0}" -f $availMB), `
            ("{0:N0}" -f $cacheMB), `
            ("{0:N1}%" -f $hitRate)) -ForegroundColor $color

    } catch {
        # Counter read failed, skip
    }

    Start-Sleep -Seconds 1
}

Write-Host "`n=== Session Summary ===" -ForegroundColor Cyan
Write-Host "Total disk read: $([math]::Round($totalDiskRead / 1024, 2)) GB"
Write-Host "Avg disk read/s: $([math]::Round($totalDiskRead / [math]::Max(1,$samples), 1)) MB/s"
Write-Host "Samples: $samples"
Write-Host "`nColor key: Green = cached (fast), Yellow = partial SSD, Red = heavy SSD streaming"
