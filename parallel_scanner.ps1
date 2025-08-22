# Parallel Subdomain Scanner
param(
    [string]$ChunksDir = "C:\Users\abhij\root_analyser\chunks",
    [string]$ResultsDir = "C:\Users\abhij\root_analyser\scan_results",
    [string]$ScanTool = "subdomain_ai_enhanced.py",
    [int]$MaxParallelJobs = 3
)

Write-Host "=== Parallel Subdomain Scanner Started ===" -ForegroundColor Green
Write-Host "Chunks Directory: $ChunksDir"
Write-Host "Results Directory: $ResultsDir"
Write-Host "Scan Tool: $ScanTool"
Write-Host "Max Parallel Jobs: $MaxParallelJobs"

# Check if scan tool exists
$toolPath = Join-Path (Get-Location) $ScanTool
if (-not (Test-Path $toolPath)) {
    Write-Error "Scan tool not found: $toolPath"
    exit 1
}

# Initialize job tracking
$jobs = @()
$completedChunks = 0
$failedChunks = 0
$startTime = Get-Date

# Create status file
$statusFile = Join-Path $ResultsDir "scanning_status.txt"
"Scanning started: $(Get-Date)" | Out-File $statusFile
"Tool: $ScanTool" | Add-Content $statusFile
"Max parallel jobs: $MaxParallelJobs" | Add-Content $statusFile

function Wait-ForAvailableSlot {
    while ($jobs.Count -ge $MaxParallelJobs) {
        # Check for completed jobs
        $completedJobs = @()
        foreach ($job in $jobs) {
            if ($job.State -eq "Completed" -or $job.State -eq "Failed") {
                $completedJobs += $job
            }
        }
        
        # Process completed jobs
        foreach ($completedJob in $completedJobs) {
            $chunkName = $completedJob.Name
            $result = Receive-Job $completedJob -ErrorAction SilentlyContinue
            
            if ($completedJob.State -eq "Completed") {
                Write-Host "✓ Completed: $chunkName" -ForegroundColor Green
                "Completed: $chunkName at $(Get-Date)" | Add-Content $statusFile
                $script:completedChunks++
            } else {
                Write-Host "✗ Failed: $chunkName" -ForegroundColor Red
                "Failed: $chunkName at $(Get-Date)" | Add-Content $statusFile
                $script:failedChunks++
            }
            
            Remove-Job $completedJob
            $jobs = $jobs | Where-Object { $_.Id -ne $completedJob.Id }
        }
        
        if ($jobs.Count -ge $MaxParallelJobs) {
            Start-Sleep -Seconds 2
        }
    }
}

function Start-ChunkScan {
    param([string]$ChunkFile)
    
    $chunkName = [System.IO.Path]::GetFileNameWithoutExtension($ChunkFile)
    $outputFile = Join-Path $ResultsDir "$chunkName`_results.json"
    
    Write-Host "Starting scan: $chunkName" -ForegroundColor Yellow
    
    # Create script block for the scan
    $scriptBlock = {
        param($ToolPath, $ChunkFile, $OutputFile)
        
        try {
            # Run the subdomain analysis tool
            $result = & python $ToolPath --analyze $ChunkFile 2>&1
            
            # Save results
            $results = @{
                chunk_file = $ChunkFile
                output_file = $OutputFile
                scan_time = Get-Date
                result = $result
                success = $true
            }
            
            $results | ConvertTo-Json -Depth 10 | Out-File $OutputFile -Encoding UTF8
            return "SUCCESS: $ChunkFile"
        }
        catch {
            $errorResults = @{
                chunk_file = $ChunkFile
                output_file = $OutputFile
                scan_time = Get-Date
                error = $_.Exception.Message
                success = $false
            }
            
            $errorResults | ConvertTo-Json -Depth 10 | Out-File $OutputFile -Encoding UTF8
            return "ERROR: $($_.Exception.Message)"
        }
    }
    
    # Start the job
    $job = Start-Job -ScriptBlock $scriptBlock -ArgumentList $toolPath, $ChunkFile, $outputFile -Name $chunkName
    $jobs += $job
}

# Monitor for new chunks and process them
Write-Host "`nMonitoring for chunks..." -ForegroundColor Cyan

$processedChunks = @{}

while ($true) {
    # Get all available chunk files
    $chunkFiles = Get-ChildItem -Path $ChunksDir -Name "chunk_*.txt" | Sort-Object
    
    # Process new chunks
    foreach ($chunkFile in $chunkFiles) {
        $fullChunkPath = Join-Path $ChunksDir $chunkFile
        
        if (-not $processedChunks.ContainsKey($chunkFile)) {
            $processedChunks[$chunkFile] = $true
            
            # Wait for an available slot
            Wait-ForAvailableSlot
            
            # Start scanning this chunk
            Start-ChunkScan -ChunkFile $fullChunkPath
        }
    }
    
    # Check if chunking is completed (look for completion marker or status file)
    $chunkingStatus = Join-Path $ChunksDir "chunking_status.txt"
    $chunkingCompleted = $false
    
    if (Test-Path $chunkingStatus) {
        $statusContent = Get-Content $chunkingStatus -Raw
        if ($statusContent -match "CHUNKING COMPLETED") {
            $chunkingCompleted = $true
        }
    }
    
    # If chunking is done and we've processed all chunks, break
    if ($chunkingCompleted -and $chunkFiles.Count -eq $processedChunks.Count) {
        Write-Host "`nChunking completed. Processing remaining jobs..." -ForegroundColor Yellow
        break
    }
    
    # Show progress
    $totalChunks = $chunkFiles.Count
    $runningJobs = $jobs.Count
    Write-Host "Progress: $totalChunks chunks found | $runningJobs active scans | $completedChunks completed | $failedChunks failed" -ForegroundColor Magenta
    
    Start-Sleep -Seconds 5
}

# Wait for all remaining jobs to complete
Write-Host "`nWaiting for all scans to complete..." -ForegroundColor Yellow
while ($jobs.Count -gt 0) {
    Wait-ForAvailableSlot
    Start-Sleep -Seconds 2
}

# Final statistics
$endTime = Get-Date
$totalTime = $endTime - $startTime

Write-Host "`n=== SCANNING COMPLETED ===" -ForegroundColor Green
Write-Host "Total chunks processed: $($completedChunks + $failedChunks)" -ForegroundColor Cyan
Write-Host "Successful scans: $completedChunks" -ForegroundColor Green
Write-Host "Failed scans: $failedChunks" -ForegroundColor Red
Write-Host "Success rate: $([math]::Round(($completedChunks / ($completedChunks + $failedChunks)) * 100, 2))%" -ForegroundColor Cyan
Write-Host "Total scanning time: $($totalTime.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan

# Save final summary
$summary = @"
=== SCANNING SUMMARY ===
Completed: $(Get-Date)
Total Time: $($totalTime.ToString('hh\:mm\:ss'))
Tool Used: $ScanTool
Total Chunks: $($completedChunks + $failedChunks)
Successful: $completedChunks
Failed: $failedChunks
Success Rate: $([math]::Round(($completedChunks / ($completedChunks + $failedChunks)) * 100, 2))%
"@

$summary | Add-Content $statusFile

Write-Host "`nAll results saved in: $ResultsDir" -ForegroundColor Green
