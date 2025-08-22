# Efficient Subdomain Chunking Script
param(
    [string]$BasePath = "C:\Users\abhij\bb\",
    [int]$ChunkSize = 5000,
    [string]$ChunksDir = "C:\Users\abhij\root_analyser\chunks"
)

Write-Host "=== Subdomain Chunking Process Started ===" -ForegroundColor Green
Write-Host "Base Path: $BasePath"
Write-Host "Chunk Size: $ChunkSize"
Write-Host "Chunks Directory: $ChunksDir"

# Initialize counters
$totalSubdomains = 0
$uniqueSubdomains = 0
$chunkNumber = 1
$currentChunk = @()
$processedFiles = 0
$startTime = Get-Date

# Create a hashtable to track unique subdomains globally
$globalUnique = @{}

# Get all assets.txt files
Write-Host "`nDiscovering assets.txt files..." -ForegroundColor Yellow
$assetsFiles = Get-ChildItem -Path $BasePath -Recurse -Name "assets.txt"
Write-Host "Found $($assetsFiles.Count) assets.txt files to process" -ForegroundColor Cyan

# Create status file
$statusFile = Join-Path $ChunksDir "chunking_status.txt"
"Started: $(Get-Date)" | Out-File $statusFile

foreach ($file in $assetsFiles) {
    $fullPath = Join-Path $BasePath $file
    $processedFiles++
    
    # Progress update every 25 files
    if ($processedFiles % 25 -eq 0) {
        $elapsed = (Get-Date) - $startTime
        $rate = $processedFiles / $elapsed.TotalSeconds
        $eta = [TimeSpan]::FromSeconds(($assetsFiles.Count - $processedFiles) / $rate)
        Write-Host "Progress: $processedFiles/$($assetsFiles.Count) files | Rate: $([math]::Round($rate, 1))/sec | ETA: $($eta.ToString('hh\:mm\:ss'))" -ForegroundColor Magenta
    }
    
    try {
        # Read file efficiently
        $content = [System.IO.File]::ReadAllLines($fullPath)
        
        foreach ($line in $content) {
            $line = $line.Trim()
            
            # Skip empty lines and comments
            if (-not $line -or $line.StartsWith("#") -or $line.StartsWith("//")) {
                continue
            }
            
            # Basic domain validation (more permissive for subdomains)
            if ($line -match '^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$') {
                $totalSubdomains++
                
                # Check if subdomain is unique globally
                if (-not $globalUnique.ContainsKey($line)) {
                    $globalUnique[$line] = $true
                    $uniqueSubdomains++
                    $currentChunk += $line
                    
                    # Check if chunk is full
                    if ($currentChunk.Count -eq $ChunkSize) {
                        $chunkFile = Join-Path $ChunksDir "chunk_$($chunkNumber.ToString('D3')).txt"
                        $currentChunk | Out-File $chunkFile -Encoding UTF8
                        Write-Host "Created chunk $chunkNumber with $($currentChunk.Count) subdomains" -ForegroundColor Green
                        
                        # Update status
                        "Chunk $chunkNumber completed: $(Get-Date) - $($currentChunk.Count) subdomains" | Add-Content $statusFile
                        
                        # Reset for next chunk
                        $currentChunk = @()
                        $chunkNumber++
                    }
                }
            }
        }
    }
    catch {
        Write-Warning "Error processing $fullPath : $($_.Exception.Message)"
    }
}

# Save remaining subdomains in final chunk
if ($currentChunk.Count -gt 0) {
    $chunkFile = Join-Path $ChunksDir "chunk_$($chunkNumber.ToString('D3')).txt"
    $currentChunk | Out-File $chunkFile -Encoding UTF8
    Write-Host "Created final chunk $chunkNumber with $($currentChunk.Count) subdomains" -ForegroundColor Green
    "Final chunk $chunkNumber completed: $(Get-Date) - $($currentChunk.Count) subdomains" | Add-Content $statusFile
}

# Final statistics
$endTime = Get-Date
$totalTime = $endTime - $startTime

Write-Host "`n=== CHUNKING COMPLETED ===" -ForegroundColor Green
Write-Host "Total assets.txt files processed: $processedFiles" -ForegroundColor Cyan
Write-Host "Total subdomains found: $totalSubdomains" -ForegroundColor Cyan
Write-Host "Unique subdomains: $uniqueSubdomains" -ForegroundColor Cyan
Write-Host "Duplicate rate: $([math]::Round((($totalSubdomains - $uniqueSubdomains) / $totalSubdomains) * 100, 2))%" -ForegroundColor Cyan
Write-Host "Total chunks created: $chunkNumber" -ForegroundColor Cyan
Write-Host "Processing time: $($totalTime.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan

# Save final status
$summary = @"
=== CHUNKING SUMMARY ===
Completed: $(Get-Date)
Total Time: $($totalTime.ToString('hh\:mm\:ss'))
Files Processed: $processedFiles
Total Subdomains: $totalSubdomains
Unique Subdomains: $uniqueSubdomains
Chunks Created: $chunkNumber
Duplicate Rate: $([math]::Round((($totalSubdomains - $uniqueSubdomains) / $totalSubdomains) * 100, 2))%
"@

$summary | Add-Content $statusFile
Write-Host "`nChunking completed! Ready for parallel scanning." -ForegroundColor Green
