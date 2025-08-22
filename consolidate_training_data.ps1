# Consolidate Training Data Script
param(
    [string]$ChunksDir = "C:\Users\abhij\root_analyser\chunks",
    [string]$OutputFile = "C:\Users\abhij\root_analyser\consolidated_training_data.txt"
)

Write-Host "=== Consolidating Training Data ===" -ForegroundColor Green
Write-Host "Chunks Directory: $ChunksDir"
Write-Host "Output File: $OutputFile"

$startTime = Get-Date

# Get all chunk files
$chunkFiles = Get-ChildItem -Path $ChunksDir -Name "chunk_*.txt" | Sort-Object
Write-Host "Found $($chunkFiles.Count) chunk files to consolidate" -ForegroundColor Cyan

# Initialize tracking
$allSubdomains = @{}  # Use hashtable for deduplication
$totalProcessed = 0
$duplicatesFound = 0

foreach ($chunkFile in $chunkFiles) {
    $fullPath = Join-Path $ChunksDir $chunkFile
    Write-Host "Processing: $chunkFile" -ForegroundColor Yellow
    
    try {
        $content = Get-Content $fullPath -Encoding UTF8
        
        foreach ($line in $content) {
            $subdomain = $line.Trim()
            if ($subdomain) {
                $totalProcessed++
                
                if ($allSubdomains.ContainsKey($subdomain)) {
                    $duplicatesFound++
                } else {
                    $allSubdomains[$subdomain] = $true
                }
            }
        }
    }
    catch {
        Write-Warning "Error processing $chunkFile : $($_.Exception.Message)"
    }
}

# Sort and save unique subdomains
$uniqueSubdomains = $allSubdomains.Keys | Sort-Object
Write-Host "Writing consolidated data..." -ForegroundColor Yellow

# Use .NET for efficient file writing
[System.IO.File]::WriteAllLines($OutputFile, $uniqueSubdomains, [System.Text.Encoding]::UTF8)

$endTime = Get-Date
$totalTime = $endTime - $startTime

# Final statistics
Write-Host "`n=== CONSOLIDATION COMPLETED ===" -ForegroundColor Green
Write-Host "Chunk files processed: $($chunkFiles.Count)" -ForegroundColor Cyan
Write-Host "Total subdomains processed: $totalProcessed" -ForegroundColor Cyan
Write-Host "Duplicates found: $duplicatesFound" -ForegroundColor Cyan
Write-Host "Unique subdomains: $($uniqueSubdomains.Count)" -ForegroundColor Cyan
Write-Host "Deduplication rate: $([math]::Round(($duplicatesFound / $totalProcessed) * 100, 2))%" -ForegroundColor Cyan
Write-Host "Processing time: $($totalTime.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "Output saved to: $OutputFile" -ForegroundColor Green

# Create metadata file
$metadata = @{
    consolidation_date = Get-Date
    total_subdomains_processed = $totalProcessed
    unique_subdomains = $uniqueSubdomains.Count
    duplicates_removed = $duplicatesFound
    chunks_processed = $chunkFiles.Count
    processing_time_seconds = $totalTime.TotalSeconds
    output_file = $OutputFile
}

$metadataFile = $OutputFile.Replace('.txt', '_metadata.json')
$metadata | ConvertTo-Json -Depth 10 | Out-File $metadataFile -Encoding UTF8

Write-Host "Metadata saved to: $metadataFile" -ForegroundColor Green
