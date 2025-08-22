# Script to collect all subdomains from assets.txt files
param(
    [string]$BasePath = "C:\Users\abhij\bb\"
)

Write-Host "Collecting subdomains from all assets.txt files in $BasePath"

# Get all assets.txt files
$assetsFiles = Get-ChildItem -Path $BasePath -Recurse -Name "assets.txt"
Write-Host "Found $($assetsFiles.Count) assets.txt files"

# Initialize array to store all subdomains
$allSubdomains = @()
$processedFiles = 0

foreach ($file in $assetsFiles) {
    $fullPath = Join-Path $BasePath $file
    $processedFiles++
    
    if ($processedFiles % 50 -eq 0) {
        Write-Host "Processed $processedFiles/$($assetsFiles.Count) files..."
    }
    
    try {
        # Read file content and filter lines that look like subdomains/domains
        $content = Get-Content $fullPath -ErrorAction Stop
        foreach ($line in $content) {
            $line = $line.Trim()
            # Skip empty lines and comments
            if ($line -and -not $line.StartsWith("#") -and -not $line.StartsWith("//")) {
                # Check if line looks like a domain (contains dot and valid characters)
                if ($line -match '^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$') {
                    $allSubdomains += $line
                }
            }
        }
    }
    catch {
        Write-Warning "Could not read file: $fullPath - $($_.Exception.Message)"
    }
}

Write-Host "Total subdomains collected: $($allSubdomains.Count)"

# Remove duplicates and sort
$uniqueSubdomains = $allSubdomains | Sort-Object -Unique
Write-Host "Unique subdomains: $($uniqueSubdomains.Count)"

# Save to file
$outputFile = "C:\Users\abhij\root_analyser\all_subdomains.txt"
$uniqueSubdomains | Out-File -FilePath $outputFile -Encoding UTF8

Write-Host "All unique subdomains saved to: $outputFile"

# Show some statistics
Write-Host "`nStatistics:"
Write-Host "- Total assets.txt files processed: $processedFiles"
Write-Host "- Total subdomains collected: $($allSubdomains.Count)"
Write-Host "- Unique subdomains: $($uniqueSubdomains.Count)"
Write-Host "- Duplicate rate: $([math]::Round((($allSubdomains.Count - $uniqueSubdomains.Count) / $allSubdomains.Count) * 100, 2))%"

# Show top 10 most common domains
Write-Host "`nTop 10 root domains:"
$rootDomains = $uniqueSubdomains | ForEach-Object {
    if ($_ -match '([^.]+\.[^.]+)$') {
        $matches[1]
    }
} | Group-Object | Sort-Object Count -Descending | Select-Object -First 10

$rootDomains | ForEach-Object {
    Write-Host "  $($_.Name): $($_.Count) subdomains"
}
