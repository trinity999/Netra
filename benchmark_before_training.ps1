# Pre-Training Performance Benchmark
Write-Host "=== Pre-Training Performance Benchmark ===" -ForegroundColor Green

$testSampleFile = "C:\Users\abhij\root_analyser\test_sample_1000.txt"

# Create a test sample from the first chunk
if (Test-Path "C:\Users\abhij\root_analyser\chunks\chunk_001.txt") {
    $testSample = Get-Content "C:\Users\abhij\root_analyser\chunks\chunk_001.txt" | Select-Object -First 1000
    $testSample | Out-File $testSampleFile -Encoding UTF8
    Write-Host "Created test sample with 1000 subdomains" -ForegroundColor Cyan
} else {
    Write-Error "No chunk files found for testing"
    exit 1
}

Write-Host "`nRunning benchmark on current model..." -ForegroundColor Yellow
$startTime = Get-Date

try {
    # Run benchmark
    $benchmarkResult = & python "C:\Users\abhij\root_analyser\subdomain_ai_enhanced.py" --benchmark $testSampleFile 2>&1
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Host "`n=== PRE-TRAINING BENCHMARK RESULTS ===" -ForegroundColor Green
    Write-Host $benchmarkResult
    Write-Host "`nBenchmark Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
    
    # Save results
    $preTrainingResults = @{
        timestamp = Get-Date
        benchmark_duration_seconds = $duration.TotalSeconds
        test_sample_size = 1000
        benchmark_output = $benchmarkResult
    }
    
    $preTrainingResults | ConvertTo-Json -Depth 10 | Out-File "C:\Users\abhij\root_analyser\pre_training_benchmark.json" -Encoding UTF8
    
    Write-Host "`n✅ Pre-training benchmark completed and saved!" -ForegroundColor Green
}
catch {
    Write-Error "Benchmark failed: $($_.Exception.Message)"
    exit 1
}
