# Multi-Epoch Parallel Training System
param(
    [string]$TrainingDataFile = "C:\Users\abhij\root_analyser\consolidated_training_data.txt",
    [int]$Epochs = 5,
    [int]$BatchSize = 10000,
    [string]$SessionName = "massive_bb_training_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int]$ParallelWorkers = 2,
    [string]$TrainingDir = "C:\Users\abhij\root_analyser\training_sessions"
)

Write-Host "=== Multi-Epoch Training System ===" -ForegroundColor Green
Write-Host "Training Data: $TrainingDataFile"
Write-Host "Epochs: $Epochs"
Write-Host "Batch Size: $BatchSize"
Write-Host "Session Name: $SessionName"
Write-Host "Parallel Workers: $ParallelWorkers"

# Create training directory structure
New-Item -ItemType Directory -Path $TrainingDir -Force | Out-Null
New-Item -ItemType Directory -Path "$TrainingDir\$SessionName" -Force | Out-Null
New-Item -ItemType Directory -Path "$TrainingDir\$SessionName\batches" -Force | Out-Null
New-Item -ItemType Directory -Path "$TrainingDir\$SessionName\logs" -Force | Out-Null
New-Item -ItemType Directory -Path "$TrainingDir\$SessionName\models" -Force | Out-Null

$sessionDir = "$TrainingDir\$SessionName"
$batchesDir = "$sessionDir\batches"
$logsDir = "$sessionDir\logs"
$modelsDir = "$sessionDir\models"

Write-Host "Session directory: $sessionDir" -ForegroundColor Cyan

# Check if training data exists
if (-not (Test-Path $TrainingDataFile)) {
    Write-Error "Training data file not found: $TrainingDataFile"
    Write-Host "Running consolidation first..." -ForegroundColor Yellow
    & "C:\Users\abhij\root_analyser\consolidate_training_data.ps1"
    
    if (-not (Test-Path $TrainingDataFile)) {
        Write-Error "Failed to create training data file"
        exit 1
    }
}

# Get training data size
$totalSubdomains = (Get-Content $TrainingDataFile | Measure-Object).Count
Write-Host "Total subdomains in training data: $totalSubdomains" -ForegroundColor Cyan

# Calculate batches
$totalBatches = [math]::Ceiling($totalSubdomains / $BatchSize)
Write-Host "Total batches per epoch: $totalBatches" -ForegroundColor Cyan

# Create session metadata
$sessionMetadata = @{
    session_name = $SessionName
    start_time = Get-Date
    epochs = $Epochs
    batch_size = $BatchSize
    total_subdomains = $totalSubdomains
    total_batches = $totalBatches
    parallel_workers = $ParallelWorkers
    training_data_file = $TrainingDataFile
}

$sessionMetadata | ConvertTo-Json -Depth 10 | Out-File "$sessionDir\session_metadata.json" -Encoding UTF8

# Function to create training batches
function Create-TrainingBatches {
    param([int]$EpochNumber)
    
    Write-Host "`n=== Creating Batches for Epoch $EpochNumber ===" -ForegroundColor Yellow
    
    $epochBatchDir = "$batchesDir\epoch_$EpochNumber"
    New-Item -ItemType Directory -Path $epochBatchDir -Force | Out-Null
    
    $batchNumber = 1
    $currentBatch = @()
    $lineNumber = 0
    
    # Read training data line by line
    $reader = [System.IO.File]::OpenText($TrainingDataFile)
    try {
        while (($line = $reader.ReadLine()) -ne $null) {
            $lineNumber++
            $subdomain = $line.Trim()
            
            if ($subdomain) {
                $currentBatch += $subdomain
                
                # Save batch when it reaches batch size
                if ($currentBatch.Count -eq $BatchSize) {
                    $batchFile = "$epochBatchDir\batch_$($batchNumber.ToString('D3')).txt"
                    [System.IO.File]::WriteAllLines($batchFile, $currentBatch, [System.Text.Encoding]::UTF8)
                    
                    Write-Host "Created batch $batchNumber for epoch $EpochNumber ($($currentBatch.Count) subdomains)" -ForegroundColor Green
                    
                    $currentBatch = @()
                    $batchNumber++
                }
            }
        }
        
        # Save remaining subdomains
        if ($currentBatch.Count -gt 0) {
            $batchFile = "$epochBatchDir\batch_$($batchNumber.ToString('D3')).txt"
            [System.IO.File]::WriteAllLines($batchFile, $currentBatch, [System.Text.Encoding]::UTF8)
            Write-Host "Created final batch $batchNumber for epoch $EpochNumber ($($currentBatch.Count) subdomains)" -ForegroundColor Green
        }
    }
    finally {
        $reader.Close()
    }
    
    return $batchNumber
}

# Function to train on a batch
function Start-BatchTraining {
    param([string]$BatchFile, [int]$EpochNumber, [int]$BatchNumber)
    
    $batchName = "Epoch$EpochNumber`_Batch$BatchNumber"
    $logFile = "$logsDir\$batchName`_log.txt"
    
    $scriptBlock = {
        param($BatchFile, $LogFile, $SessionName, $EpochNumber, $BatchNumber)
        
        $startTime = Get-Date
        $output = @()
        
        try {
            # Run training
            $trainingSessionName = "$SessionName`_epoch$EpochNumber`_batch$BatchNumber"
            $result = & python "C:\Users\abhij\root_analyser\subdomain_ai_enhanced.py" --train $BatchFile --session-name $trainingSessionName 2>&1
            
            $endTime = Get-Date
            $duration = $endTime - $startTime
            
            $output += "=== TRAINING BATCH COMPLETED ==="
            $output += "Batch File: $BatchFile"
            $output += "Session: $trainingSessionName"
            $output += "Start Time: $startTime"
            $output += "End Time: $endTime"
            $output += "Duration: $($duration.ToString('hh\:mm\:ss'))"
            $output += "=== TRAINING OUTPUT ==="
            $output += $result
            $output += "=== END OUTPUT ==="
            
            [System.IO.File]::WriteAllLines($LogFile, $output, [System.Text.Encoding]::UTF8)
            
            return @{
                success = $true
                batch_file = $BatchFile
                duration = $duration.TotalSeconds
                log_file = $LogFile
            }
        }
        catch {
            $output += "=== TRAINING BATCH FAILED ==="
            $output += "Error: $($_.Exception.Message)"
            $output += "Batch File: $BatchFile"
            $output += "Time: $(Get-Date)"
            
            [System.IO.File]::WriteAllLines($LogFile, $output, [System.Text.Encoding]::UTF8)
            
            return @{
                success = $false
                error = $_.Exception.Message
                batch_file = $BatchFile
                log_file = $LogFile
            }
        }
    }
    
    $job = Start-Job -Name $batchName -ScriptBlock $scriptBlock -ArgumentList $BatchFile, $logFile, $SessionName, $EpochNumber, $BatchNumber
    return $job
}

# Main training loop
$overallStartTime = Get-Date
$jobs = @()
$completedBatches = 0
$failedBatches = 0

Write-Host "`n=== Starting Multi-Epoch Training ===" -ForegroundColor Green

for ($epoch = 1; $epoch -le $Epochs; $epoch++) {
    Write-Host "`n🚀 EPOCH $epoch/$Epochs STARTED" -ForegroundColor Magenta
    
    # Create batches for this epoch
    $batchesCreated = Create-TrainingBatches -EpochNumber $epoch
    $epochBatchDir = "$batchesDir\epoch_$epoch"
    
    # Get batch files for this epoch
    $epochBatchFiles = Get-ChildItem -Path $epochBatchDir -Name "batch_*.txt" | Sort-Object
    
    Write-Host "Starting training on $($epochBatchFiles.Count) batches for epoch $epoch" -ForegroundColor Cyan
    
    # Process batches with parallelization
    $batchNumber = 1
    foreach ($batchFile in $epochBatchFiles) {
        $fullBatchPath = Join-Path $epochBatchDir $batchFile
        
        # Wait for available worker slot
        while ($jobs.Count -ge $ParallelWorkers) {
            # Check for completed jobs
            $completedJobs = @()
            foreach ($job in $jobs) {
                if ($job.State -eq "Completed" -or $job.State -eq "Failed") {
                    $completedJobs += $job
                }
            }
            
            # Process completed jobs
            foreach ($completedJob in $completedJobs) {
                $jobResult = Receive-Job $completedJob -ErrorAction SilentlyContinue
                
                if ($completedJob.State -eq "Completed") {
                    Write-Host "✅ Completed: $($completedJob.Name)" -ForegroundColor Green
                    $script:completedBatches++
                } else {
                    Write-Host "❌ Failed: $($completedJob.Name)" -ForegroundColor Red
                    $script:failedBatches++
                }
                
                Remove-Job $completedJob
                $jobs = $jobs | Where-Object { $_.Id -ne $completedJob.Id }
            }
            
            if ($jobs.Count -ge $ParallelWorkers) {
                Start-Sleep -Seconds 3
            }
        }
        
        # Start training for this batch
        $job = Start-BatchTraining -BatchFile $fullBatchPath -EpochNumber $epoch -BatchNumber $batchNumber
        $jobs += $job
        
        Write-Host "🔄 Started: Epoch $epoch, Batch $batchNumber" -ForegroundColor Yellow
        $batchNumber++
    }
    
    Write-Host "🎯 EPOCH $epoch BATCH CREATION COMPLETED" -ForegroundColor Magenta
}

# Wait for all jobs to complete
Write-Host "`n⏳ Waiting for all training jobs to complete..." -ForegroundColor Yellow

while ($jobs.Count -gt 0) {
    $completedJobs = @()
    foreach ($job in $jobs) {
        if ($job.State -eq "Completed" -or $job.State -eq "Failed") {
            $completedJobs += $job
        }
    }
    
    foreach ($completedJob in $completedJobs) {
        if ($completedJob.State -eq "Completed") {
            Write-Host "✅ Final: $($completedJob.Name)" -ForegroundColor Green
            $script:completedBatches++
        } else {
            Write-Host "❌ Final: $($completedJob.Name)" -ForegroundColor Red
            $script:failedBatches++
        }
        
        Remove-Job $completedJob
        $jobs = $jobs | Where-Object { $_.Id -ne $completedJob.Id }
    }
    
    if ($jobs.Count -gt 0) {
        Write-Host "⏱️  Waiting for $($jobs.Count) remaining jobs..." -ForegroundColor Magenta
        Start-Sleep -Seconds 5
    }
}

# Final statistics and model backup
$overallEndTime = Get-Date
$overallTime = $overallEndTime - $overallStartTime

Write-Host "`n🎉 === MULTI-EPOCH TRAINING COMPLETED ===" -ForegroundColor Green
Write-Host "📊 Training Statistics:" -ForegroundColor Cyan
Write-Host "   • Total Epochs: $Epochs" -ForegroundColor White
Write-Host "   • Total Subdomains: $totalSubdomains" -ForegroundColor White
Write-Host "   • Batch Size: $BatchSize" -ForegroundColor White
Write-Host "   • Completed Batches: $completedBatches" -ForegroundColor Green
Write-Host "   • Failed Batches: $failedBatches" -ForegroundColor Red
Write-Host "   • Success Rate: $([math]::Round(($completedBatches / ($completedBatches + $failedBatches)) * 100, 2))%" -ForegroundColor Cyan
Write-Host "   • Total Training Time: $($overallTime.ToString('hh\:mm\:ss'))" -ForegroundColor White

# Backup final model
if (Test-Path "subdomain_classifier.joblib") {
    Copy-Item "subdomain_classifier.joblib" "$modelsDir\final_model_$SessionName.joblib"
    Copy-Item "subdomain_vectorizer.joblib" "$modelsDir\final_vectorizer_$SessionName.joblib" -ErrorAction SilentlyContinue
    Copy-Item "label_encoders.joblib" "$modelsDir\final_encoders_$SessionName.joblib" -ErrorAction SilentlyContinue
    Write-Host "🎯 Model backed up to: $modelsDir" -ForegroundColor Green
}

# Create final training summary
$finalSummary = @{
    session_name = $SessionName
    start_time = $overallStartTime
    end_time = $overallEndTime
    total_duration_seconds = $overallTime.TotalSeconds
    epochs_completed = $Epochs
    total_subdomains = $totalSubdomains
    batch_size = $BatchSize
    completed_batches = $completedBatches
    failed_batches = $failedBatches
    success_rate = [math]::Round(($completedBatches / ($completedBatches + $failedBatches)) * 100, 2)
    parallel_workers = $ParallelWorkers
    training_data_file = $TrainingDataFile
}

$finalSummary | ConvertTo-Json -Depth 10 | Out-File "$sessionDir\final_summary.json" -Encoding UTF8

Write-Host "`n✨ Training session completed successfully!" -ForegroundColor Green
Write-Host "📂 All logs and models saved in: $sessionDir" -ForegroundColor Cyan
