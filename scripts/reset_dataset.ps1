<#
Reset dataset directory safely (Windows PowerShell)

What this script does:
- Stops backend if running in docker-compose (optional prompt)
- Makes a timestamped backup copy of the `dataset/` folder to `backups/dataset-YYYYMMDD-HHMMSS`
- Removes feature folders, labels.csv, samples.csv, and processed outputs from the active `dataset/`
- Recreates empty `labels.csv` and `samples.csv` with the correct headers so new registrations will start from class_idx = 1

Usage (from repo root):
.
# run in PowerShell
.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.
./scripts/reset_dataset.ps1

# You will be prompted to confirm.
# NOTE: This permanently removes files from the active dataset directory after backup. Use with caution.
#>

param(
    [switch]$NoPrompt
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition | Split-Path -Parent
$datasetDir = Join-Path $repoRoot 'dataset'
$backupRoot = Join-Path $repoRoot 'backups'

if (-not (Test-Path $datasetDir)) {
    Write-Host "No dataset directory found at $datasetDir" -ForegroundColor Yellow
    exit 0
}

if (-not $NoPrompt) {
    $confirm = Read-Host "This will BACKUP then CLEAR '$datasetDir'. Continue? (yes/no)"
    if ($confirm -ne 'yes') {
        Write-Host "Aborted by user." -ForegroundColor Yellow
        exit 0
    }
}

# Create backup
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$dest = Join-Path $backupRoot "dataset-$ts"
New-Item -ItemType Directory -Path $dest -Force | Out-Null

Write-Host "Backing up $datasetDir -> $dest ..."
# Use robocopy for robust copy (preserve unicode filenames)
robocopy $datasetDir $dest /MIR /NDL /NFL /NJH /NJS | Out-Null

# Remove contents of dataset folder (features, labels.csv, samples.csv, processed)
# Build array explicitly and ensure Join-Path is called per item (wrap in parentheses)
$itemsToRemove = @(
    (Join-Path $datasetDir 'features'),
    (Join-Path $datasetDir 'labels.csv'),
    (Join-Path $datasetDir 'samples.csv'),
    (Join-Path $datasetDir 'processed')
)

foreach ($it in $itemsToRemove) {
    if (Test-Path $it) {
        if (Test-Path $it -PathType Container) {
            Write-Host "Removing folder: $it"
            Remove-Item -LiteralPath $it -Recurse -Force
        }
        else {
            Write-Host "Removing file: $it"
            Remove-Item -LiteralPath $it -Force
        }
    }
}

# Recreate minimal dataset structure and CSV headers
New-Item -ItemType Directory -Path (Join-Path $datasetDir 'features') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $datasetDir 'processed') -Force | Out-Null

$labelsHeader = 'class_idx,label_original,slug,folder_name,created_at,updated_at,deleted_at,active,dataset_version,notes'
$samplesHeader = 'sample_id,class_idx,label,folder_name,file,user,session_id,frames,duration,source,status,storage_path,created_at,updated_at'

"$labelsHeader" | Out-File -FilePath (Join-Path $datasetDir 'labels.csv') -Encoding utf8
"$samplesHeader" | Out-File -FilePath (Join-Path $datasetDir 'samples.csv') -Encoding utf8

Write-Host "Dataset reset complete. Backup stored at: $dest" -ForegroundColor Green
Write-Host "New empty labels.csv and samples.csv created. Class numbering will start at 1 on next registration." -ForegroundColor Green

Write-Host "Reminder: If you run backend in docker-compose with mounted dataset, restart the backend or let it pick up changes (hot reload may detect)." -ForegroundColor Yellow

exit 0
