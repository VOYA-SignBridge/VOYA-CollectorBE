param(
    [string]$Target = "dataset",
    [string]$Url = "https://example.com/path/to/dataset.zip"
)

Write-Output "Downloading dataset to $Target from $Url"
if (-Not (Test-Path $Target)) { New-Item -ItemType Directory -Path $Target | Out-Null }

$out = Join-Path $Target "dataset.zip"
try {
    (New-Object System.Net.WebClient).DownloadFile($Url, $out)
    Expand-Archive -Path $out -DestinationPath $Target -Force
    Remove-Item $out
    Write-Output "Dataset ready in $Target"
} catch {
    Write-Error "Failed to download or extract dataset: $_"
}
