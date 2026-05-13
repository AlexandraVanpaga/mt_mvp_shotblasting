#Requires -Version 5.1
<#
.SYNOPSIS
    Stop mt_mvp Docker stack, listeners on :8000 and :7860.

.PARAMETER KillRepoUvicornPython
    Also stop python.exe whose command line contains uvicorn and this repo path.

.EXAMPLE
    .\scripts\stop_mt_mvp_stack.ps1
    .\scripts\stop_mt_mvp_stack.ps1 -KillRepoUvicornPython
#>
[CmdletBinding()]
param(
    [switch] $KillRepoUvicornPython
)

$ErrorActionPreference = 'Continue'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $repoRoot

Write-Host "Repo: $repoRoot"

Write-Host ""
Write-Host "[1] docker compose down..."
$dc = & docker compose down --remove-orphans 2>&1
$dc | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "(docker compose exit $LASTEXITCODE - ok if Docker is off or no stack)"
}

Write-Host ""
Write-Host "[2] LISTEN on ports 8000, 7860..."
foreach ($port in @(8000, 7860)) {
    $listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) {
        Write-Host "  port $port : nothing listening"
        continue
    }
    foreach ($l in $listeners) {
        $procId = $l.OwningProcess
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        $name = if ($p) { $p.ProcessName } else { '?' }
        Write-Host "  port $port : PID $procId ($name) -> Stop-Process"
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
}

if ($KillRepoUvicornPython) {
    Write-Host ""
    Write-Host "[3] python.exe + uvicorn + repo path..."
    $escaped = [regex]::Escape($repoRoot.Path)
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | ForEach-Object {
        $cl = $_.CommandLine
        if (-not $cl) { return }
        if ($cl -match 'uvicorn' -and $cl -match $escaped) {
            Write-Host "  PID $($_.ProcessId): Stop-Process"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
} else {
    Write-Host ""
    Write-Host "[3] skipped (use -KillRepoUvicornPython to kill repo uvicorn python)"
}

Write-Host ""
Write-Host "Done. If translate still OOM, set in .env:"
Write-Host "  MT_MVP_POSTEDIT_USE_QWEN=false"
Write-Host "  or MT_MVP_POSTEDIT_FORCE_CPU=true"
