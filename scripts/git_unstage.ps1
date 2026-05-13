<#
.SYNOPSIS
  Снять файлы с git stage (индекс). Рабочая копия не меняется.

.EXAMPLE
  .\scripts\git_unstage.ps1
  .\scripts\git_unstage.ps1 README_RUS.md
  .\scripts\git_unstage.ps1 results_final/
#>
param(
    [Parameter(Position = 0)]
    [string[]]$Paths = @(".")
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
    Write-Error "Not a git repo: $RepoRoot"
}
foreach ($p in $Paths) {
    git restore --staged -- $p
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host "OK: unstaged ->" ($Paths -join ", ")
