param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$venvDir = Join-Path $repoRoot ".venv"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$requirementsPath = Join-Path $repoRoot "insta360-dev\python\requirements.txt"

if (-not (Test-Path $requirementsPath)) {
    throw "requirements.txt not found: $requirementsPath"
}

if ($Recreate -and (Test-Path $venvDir)) {
    Write-Host "[local-env] Removing existing venv: $venvDir"
    Remove-Item -Recurse -Force $venvDir
}

if (-not (Test-Path $pythonExe)) {
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if (-not $systemPython) {
        throw "System python was not found in PATH. Install Python first."
    }

    Write-Host "[local-env] Creating venv at $venvDir"
    & $systemPython.Source -m venv $venvDir
}

Write-Host "[local-env] Installing requirements from $requirementsPath"
& $pythonExe -m pip install -r $requirementsPath

Write-Host "[local-env] Ready:"
Write-Host "  python: $pythonExe"
