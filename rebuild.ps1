[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipDependencies
)

$ErrorActionPreference = "Stop"
$Project = $PSScriptRoot
$Spec = Join-Path $Project "Textorcist OCR helper.spec"

Set-Location -LiteralPath $Project

# Build from the project's virtual environment so only runtime dependencies get bundled.
$VenvPython = Join-Path $Project ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python 3.10 was not found. Install it, then run this script again."
    }
    & py -3.10 -c "import sys" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.10 was not found. Install it, then run this script again."
    }
    Write-Host "Creating .venv with Python 3.10..." -ForegroundColor Cyan
    & py -3.10 -m venv (Join-Path $Project ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create .venv."
    }
}

$PythonCommand = $VenvPython
$PythonArgs = @()

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $PythonCommand @PythonArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $Spec -PathType Leaf)) {
    throw "Build specification not found: $Spec"
}

$RunningApp = Get-Process -Name "Textorcist OCR helper" -ErrorAction SilentlyContinue
if ($RunningApp) {
    $RunningApp | ForEach-Object { $_.CloseMainWindow() | Out-Null }
    try {
        Wait-Process -Id $RunningApp.Id -Timeout 5 -ErrorAction Stop
    }
    catch {
        throw "Textorcist OCR helper is still running. Close it before rebuilding."
    }
}

Write-Host "Building Textorcist OCR helper" -ForegroundColor Yellow
Write-Host "Project: $Project"

if (-not $SkipDependencies) {
    Write-Host "Installing application and build dependencies..." -ForegroundColor Cyan
    Invoke-Python -m pip install -r (Join-Path $Project "requirements.txt")
    Invoke-Python -m pip install pyinstaller==6.22.3
}

if (-not $SkipTests) {
    Write-Host "Running automatic-capture and save-import tests..." -ForegroundColor Cyan
    Invoke-Python -m unittest -v test_auto_roi.py test_save_import.py
}

# Invoke the packaged console entry point rather than relying on Scripts being on PATH.
$PyInstaller = @(
    "from PyInstaller.__main__ import run"
    "import sys"
    "sys.exit(run(sys.argv[1:]))"
) -join "`n"

Write-Host "Packaging the Windows executable..." -ForegroundColor Cyan
Invoke-Python -c $PyInstaller --noconfirm --clean $Spec

$Executable = Join-Path $Project "dist\Textorcist OCR helper.exe"
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "The build completed without creating $Executable"
}

$Size = [math]::Round((Get-Item -LiteralPath $Executable).Length / 1MB, 1)
Write-Host "Build complete: $Executable ($Size MB)" -ForegroundColor Green
