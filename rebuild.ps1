[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipDependencies
)

$ErrorActionPreference = "Stop"
$Project = $PSScriptRoot
$Spec = Join-Path $Project "Textorcist OCR helper.spec"

Set-Location -LiteralPath $Project

# Prefer the project's virtual environment; otherwise use Python 3.10.
$VenvPython = Join-Path $Project ".venv\Scripts\python.exe"
$PythonCommand = $null
$PythonArgs = @()

if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
    $PythonCommand = $VenvPython
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.10 -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PythonCommand = "py"
        $PythonArgs = @("-3.10")
    }
}

if (-not $PythonCommand) {
    throw "Python 3.10 was not found. Install it or create .venv, then run this script again."
}

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
    Invoke-Python -m pip install pyinstaller
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
