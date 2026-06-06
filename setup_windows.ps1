[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$SkipPythonInstall,
    [switch]$IncludeDev
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Find-PackageManager {
    $wingetCmd = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        try {
            $null = & $wingetCmd.Source --version
            return @{
                Name = "winget"
                Command = $wingetCmd.Source
            }
        }
        catch {
        }
    }

    $chocoCmd = Get-Command choco.exe -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        try {
            $null = & $chocoCmd.Source --version
            return @{
                Name = "choco"
                Command = $chocoCmd.Source
            }
        }
        catch {
        }
    }

    return $null
}

function Install-WithWinget {
    param(
        [string]$WingetPath,
        [string[]]$PackageIds
    )

    foreach ($packageId in $PackageIds) {
        try {
            & $WingetPath install --id $packageId --exact --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -eq 0) {
                return $packageId
            }
            Write-Warning "winget install for '$packageId' exited with code $LASTEXITCODE."
        }
        catch {
            Write-Warning "winget install for '$packageId' failed: $_"
        }
    }

    throw "Failed to install package via winget. Tried: $($PackageIds -join ', ')"
}

function Install-WithChocolatey {
    param(
        [string]$ChocoPath,
        [string[]]$PackageNames
    )

    foreach ($packageName in $PackageNames) {
        try {
            & $ChocoPath install $packageName -y
            if ($LASTEXITCODE -eq 0) {
                return $packageName
            }
            Write-Warning "choco install for '$packageName' exited with code $LASTEXITCODE."
        }
        catch {
            Write-Warning "choco install for '$packageName' failed: $_"
        }
    }

    throw "Failed to install package via Chocolatey. Tried: $($PackageNames -join ', ')"
}

function Test-PythonCandidate {
    param(
        [string]$Exe,
        [string[]]$CommandArgs
    )

    try {
        $versionOutput = & $Exe @CommandArgs --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
    }
    catch {
        return $false
    }

    $versionText = ($versionOutput | Out-String).Trim()
    if ($versionText -notmatch "Python\s+(\d+)\.(\d+)") {
        return $false
    }

    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    return ($major -gt 3) -or (($major -eq 3) -and ($minor -ge 10))
}

function Find-PythonCommand {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py -and (Test-PythonCandidate -Exe $py.Source -CommandArgs @("-3"))) {
        return @{
            Exe = $py.Source
            Args = @("-3")
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python -and (Test-PythonCandidate -Exe $python.Source -CommandArgs @())) {
        return @{
            Exe = $python.Source
            Args = @()
        }
    }

    $searchPatterns = @(
        "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe",
        "C:\Program Files\Python*\python.exe",
        "C:\Python*\python.exe"
    )
    $candidates = @()
    foreach ($pattern in $searchPatterns) {
        $candidates += Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    }
    $candidate = $candidates | Sort-Object FullName -Descending | Select-Object -First 1
    if ($candidate -and (Test-PythonCandidate -Exe $candidate.FullName -CommandArgs @())) {
        return @{
            Exe = $candidate.FullName
            Args = @()
        }
    }

    return $null
}

function Ensure-PythonCommand {
    param(
        [hashtable]$PackageManager,
        [switch]$SkipInstall
    )

    $pythonCmd = Find-PythonCommand
    if ($pythonCmd) {
        return $pythonCmd
    }

    if ($SkipInstall) {
        throw "Python 3.10+ was not found. Install it first, or rerun without -SkipPythonInstall."
    }

    if (-not $PackageManager) {
        throw "Python 3.10+ was not found and no supported package manager (winget/choco) is available."
    }

    Write-Step "Installing Python 3"
    if ($PackageManager.Name -eq "winget") {
        $null = Install-WithWinget -WingetPath $PackageManager.Command -PackageIds @("Python.Python.3.11")
    }
    elseif ($PackageManager.Name -eq "choco") {
        $null = Install-WithChocolatey -ChocoPath $PackageManager.Command -PackageNames @("python")
    }
    else {
        throw "Unsupported package manager: $($PackageManager.Name)"
    }

    $pythonCmd = Find-PythonCommand
    if (-not $pythonCmd) {
        throw "Python was installed but was not detected in this shell. Open a new PowerShell session and rerun setup_windows.ps1."
    }

    Write-Host "Python installed successfully." -ForegroundColor Green
    return $pythonCmd
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $repoRoot ".venv"
$requirementsFile = Join-Path $repoRoot "requirements-windows.txt"
$devRequirementsFile = Join-Path $repoRoot "requirements-dev.txt"
$packageManager = Find-PackageManager

Write-Step "Using repository root $repoRoot"

if (-not (Test-Path $requirementsFile)) {
    throw "Missing requirements file: $requirementsFile"
}

$pythonCmd = Ensure-PythonCommand -PackageManager $packageManager -SkipInstall:$SkipPythonInstall
$pythonExe = $pythonCmd.Exe
$pythonArgs = @($pythonCmd.Args)

if ((Test-Path $venvDir) -and $Force) {
    Write-Step "Removing existing virtual environment"
    Remove-Item -Recurse -Force $venvDir
}

if (-not (Test-Path $venvDir)) {
    Write-Step "Creating virtual environment"
    & $pythonExe @pythonArgs -m venv $venvDir
}
else {
    Write-Step "Reusing existing virtual environment"
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found: $venvPython"
}

Write-Step "Upgrading pip tooling"
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Step "Installing Python dependencies"
& $venvPython -m pip install -r $requirementsFile

if ($IncludeDev) {
    if (-not (Test-Path $devRequirementsFile)) {
        throw "Missing development requirements file: $devRequirementsFile"
    }
    Write-Step "Installing development dependencies"
    & $venvPython -m pip install -r $devRequirementsFile
}

Write-Step "Verifying TritonAnalysis imports"
& $venvPython -c "import PyQt6, cv2, matplotlib, numpy, paramiko, scipy; from triton_analysis.gui.ssh_console_window import SshConsolePage, default_analysis_ssh_presets; presets = default_analysis_ssh_presets(); assert presets; print('TritonAnalysis packages verified; SSH console and Paramiko are available')"

Write-Step "Setup complete"
Write-Host "Activate the virtual environment with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then launch the unified app with:" -ForegroundColor Green
Write-Host "  python .\main_triton_analysis.py"
Write-Host ""
Write-Host "The SSH tab uses Paramiko from this virtual environment." -ForegroundColor Green
