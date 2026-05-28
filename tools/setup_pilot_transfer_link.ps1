<#
.SYNOPSIS
Configure a Windows adapter for the TritonPilot to TritonAnalysis file transfer link.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\tools\setup_pilot_transfer_link.ps1 -ListAdapters

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\tools\setup_pilot_transfer_link.ps1 -AdapterAlias "Ethernet 3" -DryRun

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\tools\setup_pilot_transfer_link.ps1 -AdapterAlias "Ethernet 3" -Sync
#>

[CmdletBinding()]
param(
    [string]$AdapterAlias,
    [string]$AnalysisAddress = "10.77.0.2",
    [ValidateRange(1, 32)]
    [int]$PrefixLength = 24,
    [string]$PilotAddress = "10.77.0.1",
    [ValidateRange(1, 65535)]
    [int]$PilotPort = 8765,
    [string]$Output = "",
    [ValidateRange(1, 300)]
    [int]$TimeoutSeconds = 10,
    [switch]$DryRun,
    [switch]$Sync,
    [switch]$ListAdapters
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$WorkspaceRoot = Join-Path $RepoRoot "Workspace"
if (-not $Output) {
    $Output = Join-Path $WorkspaceRoot "incoming\pilot"
}
foreach ($folder in @(
    "incoming\pilot",
    "sources",
    "results",
    "results\realityscan",
    "results\crab_detection",
    "results\coral_garden",
    "results\color_correction",
    "reports",
    "exports",
    "calibrations",
    "scratch"
)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $WorkspaceRoot $folder) | Out-Null
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
}

if ($ListAdapters) {
    Get-NetAdapter |
        Sort-Object ifIndex |
        Format-Table -Auto Name, InterfaceDescription, Status, LinkSpeed, MacAddress, ifIndex
    exit 0
}

if (-not $AdapterAlias) {
    Write-Error "Pass -AdapterAlias with the Windows adapter name. Use -ListAdapters to find it."
}

if (-not (Test-IsAdministrator)) {
    Write-Error "Run this script from an Administrator PowerShell session."
}

$adapter = Get-NetAdapter -Name $AdapterAlias -ErrorAction Stop
Write-Host "Configuring '$($adapter.Name)' ($($adapter.InterfaceDescription))..."

Set-NetIPInterface -InterfaceAlias $AdapterAlias -AddressFamily IPv4 -Dhcp Disabled

$existingIPv4 = Get-NetIPAddress -InterfaceAlias $AdapterAlias -AddressFamily IPv4 -ErrorAction SilentlyContinue
foreach ($address in $existingIPv4) {
    if ($address.IPAddress -ne $AnalysisAddress -or $address.PrefixLength -ne $PrefixLength) {
        Remove-NetIPAddress -InterfaceAlias $AdapterAlias -IPAddress $address.IPAddress -Confirm:$false
    }
}

$targetAddress = Get-NetIPAddress `
    -InterfaceAlias $AdapterAlias `
    -AddressFamily IPv4 `
    -IPAddress $AnalysisAddress `
    -ErrorAction SilentlyContinue

if (-not $targetAddress) {
    New-NetIPAddress -InterfaceAlias $AdapterAlias -IPAddress $AnalysisAddress -PrefixLength $PrefixLength | Out-Null
}

Set-DnsClientServerAddress -InterfaceAlias $AdapterAlias -ResetServerAddresses

try {
    Set-NetConnectionProfile -InterfaceAlias $AdapterAlias -NetworkCategory Private
} catch {
    Write-Warning "Could not set the network category to Private yet: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "Adapter configuration:"
Get-NetIPConfiguration -InterfaceAlias $AdapterAlias |
    Format-List InterfaceAlias, InterfaceIndex, IPv4Address, IPv4DefaultGateway, DNSServer

Write-Host "IPv4 DNS servers:"
Get-DnsClientServerAddress -InterfaceAlias $AdapterAlias -AddressFamily IPv4 |
    Select-Object InterfaceAlias, ServerAddresses |
    Format-List

$baseUrl = "http://${PilotAddress}:${PilotPort}"
Write-Host "Testing $baseUrl over '$AdapterAlias'..."
$test = Test-NetConnection $PilotAddress -Port $PilotPort -InformationLevel Detailed
$test |
    Select-Object ComputerName, RemoteAddress, RemotePort, InterfaceAlias, SourceAddress, TcpTestSucceeded |
    Format-List

if (-not $test.TcpTestSucceeded) {
    Write-Warning "TCP connection failed. Confirm the pilot adapter is $PilotAddress/$PrefixLength and the TritonPilot Analysis Share is running."
    exit 1
}

if ($DryRun -or $Sync) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    $pythonArgs = @()
    if (-not $pythonCommand) {
        $pythonCommand = Get-Command py -ErrorAction SilentlyContinue
        $pythonArgs = @("-3")
    }
    if (-not $pythonCommand) {
        Write-Error "Python was not found on PATH. Install Python or run the sync command from an environment that has Python."
    }

    $syncArgs = @("-m", "tools.pilot_transfer_sync", $baseUrl, "--output", $Output, "--timeout", $TimeoutSeconds)
    if ($DryRun) {
        $syncArgs += "--dry-run"
    }

    Push-Location $RepoRoot
    try {
        & $pythonCommand.Source @pythonArgs @syncArgs
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "TritonPilot link is configured. Sync destination: $Output"
