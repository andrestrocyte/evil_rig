# Build StreamDIO's firmware and flash it onto this rig's StreamDIO, and only there.
#
# By default it builds the StreamDIO commit the rig's driver (streamdio, in .venv) was
# installed from, so firmware and driver match. The Bpod's Teensy shares the USB bus:
# the StreamDIO alone is rebooted into its bootloader (its Serial port opened at 134
# baud) and teensy_loader_cli then programs the one board waiting there.
param(
    # StreamDIO branch, tag or commit; default: the installed driver's commit
    [string]$Ref = "",
    # The StreamDIO's Serial port (not its config/stream ports); default: found from
    # the running firmware. Pass it when the module runs older firmware (on this rig
    # COM7, see Device Manager: the StreamDIO Teensy's interface MI_00).
    [string]$Port = "",
    [string]$Checkout = "",
    [switch]$BuildOnly
)
$ErrorActionPreference = "Stop"
$rigRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$rigPython = Join-Path $rigRoot ".venv\Scripts\python.exe"
$repoUrl = "https://github.com/MathiasMahn/StreamDIO"
if (-not $Checkout) { $Checkout = Join-Path $rigRoot "data\runtime\StreamDIO" }

if (-not $Ref) {
    $Ref = & $rigPython -c @"
import json
from importlib.metadata import distribution
print(json.loads(distribution('streamdio').read_text('direct_url.json') or '{}').get('vcs_info', {}).get('commit_id', ''))
"@
    if ($LASTEXITCODE -ne 0 -or -not $Ref) {
        throw "The installed streamdio has no recorded commit (a local install?); pass -Ref"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $Checkout ".git"))) {
    git clone $repoUrl $Checkout
    if ($LASTEXITCODE -ne 0) { throw "Could not clone $repoUrl" }
}
git -C $Checkout fetch --tags origin
if ($LASTEXITCODE -ne 0) { throw "Could not fetch $repoUrl" }
$commit = git -C $Checkout rev-parse -q --verify "origin/$Ref^{commit}"
if ($LASTEXITCODE -ne 0) { $commit = git -C $Checkout rev-parse -q --verify "$Ref^{commit}" }
if ($LASTEXITCODE -ne 0) { throw "StreamDIO has no branch, tag or commit '$Ref'" }
git -C $Checkout checkout --quiet --detach $commit
if ($LASTEXITCODE -ne 0) { throw "Could not check out $commit" }

$pio = (Get-Command pio -ErrorAction SilentlyContinue).Source
if (-not $pio) { $pio = Join-Path $env:USERPROFILE ".platformio\penv\Scripts\pio.exe" }
if (-not (Test-Path -LiteralPath $pio)) { throw "PlatformIO (pio) was not found" }
Write-Output "Building StreamDIO firmware at $commit"
& $pio run -d $Checkout -e teensy41
if ($LASTEXITCODE -ne 0) { throw "Firmware build failed" }
$hex = Join-Path $Checkout ".pio\build\teensy41\firmware.hex"
if ($BuildOnly) { Write-Output "Built $hex"; exit 0 }

if (-not $Port) {
    $Port = & $rigPython -c @"
from streamdio import find_streamdio_ports
from hf_dm.hardware.discovery import streamdio_serial_port
print(streamdio_serial_port(*find_streamdio_ports()))
"@
    if ($LASTEXITCODE -ne 0 -or -not $Port) {
        throw "StreamDIO not found (older firmware does not announce itself); pass -Port"
    }
}
$loader = Join-Path $env:USERPROFILE ".platformio\packages\tool-teensy\teensy_loader_cli.exe"
if (-not (Test-Path -LiteralPath $loader)) { throw "teensy_loader_cli was not found: $loader" }

Write-Output "Rebooting the StreamDIO on $Port into its bootloader"
$serial = New-Object System.IO.Ports.SerialPort $Port, 134
$serial.Open()
Start-Sleep -Milliseconds 100
$serial.Close()
# A loader left waiting would program whichever Teensy next enters its bootloader.
$flash = Start-Process -FilePath $loader -ArgumentList @("--mcu=TEENSY41", "-w", "-v", $hex) `
    -NoNewWindow -PassThru
if (-not $flash.WaitForExit(30000)) {
    $flash.Kill()
    throw "No Teensy entered the bootloader within 30 s; nothing was flashed"
}
if ($flash.ExitCode -ne 0) { throw "teensy_loader_cli failed ($($flash.ExitCode))" }
Write-Output "Flashed StreamDIO firmware $commit. Check it: .venv\Scripts\streamdio-config show"
