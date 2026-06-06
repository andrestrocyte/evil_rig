param(
    [string]$Port = "",
    [string]$Fqbn = "teensy:avr:teensy41:usb=serial3",
    [string]$BuildPath = "C:\hf_fwbuild"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$sketchPath = Join-Path $repoRoot "firmware\Module_DIO_streeaming_100us_cycle"

$arduinoCli = (Get-Command arduino-cli -ErrorAction SilentlyContinue).Source
if (-not $arduinoCli) {
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe"
    if (Test-Path -LiteralPath $candidate) {
        $arduinoCli = $candidate
    }
}
if (-not $arduinoCli) {
    throw "arduino-cli was not found on PATH or in the Arduino IDE bundle."
}

New-Item -ItemType Directory -Force -Path $BuildPath | Out-Null

Write-Host "Compiling StreamDIO firmware: $sketchPath"
& $arduinoCli compile --fqbn $Fqbn --build-path $BuildPath $sketchPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $Port) {
    Write-Host ""
    Write-Host "Compile complete. Connected boards:"
    & $arduinoCli board list
    Write-Host ""
    Write-Host "Pass -Port <upload-port> to upload, for example the StreamDIO Teensy usb:* port or COM port."
    exit 0
}

Write-Host "Uploading StreamDIO firmware to $Port"
& $arduinoCli upload --fqbn $Fqbn --port $Port --protocol teensy --input-dir $BuildPath $sketchPath
exit $LASTEXITCODE
