param(
    [string]$Python = "python",
    # Another streamdio source for this install, e.g. a local checkout
    # ("-e C:\Code\git\StreamDIO\software") or a branch/tag/commit URL.
    [string]$StreamDio = "",
    [switch]$Dev,
    [switch]$NoShortcut
)
$ErrorActionPreference = "Stop"
$rigRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$rigPython = Join-Path $rigRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $rigPython)) {
    & $Python -m venv (Join-Path $rigRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required to create the rig environment" }
}
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
function Install-RigPackages([string[]]$Packages) {
    if ($uvCommand) {
        & $uvCommand.Source pip install --python $rigPython @Packages
    } else {
        & $rigPython -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) { throw "Could not bootstrap pip" }
        & $rigPython -m pip install @Packages
    }
    if ($LASTEXITCODE -ne 0) { throw "Rig package installation failed" }
}
# Use the tested versions for runtime dependencies. Development tools are optional.
$rigLock = Join-Path $rigRoot "requirements.lock"
$excluded = '^(pytest|mypy|mypy_extensions|mypy-extensions|ruff|types-PyYAML|iniconfig|pluggy|Pygments|librt|pathspec)=='
$runtimeLock = Join-Path $rigRoot "data\runtime\runtime-requirements.txt"
New-Item -ItemType Directory -Force -Path (Split-Path $runtimeLock) | Out-Null
if ($Dev) { Copy-Item -LiteralPath $rigLock -Destination $runtimeLock -Force }
else { Get-Content -LiteralPath $rigLock | Where-Object { $_ -notmatch $excluded } | Set-Content -LiteralPath $runtimeLock }
Install-RigPackages @('-r', $runtimeLock)
Install-RigPackages @('--no-deps', '-e', $rigRoot)
# requirements.lock installs the StreamDIO driver (streamdio) from its repository.
if ($StreamDio) { Install-RigPackages (@('--no-deps', '--force-reinstall') + ($StreamDio -split ' ')) }
& $rigPython -m hf_dm.operator_app --check-install
if ($LASTEXITCODE -ne 0) { throw "Installed launcher check failed" }
if (-not $NoShortcut) {
    $desktop = [Environment]::GetFolderPath('Desktop')
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desktop 'evil_rig.lnk'))
    $shortcut.TargetPath = Join-Path $rigRoot '.venv\Scripts\pythonw.exe'
    $shortcut.Arguments = '-m hf_dm.operator_app'
    $shortcut.WorkingDirectory = $rigRoot
    $shortcut.Description = 'evil_rig session launcher, bounded diagnostics and saved logs'
    $shortcut.Save()
}
Write-Output "Installed evil_rig at $rigRoot. Open the desktop shortcut or run .\Start-Rig.cmd."
