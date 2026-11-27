# Starts the game from target\release with a control file of its own and drives it into play
# through the control file (`press`, `stick`): the title's two cutscenes skipped, Continue,
# offline, then the camera turned half round (the Nexus's hall from the save used on
# 2026-11-27). The timings fit a warm start on a fast PC; check the run's window shot.
#
# Usage: scripts\perf\run-game.ps1 -Ctl <control file> [-Profile]
#   -Profile: DS_PROFILE_GPU=1 with DS_PROFILE_GPU_STAMPS=seg (GPU time by barrier segment; it
#   costs CPU, so compare frame rates only without it). DS_PROGRAM_IDS=1 names programs by code
#   hash in the log either way.
param([Parameter(Mandatory)][string]$Ctl, [switch]$Profile)
Set-Content -Path $Ctl -Value "# perf run" -Encoding ascii
$env:DS_CONTROL = $Ctl
if ($Profile) { $env:DS_PROFILE_GPU = "1"; $env:DS_PROFILE_GPU_STAMPS = "seg"; $env:DS_PROFILE_TOP = "600" } else { $env:DS_PROFILE_GPU = "0" }
$env:DS_PROGRAM_IDS = "1"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$exe = Join-Path $root "target\release\dsport.exe"
$p = Start-Process -FilePath $exe -WorkingDirectory (Split-Path $exe) -PassThru
function c($l) { Add-Content -Path $Ctl -Value $l -Encoding ascii }
Start-Sleep 55
c "press cross 5s"; Start-Sleep 8
c "press cross 300ms"; Start-Sleep 4
c "press cross 300ms"; Start-Sleep 3
c "press down 200ms"; Start-Sleep 2
c "press cross 300ms"; Start-Sleep 18
c "stick r 127 0 1000"; Start-Sleep 4
c "wshot"
"pid $($p.Id)"
