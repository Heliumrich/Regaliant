# In-run A/B of configurations through the control file (DS_CONTROL): each round sets every
# config's `tune` switches, waits for them to settle, then marks a window with two notes.
# scripts\perf\abfps.py reads the windows' flips per second from the run's dsport.log.
#
# Usage: scripts\perf\abset.ps1 -Ctl <control file> -Configs "base=par_spins 512","new=par_spins 4096" [-Rounds 3] [-Secs 15]
# A config is "<name>=<switch> <value>[;<switch> <value>...]". Alternate rounds, compare only
# windows of the same session at the same view (a scene's load drifts over minutes).
param([Parameter(Mandatory)][string]$Ctl, [Parameter(Mandatory)][string[]]$Configs, [int]$Rounds = 3, [int]$Secs = 15)
function c($l) { Add-Content -Path $Ctl -Value $l -Encoding ascii }
for ($r = 0; $r -lt $Rounds; $r++) {
  foreach ($cfg in $Configs) {
    $name, $rest = $cfg -split '=', 2
    foreach ($s in ($rest -split ';')) { c "tune $($s.Trim())" }
    Start-Sleep 4
    c "note ab cfg=$name start"; Start-Sleep $Secs
    c "note ab cfg=$name end"
  }
}
