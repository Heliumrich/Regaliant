# Builds the player package:
#   dist\DemonsSouls-Port-<version>\     dsport.exe, DemonsSoulsLauncher.exe, README.txt, upscalers\
#   dist\DemonsSouls-Port-<version>.zip  (the same files, no top folder)
# README.txt is the player README (with the port's copyright and license notice), then the
# third-party notices of the two programs (THIRD_PARTY_NOTICES.txt, made by scripts\notices.py
# from what was just built), then the text of the port's license, the GNU GPL version 3.
# The programs are built with the dist profile (LTO, no debug information, stripped), a static C
# runtime, and the paths of the machine that builds them remapped; the package is refused if
# one of those paths, the user profile or the user name is still found in them.
# Only the files listed here go in (an allowlist): nothing of the game, no keys, saves, caches
# or logs, and not NVIDIA's nvngx_dlss.dll (players add their own copy).
#   .\scripts\package.ps1 [-NoBuild] [-Pgo <profile.profdata>] [-NoUpscalers] [-Upscalers <dir>]
# -Pgo: dsport.exe built with that profile (profile-guided optimization from scripts\pgo.ps1;
# the launcher without it).
# The "upscalers" folder: XeSS, NVIDIA Streamline and FSR as their makers ship them, with their
# license texts, from local-data\upscalers (scripts\upscalers.ps1); each DLL must carry a valid
# signature. -NoUpscalers leaves it out.
# Note: do not use ErrorActionPreference=Stop here; cargo writes progress to stderr.
param([switch]$NoBuild, [string]$Pgo, [switch]$NoUpscalers, [string]$Upscalers)
$root = Split-Path -Parent $PSScriptRoot
$target = 'x86_64-pc-windows-msvc'
$ver = (Select-String -Path (Join-Path $root 'Cargo.toml') -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
# The project-local Cargo cache when there is one (scripts\cargo.ps1), else the user's.
$cargoHome = Join-Path $root '.cargo-home'
if (Test-Path $cargoHome) { $env:CARGO_HOME = $cargoHome } else { $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' } }
$cargo = Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe'
if (-not (Test-Path $cargo)) { $cargo = 'cargo' }
if (-not $NoBuild) {
  # (Encoded flags: the paths may hold spaces.)
  $flags = @('-C', 'target-feature=+crt-static', "--remap-path-prefix=$root=ds", "--remap-path-prefix=$cargoHome=cargo", "--remap-path-prefix=$($env:USERPROFILE)=home")
  Push-Location $root
  if ($Pgo) {
    if (-not (Test-Path $Pgo)) { Pop-Location; Write-Host "no profile at $Pgo"; exit 1 }
    $env:CARGO_ENCODED_RUSTFLAGS = ($flags + @("-Cprofile-use=$((Resolve-Path $Pgo).Path)")) -join [char]0x1f
    & $cargo build --profile dist --locked --target $target -p dsport 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    if ($code -eq 0) {
      $env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]0x1f
      & $cargo build --profile dist --locked --target $target -p launcher 2>&1 | ForEach-Object { "$_" }
      $code = $LASTEXITCODE
    }
  } else {
    $env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]0x1f
    & $cargo build --profile dist --locked --target $target -p dsport -p launcher 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
  }
  Pop-Location
  Remove-Item Env:CARGO_ENCODED_RUSTFLAGS
  if ($code -ne 0) { Write-Host "build failed"; exit 1 }
  # The third-party notices of what was just built (THIRD_PARTY_NOTICES.txt).
  & python -I (Join-Path $PSScriptRoot 'notices.py')
  if ($LASTEXITCODE -ne 0) { Write-Host "notices failed"; exit 1 }
}
$bin = Join-Path $root "target\$target\dist"
$name = "DemonsSouls-Port-$ver"
$dist = Join-Path $root 'dist'
$final = Join-Path $dist $name
# (Built in a fresh staging folder, then swapped in: a launcher still running from the previous
# package locks its file, and the zip must not take that one.)
$out = Join-Path $dist ".stage-$name"
if (Test-Path $out) { Remove-Item -Recurse -Force $out -ErrorAction Stop }
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($f in 'dsport.exe', 'DemonsSoulsLauncher.exe') {
  $src = Join-Path $bin $f
  if (-not (Test-Path $src)) { Write-Host "missing $src"; exit 1 }
  Copy-Item $src $out -ErrorAction Stop
  if ((Get-FileHash $src).Hash -ne (Get-FileHash (Join-Path $out $f)).Hash) { Write-Host "$f was not copied whole"; exit 1 }
}
# The upscalers' libraries and their licenses (an allowlist of names). DLSS itself,
# nvngx_dlss.dll, is not one of them: it is under NVIDIA's own license, so players add their own.
$upNames = @(
  'libxess.dll', 'libxess.license.txt', 'libxess.third-party-programs.txt',
  'sl.interposer.dll', 'sl.common.dll', 'sl.dlss.dll', 'streamline.license.txt', 'streamline.3rd-party-licenses.md',
  'amd_fidelityfx_loader_vk.dll', 'amd_fidelityfx_upscaler_vk.dll', 'fidelityfx.license.md'
)
if (-not $NoUpscalers) {
  $upSrc = if ($Upscalers) { $Upscalers } else { Join-Path $root 'local-data\upscalers' }
  $upOut = Join-Path $out 'upscalers'
  New-Item -ItemType Directory -Force $upOut | Out-Null
  foreach ($f in $upNames) {
    $src = Join-Path $upSrc $f
    if (-not (Test-Path $src)) { Write-Host "missing $src (run scripts\upscalers.ps1)"; exit 1 }
    Copy-Item $src $upOut -ErrorAction Stop
    if ($f.EndsWith('.dll') -and (Get-AuthenticodeSignature (Join-Path $upOut $f)).Status -ne 'Valid') { Write-Host "$f has no valid signature; not packaged"; exit 1 }
  }
}
# Paths and names of this machine must not be in the executables.
$needles = @($root, $root.Replace('\', '/'), $cargoHome, $env:USERPROFILE, $env:COMPUTERNAME) | Where-Object { $_ }
if ($env:USERNAME -and $env:USERNAME.Length -ge 4) { $needles += "\$($env:USERNAME)\" }
foreach ($f in 'dsport.exe', 'DemonsSoulsLauncher.exe') {
  $bytes = [IO.File]::ReadAllBytes((Join-Path $out $f))
  $text = [Text.Encoding]::GetEncoding(28591).GetString($bytes).ToLowerInvariant()
  $wide = [Text.Encoding]::Unicode.GetString($bytes).ToLowerInvariant()
  foreach ($n in $needles) {
    $k = $n.ToLowerInvariant()
    if ($text.Contains($k) -or $wide.Contains($k)) { Write-Host "$f contains a path or name of this machine ($n); not packaged"; exit 1 }
  }
}
# README.txt: the player README, the notices, the license (CRLF, UTF-8).
$read = { param($p) ([IO.File]::ReadAllText($p)).Replace("`r`n", "`n").TrimEnd() }
$rule = '=' * 78
$license = Join-Path $root 'release\LICENSE'
if (-not (Test-Path $license)) { $license = Join-Path $root 'LICENSE' }
$readme = (& $read (Join-Path $root 'docs\PLAYER_README.txt')) + "`n`n`n$rule`nThird-party notices`n$rule`n`n" + (& $read (Join-Path $root 'THIRD_PARTY_NOTICES.txt')) + "`n`n`n$rule`nLicense: GNU General Public License, version 3`n$rule`n`n" + (& $read $license) + "`n"
[IO.File]::WriteAllText((Join-Path $out 'README.txt'), $readme.Replace("`n", "`r`n"), (New-Object Text.UTF8Encoding $true))
# Allowlist: nothing else may be in the package.
$allowed = @('dsport.exe', 'DemonsSoulsLauncher.exe', 'README.txt') + @($upNames | ForEach-Object { "upscalers\$_" })
$extra = @(Get-ChildItem -Recurse -File $out | Where-Object { $allowed -notcontains $_.FullName.Substring($out.Length + 1) })
if ($extra.Count -gt 0) { Write-Host "unexpected files in the package: $($extra.FullName -join ', ')"; exit 1 }
$zip = Join-Path $dist "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path (Join-Path $out '*') -DestinationPath $zip
$size = (Get-ChildItem -Recurse -File $out | Measure-Object Length -Sum).Sum
$count = (Get-ChildItem -Recurse -File $out).Count
# The staged files become dist\DemonsSouls-Port-<version> unless the earlier folder is in use.
if (Test-Path $final) {
  if (Test-Path "$final.old") { Remove-Item -Recurse -Force "$final.old" -ErrorAction SilentlyContinue }
  try { Rename-Item $final "$name.old" -ErrorAction Stop } catch { }
}
if (-not (Test-Path $final)) {
  Move-Item $out $final -ErrorAction Stop
  if (Test-Path "$final.old") { Remove-Item -Recurse -Force "$final.old" -ErrorAction SilentlyContinue }
  $out = $final
} else {
  Write-Host "$final is in use (a program from it is running): the new package's files are in $out; the zip is complete"
}
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
Write-Host ("package {0}: {1} files, {2:N1} MB; zip {3:N1} MB, SHA-256 {4}" -f $out, $count, ($size / 1MB), ((Get-Item $zip).Length / 1MB), $hash)
