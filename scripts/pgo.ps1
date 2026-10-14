# Profile-guided optimization of dsport on top of the release-lto profile.
#   .\scripts\pgo.ps1 gen     instrumented build -> target\release-pgo-gen\dsport.exe; its runs
#                             (DSPORT_PROFILE=release-pgo-gen) write counters to local-data\pgo\raw
#                             when they shut down (`quit`)
#   .\scripts\pgo.ps1 merge   local-data\pgo\raw\*.profraw -> local-data\pgo\merged.profdata
#   .\scripts\pgo.ps1 use     optimized build -> target\release-pgo\dsport.exe
# llvm-profdata comes from the project-local copy of rustc's own llvm-tools component (it must match
# rustc's LLVM; the shared toolchain is not changed): local-data\tools\llvm-tools-<version>-...
# Builds go through --target so the flags reach dsport's crates but not build scripts.
# Note: do not use ErrorActionPreference=Stop here; cargo writes progress to stderr.
param([Parameter(Mandatory = $true)][ValidateSet('gen', 'merge', 'use')][string]$Step)
$root = Split-Path -Parent $PSScriptRoot
$pgo = Join-Path $root 'local-data\pgo'
$raw = Join-Path $pgo 'raw'
$merged = Join-Path $pgo 'merged.profdata'
$target = 'x86_64-pc-windows-msvc'
$cargo = Join-Path $PSScriptRoot 'cargo.ps1'
# Arguments separated by 0x1f: the paths contain a space.
function Set-Flags([string[]]$flags) { $env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]0x1f }
function Copy-Build([string]$prof) {
    $from = Join-Path $root "target\$target\$prof"
    $to = Join-Path $root "target\$prof"
    New-Item -ItemType Directory -Force $to | Out-Null
    Copy-Item (Join-Path $from 'dsport.exe') $to -Force
    if (Test-Path (Join-Path $from 'dsport.pdb')) { Copy-Item (Join-Path $from 'dsport.pdb') $to -Force }
    "copied to target\$prof\dsport.exe"
}
switch ($Step) {
    'gen' {
        New-Item -ItemType Directory -Force $raw | Out-Null
        Set-Flags @("-Cprofile-generate=$raw", '--cfg', 'dsport_pgo_gen')
        & $cargo build --profile release-pgo-gen --target $target -p dsport
        if ($LASTEXITCODE -ne 0) { "build failed"; exit 1 }
        Copy-Build 'release-pgo-gen'
    }
    'merge' {
        $tool = Get-ChildItem (Join-Path $root 'local-data\tools') -Recurse -Filter 'llvm-profdata.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $tool) { "llvm-profdata.exe not found under local-data\tools"; exit 2 }
        $files = @(Get-ChildItem $raw -Filter '*.profraw' -ErrorAction SilentlyContinue)
        if (-not $files.Count) { "no .profraw files in $raw"; exit 2 }
        & $tool.FullName merge -o $merged @($files | ForEach-Object { $_.FullName })
        if ($LASTEXITCODE -ne 0) { "merge failed"; exit 1 }
        "merged $($files.Count) file(s) into $merged ($((Get-Item $merged).Length) bytes)"
    }
    'use' {
        if (-not (Test-Path $merged)) { "no $merged (run merge first)"; exit 2 }
        Set-Flags @("-Cprofile-use=$merged", '-Cllvm-args=-pgo-warn-mismatch=false')
        & $cargo build --profile release-pgo --target $target -p dsport
        if ($LASTEXITCODE -ne 0) { "build failed"; exit 1 }
        Copy-Build 'release-pgo'
    }
}
