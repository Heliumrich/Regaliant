# Building from source

[Back to the main page](../README.md)

1. Install Rust with rustup (the x86_64-pc-windows-msvc toolchain) and the Visual Studio Build Tools with the C++ workload.
2. Run `scripts\upscalers.ps1` once. It downloads the upscalers' libraries (FSR, XeSS and NVIDIA Streamline) with their licenses into `local-data\upscalers` and checks their signatures. NVIDIA's own DLSS file is never packaged.
3. Run `scripts\package.ps1`. It builds both programs with the release settings and writes `dist\Regaliant-<version>.zip`. `-NoUpscalers` leaves the upscalers out.

`scripts\cargo.ps1 build --release` builds the programs without packaging them. The game itself is never part of the build. The programs read it from the folder you choose in the launcher.

## Profiling with Tracy

`scripts\cargo.ps1 build --release --features tracy` builds the game with Tracy's client (off in every other build). Start the game as usual, then connect the Tracy profiler **v0.14.1** (the version the `tracy-client` crate bundles; another version cannot connect) from the same PC: nothing is recorded until it connects.

What it shows:

- A frame per game flip.
- Zones on the port's threads: the command processors (`cp run`, `cp chunk`, `cp wait reg mem`), draws and dispatches and their recording (`draw`, `draw record`, `dispatch`, `dispatch record`, `bind stage`), shader preparation (`prepare ps`, `prepare ngg`), the image cache (`images lock wait`, `images (held)`), batches (`begin batch (slot)`, `submit batch`, `flush`, `vkQueueSubmit2`), their completion (`wait batch on GPU`, `batch completion (CP effects)`), the upscaler and the present.
- Every PS5 system-library call the game makes, named after the function, on the game's threads.
- A GPU track per queue with every batch's time on the GPU, named by its queue, number and leading programs (from the batch timestamps; needs VK_EXT_calibrated_timestamps, which NVIDIA and AMD drivers offer).

Run Tracy as administrator for context switches and CPU sampling. The game runs in a second `dsport.exe` under the first (its host, which answers page faults): only the game's own process has Tracy's client, so the profiler lists one `dsport.exe`.

A capture without the profiler's window: `tracy-capture.exe -o capture.tracy -s 15` (15 seconds, from the Tracy folder, while the game runs). `tracy-csvexport.exe capture.tracy > zones.csv` lists every zone's count and times.

The GPU's own cost per shader comes from the existing profile:

1. Set `DS_PROFILE_GPU=1` and `DS_CONTROL=<a text file>` before starting the launcher (PowerShell: `$env:DS_PROFILE_GPU = "1"`, `$env:DS_CONTROL = "F:\ctl.txt"`; the game started by the launcher inherits them).
2. In game, add the line `profile reset` to that file and save it.
3. Play 10 to 15 seconds, then add the line `profile report` and save.

The game runs each line added to the file (and logs it); the report goes to the game's log (`dsport.log`), not to the file: the GPU time by compute shader and by pixel program.
