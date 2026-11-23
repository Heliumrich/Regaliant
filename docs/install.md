# Install, update and uninstall

[Back to the main page](../README.md)

## What you need

- Windows 10 or 11, 64-bit.
- A graphics card with a Vulkan 1.3 driver.
- A processor with AVX2: Intel from the Haswell generation (2013) on, or AMD Ryzen.
- Your own copy of Demon's Souls for PS5 (PPSA01341) at version 01.007.000, as a folder that holds sce_sys, the game's data folders and eboot.bin. The executable eboot.bin and the libraries libc.prx and libSceNpCppWebApi.prx in sce_module must be decrypted (in the folder itself or in its decrypted folder).
- Up to 5 GB of disk space for prepared shaders.

See [System requirements](requirements.md) for what was tested.

## Install

1. Download `Regaliant-<version>.zip` from the [latest release](https://github.com/SulphurS16/Regaliant/releases/latest).
2. Right click the zip, choose **Extract All...**, and pick a folder, for example 'C:\Games\Regaliant'.
3. Start 'regaliant-launcher.exe'. If Windows says "Windows protected your PC" (the programs are not code-signed), click **More info**, then **Run anyway**.
4. On **Game**, choose the game folder. The launcher checks it and says what is missing.
5. Click **Play**. The first time, the launcher prepares the game's shaders before the game starts: the game plays a new game and its levels by itself in the background, hidden and silent, to learn them. This takes about an hour once on a fast PC and longer on others, and the launcher says how long. It can be stopped and continued later, and turned off on the **Shaders** page. On battery, Play starts the game without preparing: plug in the PC, then press **Play** or **Prepare now**.

## Update

Unzip the new release over the old one or into a new folder, and start its 'regaliant-launcher.exe'. Settings, key bindings and saves are in '%APPDATA%\Regaliant', which every version uses. Prepared shaders in '%LOCALAPPDATA%\Regaliant' carry on too.

## Uninstall

Delete the folder. Settings and saves are in '%APPDATA%\Regaliant', logs and prepared shaders in '%LOCALAPPDATA%\Regaliant'.
