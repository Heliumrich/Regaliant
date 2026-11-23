# Settings and saves

[Back to the main page](../README.md)

## Settings

- In the game: SETTINGS > PC. The tab has window mode, resolution, render scale, Anti-Aliasing / Upscaler, the frame rate limit, the Motion Blur Fix and mouse sensitivity. Its lists hold the rest: Graphics & Display, Controls, Sound & Saves, and the key and controller bindings. Most settings take effect at once.
- In the launcher: the **Display** page sets the window mode, monitor, resolution, refresh rate, render scale and frame rate limit before the game starts.
- Motion Blur Fix is on by default. It turns the game's motion blur off, so plants and detail stay sharp while the camera moves. Off uses the game's own Motion Blur setting.
- Shadow Distance (Graphics & Display) shortens how far the sun's shadows reach. A shorter distance raises the frame rate in busy places.

## Frame rate and refresh rate

The game runs at the same speed at every frame rate. With the 60 limit each frame stays on the screen for the same number of refreshes, so motion stays even:

- At 60, 120 or 240 Hz the game shows 60 frames a second.
- With G-SYNC or FreeSync it shows 60 in borderless or exclusive fullscreen.
- In a window on a 165 Hz or 144 Hz screen it shows an even 55 or 48 (G-SYNC is often on only for fullscreen). For 60 there, use borderless or exclusive fullscreen with G-SYNC or FreeSync, or set the screen to 120 Hz.
- Exclusive fullscreen with Refresh Rate on Automatic picks a rate where 60 fits, unless the screen has G-SYNC or FreeSync.

## DLSS, FSR and XeSS

NVIDIA's 'nvngx_dlss.dll' is under NVIDIA's own license and not included. Copy your own into the 'upscalers' folder beside 'regaliant.exe' (games with DLSS have it in their installation folder). Then DLSS appears in the Anti-Aliasing list. XeSS, FSR and NVIDIA Streamline ship in the 'upscalers' folder with their licenses, and each is loaded only after its maker's digital signature checks out.

NVIDIA and DLSS are trademarks of NVIDIA Corporation. Intel and XeSS are trademarks of Intel Corporation. AMD and FSR are trademarks of Advanced Micro Devices, Inc.

## Saves and files

- Settings and saves are in '%APPDATA%\Regaliant'. Before a session first changes a save, the game copies it to 'save-backups' there. The launcher's **Saves** page restores a backup.
- Logs and prepared shaders are in '%LOCALAPPDATA%\Regaliant'. The launcher's **Open logs** button opens the logs.
- An empty file named 'portable.txt' next to 'regaliant.exe' keeps everything in a 'user' folder beside it.
- When the game cannot go on, it says so and names its log folder. Logs leave out your user name, your computer's name and your profile folder.
