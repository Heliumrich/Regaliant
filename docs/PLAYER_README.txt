Demon's Souls - unofficial PC port 0.1.0 for Windows (64-bit)
=============================================================

An unofficial PC port of Demon's Souls (PS5, PPSA01341). It is not rebuilt from the game's
source code. It runs the game's own PS5 program from your copy of the game's files. The
program's x86-64 code runs directly on your processor. The PS5 system libraries it calls are
reimplemented for Windows, and its graphics commands and shaders are translated to Vulkan.
Nothing of the game is included.

Not affiliated with, endorsed or sponsored by Sony Interactive Entertainment, Bluepoint Games,
FromSoftware or any other rights holder.

This is an early version. Read "Known issues" below before you play.


What you need
-------------
- Your own copy of Demon's Souls for PS5 (PPSA01341) at version 01.007.000, as a folder. The
  folder holds sce_sys, the game's data folders and eboot.bin. The executable and its two
  libraries must be decrypted: a decrypted eboot.bin (in the folder or in its "decrypted"
  folder) and decrypted libc.prx and libSceNpCppWebApi.prx in sce_module.
- Windows 10 or 11, 64-bit.
- A graphics card and driver with Vulkan 1.3.
- A processor with AVX2 (Intel from 2013 on, AMD Ryzen).
- Disk space: up to 4 GB for prepared shaders, besides the game files.

Tested: Ryzen 9 5900X (12 cores), 32 GB of memory and an NVIDIA RTX 4090. See "System
requirements" for what was measured and what is estimated.


Getting started
---------------
1. Unpack this zip into a folder, for example C:\Games\DemonsSouls-Port.
2. Start DemonsSoulsLauncher.exe. If Windows says "Windows protected your PC", click
   "More info", then "Run anyway". The programs are not code-signed.
3. On the Game page, choose the game folder. The launcher checks it and says what is missing.
4. Click Play. The first time, the launcher prepares the game's shaders before the game
   starts. This takes about 15 to 45 minutes once. You can turn it off on the Shaders page.


Controls
--------
Controllers: Xbox controllers and other XInput controllers. A DualSense or DualShock 4 works
through Steam Input, which presents it as an Xbox controller. The prompts show the
controller's own buttons.

Keyboard and mouse (defaults, Dark Souls III-style):
  W A S D            move                      Left Alt         walk
  mouse              camera                    I J K L          camera (keys)
  Space              dash (hold), roll, back in menus
  left button        attack                    Shift+left button  strong attack
  right button       guard                     Left Ctrl        parry
  F                  two-hand                  R                use item
  E, Enter           interact, confirm         Q, middle button lock on
  G                  tool belt                 Escape           menu
  arrow keys, wheel  switch spells, items and weapons
  F1                 free the mouse            Alt+Enter        fullscreen or window

In menus the mouse pointer selects what it points at, the left button confirms and the right
button goes back. The prompts show the keys you bound. Every key can be changed in the game:
SETTINGS > PC > Key Bindings.


PC settings
-----------
The game's SETTINGS screen has a PC tab with the port's settings. Most take effect at once.
- Display: window mode (windowed, borderless or exclusive fullscreen), monitor, resolution and
  refresh rate, HDR, aspect ratio (16:9, 16:10, 21:9, 32:9), render scale, sharpening.
- Graphics: quality preset, geometry detail, texture filtering, anti-aliasing (the game's TAA,
  off, DLSS, FSR or XeSS), bloom, vignette, frame rate limit (30, 60 or uncapped).
- Game and controls: menu and HUD size, HUD margin, button prompts, mouse sensitivity and
  inversion, mouse look blur, skip intro logos, save folder, import of a PS5 save, key and
  controller bindings.
- Sound: output (headphones, stereo, 5.1, 7.1, Windows Sonic or Dolby Atmos) and device.

DLSS needs one more file. NVIDIA's nvngx_dlss.dll is under NVIDIA's own license and does not
come with the port. Copy your own into the "upscalers" folder beside dsport.exe. Games with DLSS
have the file in their installation folder. Then DLSS appears in the Anti-Aliasing list. XeSS and
FSR are in the "upscalers" folder already. Each library is checked for its maker's digital
signature before it is loaded. NVIDIA and DLSS are trademarks of NVIDIA Corporation. Intel and
XeSS are trademarks of Intel Corporation. AMD and FSR are trademarks of Advanced Micro Devices,
Inc.


The launcher
------------
- Game: the game folder and its check, what your PC brings, and safe mode for the next start.
  Safe mode starts the game in a window with the default settings and without an upscaler. Use
  it when the game does not start.
- Shaders: prepares the game's shaders ahead. Without it the game stutters while it compiles
  each shader the first time. Prepared shaders hold for one graphics card and driver. After a
  driver update the launcher asks to prepare them again.
- Saves: the save folder and the backups the game makes. The game copies each save before a
  session first changes it. "Restore..." puts a backup back. The save as it is now is kept
  as a backup first.
- Reset settings puts the PC settings, display choices and key bindings back to their
  defaults. The old files are kept as .bak files.
- Open logs opens the folder with each session's log.


Settings, saves and logs
------------------------
- Settings, key bindings and saves: %APPDATA%\DemonsSoulsPort.
- Logs, prepared shaders and the game's own temporary files: %LOCALAPPDATA%\DemonsSoulsPort.
- Portable use: an empty file named portable.txt next to dsport.exe keeps everything in a
  "user" folder beside it.
- Logs leave out your user name, your computer's name and your profile folder.


When something goes wrong
-------------------------
The game tells you when it cannot go on, for example when the graphics card stops responding.
It then closes and names its log folder. Please report the problem on the project's GitHub
issues page and attach that session's log folder (dsport.log, and crash.txt when there is
one). If the game does not start at all, try safe mode in the launcher.


System requirements
-------------------
Tested on one PC: Ryzen 9 5900X, 32 GB of memory, NVIDIA RTX 4090 with driver 610.88,
Windows 11.

Estimated, not tested:
- Memory: see "Known issues".
- Graphics: Vulkan 1.3. AMD and Intel graphics cards have not been tested.
- Processor: the game uses many threads. Fewer cores lower the frame rate.


Known issues
------------
- The frame rate is below 60 in most places. On the tested PC it is about 30 to 45 fps.
- The first visit to an area can still stutter for a moment while graphics shaders compile,
  even with prepared shaders.
- Areas, bosses and the story were reached by the developer's shortcuts in testing. A full
  playthrough has not been made.
- World 8-3 sometimes loads too dark. Restarting the game fixes it.
- Water outside the Nexus can show bands. Some other rare picture faults are known.
- At 32:9 the character preview in the equipment menu stands at the right edge.
- HDR has not been seen on an HDR screen yet.
- AMD and Intel graphics cards have not been tested.
- Online features (messages, bloodstains, co-op and invasions) are not available. The game
  plays offline.
- Linux, Proton and the Steam Deck are not supported yet.


License
-------
Copyright (C) 2026 SulphurS16 and the Demon's Souls unofficial PC port contributors.

The port is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version. It is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY, without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License, whose full text follows
at the end of this file, for more details.

Additional permission under GNU GPL version 3 section 7: If you modify this program, or any
covered work, by linking or combining it with NVIDIA DLSS (the NGX runtime nvngx_dlss.dll and
the NVIDIA Streamline libraries) or Intel XeSS (libxess.dll), or modified versions of those
libraries, containing parts covered by the terms of their own licenses, the licensors of this
program grant you additional permission to convey the resulting work. The Corresponding Source
for a non-source form of such a combination need not include the source code of those
libraries.

The license covers the port's own code only, not the game. The notices of the third-party
components in the two programs follow below.
