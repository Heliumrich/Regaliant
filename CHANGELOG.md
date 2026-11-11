# Changelog

## 0.1.0 (unreleased draft)

The first version. It runs the PS5 game Demon's Souls (PPSA01341, version 01.007.000) from the
player's own game files on Windows.

- The game's own program runs on the PC's processor. Its system calls, audio, saves, controllers
  and video output are reimplemented for Windows. Its graphics commands and shaders run on Vulkan.
- A launcher checks the game files and their version, prepares the shaders ahead, lists save
  backups and restores them, and starts the game. It also offers a safe mode and a settings reset.
- PC settings inside the game's own menus (SETTINGS > PC): window mode with monitor, resolution and
  refresh rate, HDR, aspect ratios up to 32:9, render scale, sharpening, quality presets,
  geometry detail, texture filtering, anti-aliasing with DLSS, FSR or XeSS, bloom, vignette, a
  frame rate limit of 30, 60 or uncapped, menu and HUD size, HUD margin, button prompts, key and
  controller bindings, mouse sensitivity and inversion, sound output and device, the save folder
  and the import of a decrypted PS5 save.
- Keyboard and mouse with raw mouse look, key prompts in the game's style, a mouse pointer in
  menus, and Xbox prompts for Xbox controllers.
- 5.1 and 7.1 sound, headphone and Windows Sonic or Dolby Atmos output of the game's 3D sound.
- Trophies as achievements with a pop-up and a list.
- Saves are backed up before each session changes them.
