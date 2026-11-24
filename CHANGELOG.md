# Changelog

## 0.1.4

- Fixed the mostly black picture in play on RTX 50-series graphics cards (black shapes and streaks
  across the view from the first tunnel on). The game gives the unused slots of some draws
  "not a number" positions so that the graphics card drops them, and the port moves them out of
  view; the driver for these cards was allowed to skip that check and drew them across the
  screen. The game's shaders now tell the driver to keep it, on RTX 50-series cards and on AMD and
  Intel ones. Confirmed on an RTX 5090. On these cards the launcher prepares the shaders again
  once; RTX 40-series and older cards keep their prepared shaders.

## 0.1.3

- Game folders whose executable and libraries are decrypted but still in their SELF containers
  (as some dumps keep them) now work as they are, and so do .zar archives made from them.
  Before, the launcher called them encrypted and they had to be converted to ELF files first. A
  container the port cannot read, for example one cut short by a copy, now says why instead of
  being called encrypted.
- Fixed a timing fault on the graphics card in the texture streaming's statistics: resetting them
  could overtake reading them, and the game then kept textures at low detail. Vulkan's
  synchronization checks found it. It was not seen to cause problems on the tested cards.
- Your own pre-order and Digital Deluxe add-on packages count: put them in a folder named
  'Add-ons' in '%APPDATA%\Regaliant', and the game finds them as owned when it starts. Before,
  it was told that no add-ons were installed.
- Import PS5 Saves also takes save folders named 'sdimg_SAVEDATA...', as PS5 save tools name
  them. Before, such a save was not found. The PS5's own backups beside them are left out.
- Fixed a rare hang after slow first shader compiles: a part of the graphics work that fell
  behind could wait forever for a frame that had already passed, and the game stopped.

## 0.1.2

- On RTX 50-series graphics cards (and AMD and Intel ones) the game's shaders now tell the driver
  to keep each group of shader threads together after they take different paths. The game's
  translated shaders rely on it, and RTX 40-series cards do it on their own. On an RTX 5090, an
  RTX 5080 and an RTX 5060 Ti the tunnel at the game's start showed black shapes, stretched
  geometry or an almost black picture, in 0.1.1 too. This is the likely cause, not yet confirmed
  on such a card. On these cards the launcher prepares the shaders again once.
- **Open logs** in the launcher now selects the newest session's folder, the one to send with a
  bug report. The launcher's own quick checks keep their logs in a `checks` folder.
- When Windows reports more video memory in use by other programs than the graphics card has (a
  12 GB card reported 38 GB), the game no longer shrinks its texture memory to half the card
  because of it. That caused stutter, and restarts of the shader preparation, on such PCs. The
  log also names the programs that hold the most video memory.

## 0.1.1

- The game can be one .zar archive instead of its folder, read in place: about two thirds of the
  folder's size (54 GB instead of 82 GB for the tested copy). On the launcher's Game page,
  "Compress to .zar..." makes one from the game folder and then offers to play from it, and
  ".zar file..." chooses an existing one. The game folder is left as it is. On the tested PC the
  game started about 2 s sooner from the archive and loaded levels from the drive as fast or
  faster; reloading a level still in memory took about 0.3 s longer. Frame rate and stutter were
  the same.
- PlayStation controllers work directly, without Steam Input: DualSense, DualSense Edge and
  DUALSHOCK 4, over USB or Bluetooth, also when connected while the game runs. While you use one,
  the game shows its own PlayStation buttons. Vibration and the light bar work on them.
- All connected controllers play together as one pad. Before, only the first Xbox controller was
  read.
- The picture fills the window and the screen in every display mode when Windows display scaling is
  above 100%. Before, it could cover only the top-left part, as on a 4K screen at 175% or a 1440p
  screen at 200%.
- On RTX 50-series graphics cards (and AMD and Intel ones) every compute shader now runs in its
  portable form. The faster forms some shaders used were only checked on RTX 40-series cards, and
  on an RTX 5090 and an RTX 5080 the tunnel at the game's start was drawn black. This is the likely
  cause, not yet confirmed on such a card.

## 0.1.0

The first version of Regaliant. It runs the PS5 game Demon's Souls (PPSA01341, version 01.007.000) from the
player's own game files on Windows.

- The game's own program runs on the PC's processor. Its system calls, audio, saves, controllers
  and video output are reimplemented for Windows. Its graphics commands and shaders run on Vulkan.
- A launcher checks the game files and their version, sets the display (window mode, monitor,
  resolution, refresh rate, render scale, frame rate limit), prepares the shaders ahead (the game
  plays a new game and then its levels, enemies, bosses and cutscenes by itself in the background,
  hidden and silent: about an hour once on a fast PC, longer on others), lists save backups and restores them, and starts the game. It also offers a safe mode, a settings reset and
  a button that opens the Discord invite.
- PC settings inside the game's own menus (SETTINGS > PC): window mode with monitor, resolution and
  refresh rate, HDR, aspect ratios up to 32:9, render scale, sharpening, quality presets,
  geometry detail, shadow distance, texture filtering, anti-aliasing with DLSS, FSR or XeSS, bloom, vignette, a
  frame rate limit of 30, 60 or uncapped, menu and HUD size, HUD margin, button prompts, key and
  controller bindings, mouse sensitivity and inversion, a motion blur fix that keeps plants sharp
  while the camera moves, sound output and device, the save folder and the import of a decrypted
  PS5 save.
- Keyboard and mouse with raw mouse look, key prompts in the game's style, a mouse pointer that
  points and clicks in the menus (it selects what it points at at once and hides in cutscenes), a text box in the game's window for the character's name
  (with a keyboard of its own for controllers), and Xbox prompts for Xbox controllers.
- 5.1 and 7.1 sound, headphone and Windows Sonic or Dolby Atmos output of the game's 3D sound.
  The controller speaker's sounds can play through the PC's speakers (Pad Speaker).
- The game keeps its normal speed at any frame rate, and its frames are paced evenly at any
  refresh rate, with G-SYNC and FreeSync in borderless and exclusive fullscreen.
- Shaders the launcher did not prepare compile in the background where that is safe: an object
  can be missing for a moment instead of the game pausing.
- The shader preparation looks after itself: a step that stops responding or crashes runs once
  more, and when preparing cannot finish on a PC (no usable graphics card, a full drive, steps
  that keep failing) the launcher says why and Play starts the game. An interrupted preparation
  goes on where it stopped, and after a graphics driver or port update only the compile is done
  again. Some graphics drivers stop taking prepared shaders after Windows events. The launcher
  notices this when it starts and compiles them again. On battery, Play starts the game without
  preparing, and a preparation stops when the battery runs low.
- Trophies as achievements with a pop-up and a list.
- Saves are backed up before each session changes them.

Fixed and added while 0.1.0 was prepared, before it was published:

- A shorter freeze when the world first appears after loading a save: large textures the game
  creates while arriving are prepared on the graphics card again.
- 3D textures are prepared on the graphics card, which takes work off the processor while playing.
- Fewer crashes while shaders are prepared: a shader whose preparation crashed the graphics
  driver is not prepared ahead again.
- No more flicker right after loading: the world no longer vanishes in every other frame for
  a moment.
- Smoother frame rates where many shadows are drawn: clearing shadows no longer stalls the
  graphics card.
- No more flicker while turning the camera at the start of the first level.
- The shader preparation finishes on more PCs. Pressing Prepare or Play while the launcher
  checks the game now waits for the check. An old preparation is never taken as done.
- When the graphics driver crashes on a shader, the preparation leaves that shader out and goes
  on. Play does the same.
- When the graphics card runs out of memory, the game no longer slows to a crawl or stops
  responding.
- When graphics memory runs short, textures now go to system memory instead of being left out,
  so the picture stays whole. The frame rate can drop in busy places.
- Graphics cards with 4 to 6 GB of video memory: the game runs there and draws the whole
  picture, slowly. On cards with 4 or 5 GB, large images stay in system memory from the start.
- Fixed a crash when graphics memory ran out while the game made room for its work.
- The launcher says what to expect on a card with little video memory.
- A long preparation that runs short of graphics memory goes on in a fresh session.
- PCs with 16 GB of memory: in a simulation of one, the game no longer freezes at level loads.
  When the graphics card's share of memory runs short, the game frees memory it has not used
  lately.
- A preparation step that finished its work is no longer run again when it is slow to close.
- Prepare now on prepared shaders gives shaders that once crashed the graphics driver another
  chance.
