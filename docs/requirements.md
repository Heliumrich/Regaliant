# System requirements

[Back to the main page](../README.md)

## Tested

One PC: Ryzen 9 5900X (12 cores), 32 GB of memory, NVIDIA RTX 4090 (driver 610.88), Windows 11.

## Memory

Measured on that PC: the game reserves about 35 to 40 GB of memory and page file. While you play an area it keeps about 4 to 5 GB in active use. 32 GB is recommended. 16 GB has not been tested on a real PC. The graphics card reads part of the game's memory directly, and Windows lets it use about half of the installed memory for that. In a simulation of a 16 GB PC on the tested PC, the game played every tested level at the same frame rates, and a first session's level loads took about as long. With 16 GB, keep Windows' page file on automatic and about 30 GB free on the system drive.

## Graphics card

The card needs Vulkan 1.3. Only an RTX 4090 was tested. Smaller cards were simulated on it by giving the game less video memory. That shows how much memory the game needs, but not how fast a smaller card is.

- **12 GB or more:** the game fits.
- **8 to 12 GB:** in busy places the game keeps some textures in system memory. The picture stays right. The frame rate there can drop.
- **6 GB:** the game fits once some textures go to system memory. In the tested levels it drew right, at the same frame rates as with 7 GB on the 4090.
- **4 GB:** the game runs and draws right in the tested levels, but even on the 4090 it ran at about 9 to 30 frames a second with what a 4 GB card leaves a game. On such a card, large images go to system memory and the graphics card reads them over its connection to the PC. Cards with 4 GB are usually slow graphics chips too, so expect low frame rates. The Low preset in the game's PC tab lowers the graphics card's work. A 4 GB card in a PC with 16 GB of memory has not been tested: Windows then has less room for those images, and parts of the game may draw wrong.
- **Less than 4 GB:** not supported. With what a 3 GB card leaves a game, parts of some levels drew wrong in the simulation.

Graphics features, from the cards' published Vulkan support (not tested):

- NVIDIA GTX 900 series and newer have everything the game uses. The GTX 10 series and older lack fragment shader barycentrics. The game draws some surfaces through an extra shader stage there, which costs some speed. Intel Arc A-series, Meteor Lake and Arrow Lake graphics lack them too. This was only simulated on an NVIDIA card.
- AMD and Intel graphics cards have not been tested. Older AMD cards (RX 400 and 500 series, Vega) run the game's shaders in groups of 64, not the 32 the game was written for. Some effects may look wrong there.

## Processor

Limited to 6 of its 12 cores, the tested processor kept 91 to 94% of its frame rate. A slower processor lowers the frame rate in proportion to its single-core speed.
