/* Thin wrapper over minimp3 (CC0) for dsport: float output, heap decoder state. */
#define MINIMP3_IMPLEMENTATION
#define MINIMP3_FLOAT_OUTPUT
#include "minimp3.h"
#include <stdlib.h>

typedef struct { int frame_bytes, frame_offset, channels, hz, layer, bitrate_kbps; } ds_mp3_info;

void* ds_mp3_new(void) {
    mp3dec_t* d = (mp3dec_t*)malloc(sizeof(mp3dec_t));
    if (d) mp3dec_init(d);
    return d;
}
void ds_mp3_free(void* d) { free(d); }
/* Decodes one frame; returns samples per channel (0 = none), fills info (frame_bytes = bytes consumed). */
int ds_mp3_decode(void* d, const unsigned char* data, int len, float* pcm, ds_mp3_info* info) {
    mp3dec_frame_info_t fi;
    int n = mp3dec_decode_frame((mp3dec_t*)d, data, len, pcm, &fi);
    info->frame_bytes = fi.frame_bytes;
    info->frame_offset = fi.frame_offset;
    info->channels = fi.channels;
    info->hz = fi.hz;
    info->layer = fi.layer;
    info->bitrate_kbps = fi.bitrate_kbps;
    return n;
}
