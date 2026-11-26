# Drops the capabilities naga declares but a kernel does not use (Image1D 44, StorageImageExtendedFormats 49)
# from a SPIR-V module in place: python scripts/spv_strip_caps.py <file.spv>
import struct,sys
d=open(sys.argv[1],'rb').read(); w=list(struct.unpack('<%dI'%(len(d)//4),d))
out=w[:5]; i=5
while i<len(w):
    op=w[i]&0xffff; n=w[i]>>16
    if op==17 and w[i+1] in (44,49): i+=n; continue
    out+=w[i:i+n]; i+=n
open(sys.argv[1],'wb').write(struct.pack('<%dI'%len(out),*out))
