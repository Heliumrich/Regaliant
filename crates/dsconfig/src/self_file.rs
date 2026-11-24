//! SELF containers (PS4 and PS5 signed executables) whose segments are decrypted: some dumps keep
//! the decrypted executable and libraries in their container instead of a plain ELF (issue #8,
//! 2026-11-24: "eboot.bin is encrypted" for a decrypted dump that ran in other software). The
//! container is an ELF with its segments moved into entries; `to_elf` puts them back.
//!
//! The layout, read from the game's own retail container beside its decrypted ELF (field
//! offsets only): a 32-byte header (magic, version, mode, endianness, attributes, key type,
//! header and metadata sizes, the signed size, the entry count at 0x18), then 32-byte entries
//! (properties, file offset, stored size, plain size), then the ELF header and program headers
//! as the ELF has them. Entries come in pairs: a digest entry, then a data entry (properties bit
//! 11) whose bits 20-31 name the program header it fills, its plain size that segment's file
//! size. The version segment (PT_SCE_VERSION) has no entry: its bytes end the container, right
//! after the last data entry. Everything else of the ELF is zeros. Rebuilt so, the game's
//! container (with its segments decrypted) gives its decrypted ELF byte for byte.

/// The containers' first bytes: PS4 (0x1D3D154F) and PS5 (0xEEF51454), little-endian.
pub const PS4_MAGIC: [u8; 4] = [0x4F, 0x15, 0x3D, 0x1D];
pub const PS5_MAGIC: [u8; 4] = [0x54, 0x14, 0xF5, 0xEE];

const ENTRY: usize = 0x20;
const PHDR: usize = 0x38;
const PT_DYNAMIC: u32 = 2;
const PT_SCE_VERSION: u32 = 0x6fff_ff01;
/// Properties bit 11: the entry holds a segment's data.
const DATA_ENTRY: u64 = 1 << 11;
/// Properties bit 3: the entry's data is compressed.
const COMPRESSED: u64 = 1 << 3;

/// Whether `head` (a file's first bytes) starts a SELF container.
pub fn is_self(head: &[u8]) -> bool {
    head.len() >= 4 && (head[..4] == PS4_MAGIC || head[..4] == PS5_MAGIC)
}

/// `decrypted_elf`'s answer for a container whose segments are still encrypted (the game check
/// tells it apart from containers it cannot read).
pub const ENCRYPTED: &str = "SELF container: its segments are encrypted";

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at.checked_add(2)?)?.try_into().ok()?))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at.checked_add(4)?)?.try_into().ok()?))
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at.checked_add(8)?)?.try_into().ok()?))
}

/// One program header.
#[derive(Clone, Copy)]
struct Ph {
    ty: u32,
    flags: u32,
    off: usize,
    size: usize,
}

impl Ph {
    /// The first loadable executable segment's test: PT_LOAD with PF_X.
    fn code(&self) -> bool {
        self.ty == 1 && self.flags & 1 != 0
    }
}

fn phdrs(elf_head: &[u8], phoff: usize, phnum: usize) -> Option<Vec<Ph>> {
    (0..phnum)
        .map(|i| {
            let at = phoff.checked_add(i.checked_mul(PHDR)?)?;
            Some(Ph {
                ty: u32_at(elf_head, at)?,
                flags: u32_at(elf_head, at.checked_add(4)?)?,
                off: usize::try_from(u64_at(elf_head, at.checked_add(8)?)?).ok()?,
                size: usize::try_from(u64_at(elf_head, at.checked_add(0x20)?)?).ok()?,
            })
        })
        .collect()
}

/// The ELF a SELF container holds, its segments as the container has them (decrypted or not:
/// `plaintext` tells). Err: not a container this reads (compressed entries, sizes that do not
/// fit, a broken header).
pub fn to_elf(s: &[u8]) -> Result<Vec<u8>, String> {
    let bad = |what: &str| format!("SELF container: {what}");
    if !is_self(s) {
        return Err(bad("no SELF signature"));
    }
    let n = u16_at(s, 0x18).ok_or_else(|| bad("short header"))? as usize;
    let eh = 0x20 + n * ENTRY;
    if s.get(eh..eh + 4) != Some(b"\x7fELF".as_slice()) {
        return Err(bad("no ELF header after its entries"));
    }
    let phoff = usize::try_from(u64_at(s, eh + 0x20).ok_or_else(|| bad("short ELF header"))?).map_err(|_| bad("program header offset"))?;
    let phentsize = u16_at(s, eh + 0x36).ok_or_else(|| bad("short ELF header"))? as usize;
    let phnum = u16_at(s, eh + 0x38).ok_or_else(|| bad("short ELF header"))? as usize;
    if phentsize != PHDR || phoff < 0x40 || phoff > 0x1000 {
        return Err(bad("unexpected program headers"));
    }
    let head_len = phoff + phnum * PHDR;
    let head = s.get(eh..eh + head_len).ok_or_else(|| bad("program headers past its end"))?;
    let ph = phdrs(head, phoff, phnum).ok_or_else(|| bad("program headers"))?;
    let out_len = ph.iter().map(|p| p.off.saturating_add(p.size)).max().unwrap_or(0).max(head_len);
    // (A sane size: the ELF is about the container's size.)
    if out_len > s.len().saturating_mul(4).max(1 << 20) {
        return Err(bad("segments past any plausible size"));
    }
    let mut out = vec![0u8; out_len];
    out[..head_len].copy_from_slice(head);
    let mut covered = vec![false; phnum];
    // (The ELF's byte ranges the data entries filled.)
    let mut filled: Vec<(usize, usize)> = Vec::new();
    let mut data_end = 0usize;
    for i in 0..n {
        let at = 0x20 + i * ENTRY;
        let props = u64_at(s, at).ok_or_else(|| bad("short entry table"))?;
        if props & DATA_ENTRY == 0 {
            continue;
        }
        let seg = (props >> 20 & 0xfff) as usize;
        let (off, stored, plain) = (u64_at(s, at + 8), u64_at(s, at + 16), u64_at(s, at + 24));
        let (Some(off), Some(stored), Some(plain)) = (off, stored, plain) else { return Err(bad("short entry table")) };
        let (off, stored) = (usize::try_from(off).map_err(|_| bad("entry offset"))?, usize::try_from(stored).map_err(|_| bad("entry size"))?);
        if props & COMPRESSED != 0 || stored as u64 != plain {
            return Err(bad("compressed segments (not supported: convert it to an ELF)"));
        }
        let p = *ph.get(seg).ok_or_else(|| bad("an entry for a program header it lacks"))?;
        if stored != p.size {
            return Err(bad("an entry's size differs from its segment's"));
        }
        // (A segment never lies over the headers: else they would no longer be the container's.)
        if p.off < head_len && p.size != 0 {
            return Err(bad("a segment over its headers"));
        }
        let end = off.checked_add(stored).ok_or_else(|| bad("an entry past its end"))?;
        let data = s.get(off..end).ok_or_else(|| bad("an entry past its end (the file may be cut short)"))?;
        out[p.off..p.off + p.size].copy_from_slice(data);
        covered[seg] = true;
        filled.push((p.off, p.off + p.size));
        data_end = data_end.max(end);
    }
    // (What the loader needs, the dynamic table and the code, came from data entries: without
    // them the rebuilt ELF is zeros there. The game's dynamic table lies inside another
    // segment's data, so by bytes, not by entry.)
    let inside = |p: &Ph| p.size == 0 || filled.iter().any(|&(a, b)| a <= p.off && p.off + p.size <= b);
    if ph.iter().filter(|p| p.ty == PT_DYNAMIC).chain(ph.iter().find(|p| p.code())).any(|p| !inside(p)) {
        return Err(bad("its code or dynamic table has no data entry"));
    }
    // The version segment: the container's last bytes, after the last data entry. (A file that
    // ends at its data leaves it zeros: the loader does not read it, the version check then
    // tells a differing executable.)
    for (i, p) in ph.iter().enumerate() {
        if p.ty != PT_SCE_VERSION || covered[i] || p.size == 0 {
            continue;
        }
        if s.len() >= p.size && s.len() - p.size >= data_end && p.off >= head_len {
            out[p.off..p.off + p.size].copy_from_slice(&s[s.len() - p.size..]);
        }
    }
    Ok(out)
}

/// Whether an ELF's segments read as plain data: its dynamic table's tags before the terminator
/// all standard or system ones (an encrypted segment's random words never are), and its first
/// executable segment's bytes as varied as machine code, not as encrypted data (byte entropy of
/// the first MiB under 7.5 bits a byte: the game's code segments 6.3-6.6, encrypted data ~8.0).
/// False without a dynamic table.
pub fn plaintext(elf: &[u8]) -> bool {
    let (Some(phoff), Some(phnum)) = (u64_at(elf, 0x20), u16_at(elf, 0x38)) else { return false };
    let Ok(phoff) = usize::try_from(phoff) else { return false };
    let Some(ph) = phdrs(elf, phoff, phnum as usize) else { return false };
    let Some(d) = ph.iter().find(|p| p.ty == PT_DYNAMIC) else { return false };
    let Some(table) = elf.get(d.off..d.off.saturating_add(d.size)) else { return false };
    // (At least one tag before the terminator: a table of zeros is no table.)
    let mut tags = 0usize;
    let mut ended = false;
    for entry in table.chunks_exact(16) {
        let tag = u64::from_le_bytes(entry[..8].try_into().unwrap());
        if tag == 0 {
            ended = true;
            break;
        }
        let known = tag < 0x100 || (0x6000_0000..0x7000_0000).contains(&tag);
        if !known {
            return false;
        }
        tags += 1;
    }
    if !ended || tags == 0 {
        return false;
    }
    match ph.iter().find(|p| p.code()) {
        Some(c) => elf.get(c.off..c.off.saturating_add(c.size.min(1 << 20))).is_some_and(|b| entropy(b) < 7.5),
        None => true,
    }
}

/// Shannon entropy of `b`'s bytes, bits a byte.
fn entropy(b: &[u8]) -> f64 {
    if b.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &x in b {
        counts[x as usize] += 1;
    }
    let n = b.len() as f64;
    counts.iter().filter(|&&c| c > 0).map(|&c| c as f64 / n).map(|p| -p * p.log2()).sum()
}

/// `to_elf` of a container whose segments are plain: the ELF, or why not (an encrypted one
/// included).
pub fn decrypted_elf(s: &[u8]) -> Result<Vec<u8>, String> {
    let elf = to_elf(s)?;
    if !plaintext(&elf) {
        return Err(ENCRYPTED.into());
    }
    Ok(elf)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// `sample`'s container with its second data entry's bytes scrambled as encryption leaves
    /// them (the game check's tests).
    pub(crate) fn encrypted_sample() -> Vec<u8> {
        let (_, mut s) = sample();
        let e = 0x20 + 3 * ENTRY;
        let at = u64::from_le_bytes(s[e + 8..e + 16].try_into().unwrap()) as usize;
        for (i, b) in s[at..at + 0x200].iter_mut().enumerate() {
            *b = b.wrapping_mul(31).wrapping_add(i as u8 ^ 0x5c) | 0x80;
        }
        s
    }

    /// A small ELF (two loadable segments, one holding a dynamic table, a version segment, a
    /// trailing note of zeros) and the SELF container holding it as the game's does.
    pub(crate) fn sample() -> (Vec<u8>, Vec<u8>) {
        let phnum = 5usize;
        let mut elf = vec![0u8; 0x3000];
        elf[..4].copy_from_slice(b"\x7fELF");
        elf[0x20..0x28].copy_from_slice(&0x40u64.to_le_bytes());
        elf[0x36..0x38].copy_from_slice(&(PHDR as u16).to_le_bytes());
        elf[0x38..0x3a].copy_from_slice(&(phnum as u16).to_le_bytes());
        // (type, offset, size)
        let segs: [(u32, usize, usize); 5] = [(1, 0x1000, 0x400), (1, 0x2000, 0x200), (PT_DYNAMIC, 0x2100, 0x40), (PT_SCE_VERSION, 0x2200, 0x30), (4, 0x2230, 0x10)];
        for (i, &(ty, off, size)) in segs.iter().enumerate() {
            let at = 0x40 + i * PHDR;
            elf[at..at + 4].copy_from_slice(&ty.to_le_bytes());
            // (The first segment executable: PF_X.)
            elf[at + 4..at + 8].copy_from_slice(&(if i == 0 { 1u32 } else { 4 }).to_le_bytes());
            elf[at + 8..at + 16].copy_from_slice(&(off as u64).to_le_bytes());
            elf[at + 0x20..at + 0x28].copy_from_slice(&(size as u64).to_le_bytes());
        }
        // (Code-like bytes: a few values often, as machine code has them.)
        let ops = [0x48u8, 0x89, 0xe5, 0x8b, 0x45, 0xc3, 0x90, 0x0f, 0x1f, 0x00, 0xe8, 0x83];
        for (i, b) in elf[0x1000..0x1400].iter_mut().enumerate() {
            *b = ops[(i * 7 + i / 5) % ops.len()];
        }
        for (i, b) in elf[0x2000..0x2100].iter_mut().enumerate() {
            *b = (i * 13 + 1) as u8;
        }
        // The dynamic table: DT_NEEDED, a system tag, DT_NULL.
        let dynamic = [(1u64, 5u64), (0x6100_0035, 7), (0, 0)];
        for (i, &(t, v)) in dynamic.iter().enumerate() {
            let at = 0x2100 + i * 16;
            elf[at..at + 8].copy_from_slice(&t.to_le_bytes());
            elf[at + 8..at + 16].copy_from_slice(&v.to_le_bytes());
        }
        for (i, b) in elf[0x2200..0x2230].iter_mut().enumerate() {
            *b = (i * 5 + 9) as u8;
        }
        elf.truncate(0x2240);
        // The container: header, two entry pairs (digest, data), the ELF's headers, the data,
        // the version segment last.
        let n = 4usize;
        let head_len = 0x40 + phnum * PHDR;
        let mut s = vec![0u8; 0x20 + n * ENTRY];
        s[..4].copy_from_slice(&PS5_MAGIC);
        s[0x18..0x1a].copy_from_slice(&(n as u16).to_le_bytes());
        s.extend_from_slice(&elf[..head_len]);
        let mut entries = Vec::new();
        for (seg, &(_, off, size)) in segs.iter().enumerate().take(2) {
            let digest_at = s.len();
            s.extend_from_slice(&[0xAA; 0x20]);
            entries.push(((1u64 << 16) | ((seg as u64 * 2 + 1) << 20), digest_at, 0x20usize));
            let data_at = s.len();
            s.extend_from_slice(&elf[off..off + size]);
            entries.push((DATA_ENTRY | (1 << 13) | ((seg as u64) << 20), data_at, size));
        }
        s.extend_from_slice(&elf[0x2200..0x2230]);
        for (i, &(props, at, size)) in entries.iter().enumerate() {
            let e = 0x20 + i * ENTRY;
            s[e..e + 8].copy_from_slice(&props.to_le_bytes());
            s[e + 8..e + 16].copy_from_slice(&(at as u64).to_le_bytes());
            s[e + 16..e + 24].copy_from_slice(&(size as u64).to_le_bytes());
            s[e + 24..e + 32].copy_from_slice(&(size as u64).to_le_bytes());
        }
        (elf, s)
    }

    #[test]
    fn rebuilds_the_elf() {
        let (elf, s) = sample();
        assert!(is_self(&s));
        assert!(!is_self(&elf));
        assert_eq!(to_elf(&s).unwrap(), elf);
        assert_eq!(decrypted_elf(&s).unwrap(), elf);
        assert!(plaintext(&elf));
    }

    #[test]
    fn encrypted_segments_refused() {
        let (_, mut s) = sample();
        // (The data entry holding the dynamic table, its bytes scrambled as encryption leaves them.)
        let e = 0x20 + 3 * ENTRY;
        let at = u64::from_le_bytes(s[e + 8..e + 16].try_into().unwrap()) as usize;
        for (i, b) in s[at..at + 0x200].iter_mut().enumerate() {
            *b = b.wrapping_mul(31).wrapping_add(i as u8 ^ 0x5c) | 0x80;
        }
        assert!(to_elf(&s).is_ok());
        assert!(decrypted_elf(&s).unwrap_err().contains("encrypted"));
        // (Only the code segment encrypted, the dynamic table plain: its random bytes refuse it.)
        let (_, mut s) = sample();
        let e = 0x20 + ENTRY;
        let at = u64::from_le_bytes(s[e + 8..e + 16].try_into().unwrap()) as usize;
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        for b in s[at..at + 0x400].iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x >> 24) as u8;
        }
        assert!(to_elf(&s).is_ok());
        assert!(decrypted_elf(&s).unwrap_err().contains("encrypted"));
    }

    #[test]
    fn broken_containers_refused() {
        let (_, s) = sample();
        // Compressed data, a size past the end, a missing ELF header.
        let mut c = s.clone();
        let e = 0x20 + ENTRY;
        let props = u64::from_le_bytes(c[e..e + 8].try_into().unwrap()) | COMPRESSED;
        c[e..e + 8].copy_from_slice(&props.to_le_bytes());
        assert!(to_elf(&c).is_err());
        let mut c = s.clone();
        c[e + 16..e + 24].copy_from_slice(&(1u64 << 40).to_le_bytes());
        c[e + 24..e + 32].copy_from_slice(&(1u64 << 40).to_le_bytes());
        assert!(to_elf(&c).is_err());
        let mut c = s.clone();
        c[0x20 + 4 * ENTRY] = 0;
        assert!(to_elf(&c).is_err());
        assert!(to_elf(&s[..0x30]).is_err());
        assert!(to_elf(b"\x7fELF").is_err());
    }

    #[test]
    fn cut_short_empty_or_without_version() {
        let (elf, s) = sample();
        // Cut inside its last data entry (a copy that stopped): past its end.
        assert!(to_elf(&s[..s.len() - 0x40]).unwrap_err().contains("past its end"));
        // Ending right after its data (no version segment): that segment zeros, the rest as before.
        let short = &s[..s.len() - 0x30];
        let e = to_elf(short).unwrap();
        assert_eq!(&e[0x2200..0x2230], &[0u8; 0x30][..]);
        assert_eq!(&e[..0x2200], &elf[..0x2200]);
        assert!(decrypted_elf(short).is_ok());
        // No data entries: its code and dynamic table would be zeros.
        let mut none = s.clone();
        for i in 0..4 {
            let at = 0x20 + i * ENTRY;
            let props = u64::from_le_bytes(none[at..at + 8].try_into().unwrap()) & !DATA_ENTRY;
            none[at..at + 8].copy_from_slice(&props.to_le_bytes());
        }
        assert!(to_elf(&none).unwrap_err().contains("no data entry"));
        // A dynamic table of zeros is no table.
        let mut z = elf.clone();
        z[0x2100..0x2140].fill(0);
        assert!(!plaintext(&z));
        // The encrypted verdict is the one the game check tells apart.
        let mut enc = s.clone();
        let e = 0x20 + 3 * ENTRY;
        let at = u64::from_le_bytes(enc[e + 8..e + 16].try_into().unwrap()) as usize;
        for (i, b) in enc[at..at + 0x200].iter_mut().enumerate() {
            *b = b.wrapping_mul(31).wrapping_add(i as u8 ^ 0x5c) | 0x80;
        }
        assert_eq!(decrypted_elf(&enc).unwrap_err(), ENCRYPTED);
    }
}
