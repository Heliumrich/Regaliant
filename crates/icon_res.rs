// The program icon (assets/regaliant.ico) as a Windows resource, written without a resource
// compiler: the launcher's and the runtime's build scripts include this file. The MSVC linker
// takes the .res file among its inputs (beside an embedded manifest).

const RT_ICON: u16 = 3;
const RT_GROUP_ICON: u16 = 14;
/// US English, as a resource compiler marks resources by default.
const LANG_EN_US: u16 = 0x0409;

/// Writes `<OUT_DIR>/regaliant.res` from the icon file `ico` (its images as icons 1..n, the
/// group as icon 1: the programs' icon and their windows' class icon) and links it into the
/// package's programs.
fn link_icon(ico: &std::path::Path) {
    println!("cargo:rerun-if-changed={}", ico.display());
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }
    let data = std::fs::read(ico).unwrap_or_else(|e| panic!("{}: {e}", ico.display()));
    let u16_at = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
    assert!(data.len() >= 6 && u16_at(0) == 0 && u16_at(2) == 1, "{}: not an icon file", ico.display());
    let count = u16_at(4) as usize;
    assert!(count > 0 && data.len() >= 6 + 16 * count, "{}: no images", ico.display());
    // The empty resource a .res file begins with, each image as an icon, then the group: the
    // file's directory with resource numbers in place of file offsets.
    let mut res = Vec::new();
    resource(&mut res, 0, 0, 0, 0, &[]);
    let mut group = Vec::with_capacity(6 + 14 * count);
    group.extend_from_slice(&[0, 0, 1, 0]);
    group.extend_from_slice(&(count as u16).to_le_bytes());
    for i in 0..count {
        let e = 6 + 16 * i;
        let (size, off) = (u32_at(e + 8) as usize, u32_at(e + 12) as usize);
        assert!(off.checked_add(size).is_some_and(|end| end <= data.len()), "{}: image {i} out of bounds", ico.display());
        let id = i as u16 + 1;
        resource(&mut res, RT_ICON, id, 0x1010, LANG_EN_US, &data[off..off + size]);
        // Width, height, colours, reserved, planes, bit count and size as in the file.
        group.extend_from_slice(&data[e..e + 12]);
        group.extend_from_slice(&id.to_le_bytes());
    }
    resource(&mut res, RT_GROUP_ICON, 1, 0x1030, LANG_EN_US, &group);
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("regaliant.res");
    std::fs::write(&out, &res).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
    println!("cargo:rustc-link-arg-bins={}", out.display());
}

/// Appends one resource to a .res file: its header (type and name as numbers) and its data,
/// padded to four bytes.
fn resource(res: &mut Vec<u8>, ty: u16, name: u16, flags: u16, lang: u16, data: &[u8]) {
    res.extend_from_slice(&(data.len() as u32).to_le_bytes());
    res.extend_from_slice(&32u32.to_le_bytes());
    res.extend_from_slice(&[0xff, 0xff]);
    res.extend_from_slice(&ty.to_le_bytes());
    res.extend_from_slice(&[0xff, 0xff]);
    res.extend_from_slice(&name.to_le_bytes());
    res.extend_from_slice(&0u32.to_le_bytes()); // data version
    res.extend_from_slice(&flags.to_le_bytes());
    res.extend_from_slice(&lang.to_le_bytes());
    res.extend_from_slice(&0u32.to_le_bytes()); // version
    res.extend_from_slice(&0u32.to_le_bytes()); // characteristics
    res.extend_from_slice(data);
    while res.len() % 4 != 0 {
        res.push(0);
    }
}
