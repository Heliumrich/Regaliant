//! SHA-256 through Windows' own provider (BCrypt): the game files' fingerprints.

use std::io::Read;
use std::path::Path;
use windows_sys::Win32::Security::Cryptography::{BCRYPT_HASH_HANDLE, BCRYPT_SHA256_ALG_HANDLE, BCryptCreateHash, BCryptDestroyHash, BCryptFinishHash, BCryptHashData};

/// The file's SHA-256 as 64 lowercase hex digits.
pub fn file_sha256(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h: BCRYPT_HASH_HANDLE = std::ptr::null_mut();
    let st = unsafe { BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE, &mut h, std::ptr::null_mut(), 0, std::ptr::null(), 0, 0) };
    if st != 0 {
        return Err(std::io::Error::other(format!("BCryptCreateHash {st:#x}")));
    }
    let mut buf = vec![0u8; 1 << 20];
    let r = (|| {
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            let st = unsafe { BCryptHashData(h, buf.as_ptr(), n as u32, 0) };
            if st != 0 {
                return Err(std::io::Error::other(format!("BCryptHashData {st:#x}")));
            }
        }
        let mut out = [0u8; 32];
        let st = unsafe { BCryptFinishHash(h, out.as_mut_ptr(), 32, 0) };
        if st != 0 {
            return Err(std::io::Error::other(format!("BCryptFinishHash {st:#x}")));
        }
        Ok(out.iter().map(|b| format!("{b:02x}")).collect())
    })();
    unsafe { BCryptDestroyHash(h) };
    r
}

#[cfg(test)]
mod tests {
    #[test]
    fn known_value() {
        let p = std::env::temp_dir().join(format!("dsconfig-sha-{}", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        let h = super::file_sha256(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
