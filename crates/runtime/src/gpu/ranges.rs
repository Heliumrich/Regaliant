//! Unions of guest byte ranges for the command processors' dependency
//! checks: overlapping and touching ranges are coalesced, so a batch that
//! writes the same buffers thousands of times (the title's GPU-driven culling
//! writes its 112 MB index buffer and argument slots once per visible item)
//! keeps a few entries, and an overlap test is one ordered lookup.

use std::collections::BTreeMap;

/// Union of half-open ranges [start, end), kept disjoint and coalesced.
///
/// A filter of the 64 KiB pages it covers (1024 bits, hashed) answers most
/// overlap tests without the ordered lookup: the command processors test
/// ~15-40 descriptor and constant reads per draw against the writes of the
/// batch being recorded (~0.5 us per test with lookups, run
/// 20261011-135920). Ranges over FILTER_BIG_PAGES pages are kept exactly
/// beside it (`big`, coalesced, few): they used to switch the filter off, and
/// the title's culls write their 112 MB output buffer in every dispatch, so
/// every test of a batch holding one took the lookup. Removals leave their
/// bits and big entries (stale ones only cost a lookup).
#[derive(Default, Clone, Debug)]
pub struct Ranges {
    by_start: BTreeMap<u64, u64>,
    filter: [u64; 16],
    big: Vec<(u64, u64)>,
}

const FILTER_SHIFT: u32 = 16;
const FILTER_BIG_PAGES: u64 = 64;

fn filter_bit(page: u64) -> usize {
    (page.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - 10)) as usize
}

impl Ranges {
    pub const fn new() -> Ranges {
        Ranges { by_start: BTreeMap::new(), filter: [0; 16], big: Vec::new() }
    }

    fn filter_add(&mut self, a: u64, n: u64) {
        let (first, last) = (a >> FILTER_SHIFT, (a.saturating_add(n) - 1) >> FILTER_SHIFT);
        if last - first >= FILTER_BIG_PAGES {
            // Exactly, merged with the big ranges it overlaps or touches.
            let (mut s, mut e) = (a, a.saturating_add(n));
            self.big.retain(|&(bs, be)| {
                if bs <= e && s <= be {
                    s = s.min(bs);
                    e = e.max(be);
                    false
                } else {
                    true
                }
            });
            self.big.push((s, e));
            return;
        }
        for page in first..=last {
            let b = filter_bit(page);
            self.filter[b / 64] |= 1 << (b % 64);
        }
    }

    /// False when [a, a + n) certainly does not overlap.
    fn filter_may(&self, a: u64, n: u64) -> bool {
        let end = a.saturating_add(n);
        // `tune exact_big 0`: any big range sends every test to the lookup
        // (the earlier behavior, for A/B).
        if !self.big.is_empty() && !super::tune::exact_big() {
            return true;
        }
        if self.big.iter().any(|&(s, e)| s < end && a < e) {
            return true;
        }
        let (first, last) = (a >> FILTER_SHIFT, (end - 1) >> FILTER_SHIFT);
        if last - first >= FILTER_BIG_PAGES {
            return true;
        }
        (first..=last).any(|page| {
            let b = filter_bit(page);
            self.filter[b / 64] & (1 << (b % 64)) != 0
        })
    }

    /// Adds [a, a + n).
    pub fn insert(&mut self, a: u64, n: u64) {
        if n == 0 {
            return;
        }
        self.filter_add(a, n);
        let mut start = a;
        let mut end = a.saturating_add(n);
        // A predecessor that overlaps or touches absorbs the new range.
        if let Some((&s, &e)) = self.by_start.range(..=start).next_back() {
            if e >= start {
                if e >= end {
                    return;
                }
                start = s;
                self.by_start.remove(&s);
            }
        }
        // Successors starting inside or right after it merge into it.
        while let Some((&s, &e)) = self.by_start.range(start..=end).next() {
            end = end.max(e);
            self.by_start.remove(&s);
        }
        self.by_start.insert(start, end);
    }

    /// Whether the union holds all of [a, a + n) (touching ranges are
    /// merged, so one range must).
    pub fn contains(&self, a: u64, n: u64) -> bool {
        if n == 0 {
            return true;
        }
        let end = a.saturating_add(n);
        self.by_start.range(..=a).next_back().is_some_and(|(_, &e)| e >= end)
    }

    /// Whether [a, a + n) overlaps the union.
    pub fn overlaps(&self, a: u64, n: u64) -> bool {
        if n == 0 || self.by_start.is_empty() || !self.filter_may(a, n) {
            return false;
        }
        let end = a.saturating_add(n);
        // Only the last range starting before `end` can reach past `a`.
        self.by_start.range(..end).next_back().is_some_and(|(_, &e)| e > a)
    }

    pub fn overlaps_any(&self, reads: &[(u64, u64)]) -> bool {
        reads.iter().any(|&(a, n)| self.overlaps(a, n))
    }

    pub fn is_empty(&self) -> bool {
        self.by_start.is_empty()
    }

    pub fn clear(&mut self) {
        self.by_start.clear();
        self.filter = [0; 16];
        self.big.clear();
    }

    /// Number of disjoint ranges.
    pub fn len(&self) -> usize {
        self.by_start.len()
    }

    /// The disjoint ranges as (start, length), in address order.
    pub fn iter(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.by_start.iter().map(|(&s, &e)| (s, e - s))
    }

    pub fn extend(&mut self, other: &Ranges) {
        for (a, n) in other.iter() {
            self.insert(a, n);
        }
    }

    /// Removes [a, a + n) from the union.
    pub fn remove(&mut self, a: u64, n: u64) {
        if n == 0 {
            return;
        }
        let end = a.saturating_add(n);
        let hit: Vec<(u64, u64)> = self.by_start.range(..end).rev().take_while(|&(_, &e)| e > a).map(|(&s, &e)| (s, e)).collect();
        for (s, e) in hit {
            self.by_start.remove(&s);
            if s < a {
                self.by_start.insert(s, a);
            }
            if e > end {
                self.by_start.insert(end, e);
            }
        }
    }

    /// The parts of the union inside [a, a + n), as (start, length).
    pub fn intersect(&self, a: u64, n: u64) -> Vec<(u64, u64)> {
        if n == 0 {
            return Vec::new();
        }
        let end = a.saturating_add(n);
        let mut v: Vec<(u64, u64)> = self.by_start.range(..end).rev().take_while(|&(_, &e)| e > a).map(|(&s, &e)| (s.max(a), e.min(end))).map(|(s, e)| (s, e - s)).collect();
        v.reverse();
        v
    }
}

/// Pages of the summary (64 KiB) and its buckets (hashed pages).
const SUMMARY_SHIFT: u32 = 16;
const SUMMARY_BUCKETS: usize = 1 << 14;
/// Ranges over this many pages are kept exactly instead (`big`): the
/// unbounded pointer windows (up to 256 MiB) would mark a quarter of the
/// buckets.
const BIG_PAGES: u64 = 64;

fn summary_bucket(page: u64) -> usize {
    (page.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - 14)) as usize
}

/// Written ranges by batch sequence number: the latest batch that writes
/// into a range, for batches that have not completed.
///
/// A summary answers most lookups without the per-batch search (a draw's
/// ~20-40 descriptor and constant reads against every unfinished batch of
/// its queue: ~0.7 us per check, run 20261011-135122): per bucket of pages
/// the latest batch that writes into one of them, and the large ranges
/// exactly. Neither is lowered when batches complete: a value at or below
/// `done` simply no longer counts.
#[derive(Default, Debug)]
pub struct SeqRanges {
    by_seq: BTreeMap<u64, Ranges>,
    summary: Vec<u64>,
    big: Vec<(u64, u64, u64)>,
}

impl SeqRanges {
    pub const fn new() -> SeqRanges {
        SeqRanges { by_seq: BTreeMap::new(), summary: Vec::new(), big: Vec::new() }
    }

    fn summarize(&mut self, seq: u64, a: u64, n: u64) {
        if n == 0 {
            return;
        }
        let end = a.saturating_add(n);
        let (first, last) = (a >> SUMMARY_SHIFT, (end - 1) >> SUMMARY_SHIFT);
        if last - first >= BIG_PAGES {
            self.big.push((a, end, seq));
            return;
        }
        if self.summary.is_empty() {
            self.summary = vec![0; SUMMARY_BUCKETS];
        }
        for page in first..=last {
            let b = &mut self.summary[summary_bucket(page)];
            *b = (*b).max(seq);
        }
    }

    /// Whether some batch after `done` may write into one of `reads`
    /// (false: certainly none).
    pub fn may_overlap(&self, reads: &[(u64, u64)], done: u64) -> bool {
        reads.iter().any(|&(a, n)| {
            if n == 0 {
                return false;
            }
            let end = a.saturating_add(n);
            if self.big.iter().any(|&(s, e, q)| q > done && s < end && a < e) {
                return true;
            }
            let (first, last) = (a >> SUMMARY_SHIFT, (end - 1) >> SUMMARY_SHIFT);
            if last - first >= BIG_PAGES {
                return true;
            }
            !self.summary.is_empty() && (first..=last).any(|page| self.summary[summary_bucket(page)] > done)
        })
    }

    pub fn insert(&mut self, seq: u64, a: u64, n: u64) {
        if n != 0 {
            self.by_seq.entry(seq).or_default().insert(a, n);
            self.summarize(seq, a, n);
        }
    }

    pub fn insert_all(&mut self, seq: u64, r: &Ranges) {
        if !r.is_empty() {
            self.by_seq.entry(seq).or_default().extend(r);
            for (a, n) in r.iter() {
                self.summarize(seq, a, n);
            }
        }
    }

    /// Drops the batches up to `done` (completed).
    pub fn prune(&mut self, done: u64) {
        while let Some((&s, _)) = self.by_seq.first_key_value() {
            if s > done {
                break;
            }
            self.by_seq.pop_first();
        }
        if self.big.iter().any(|&(_, _, q)| q <= done) {
            self.big.retain(|&(_, _, q)| q > done);
        }
    }

    /// The latest batch after `done` that writes into any of `reads`.
    pub fn latest(&self, reads: &[(u64, u64)], done: u64) -> Option<u64> {
        if !self.may_overlap(reads, done) {
            return None;
        }
        for (&s, r) in self.by_seq.iter().rev() {
            if s <= done {
                break;
            }
            if r.overlaps_any(reads) {
                return Some(s);
            }
        }
        None
    }

    pub fn clear(&mut self) {
        self.by_seq.clear();
        self.summary.clear();
        self.big.clear();
    }

    /// Total disjoint ranges over all batches (diagnostics).
    pub fn ranges(&self) -> usize {
        self.by_seq.values().map(Ranges::len).sum()
    }
}

#[cfg(test)]
mod summary_tests {
    use super::*;

    /// The summary never hides a write of an unfinished batch.
    #[test]
    fn summary_keeps_every_unfinished_write() {
        let mut s = SeqRanges::new();
        s.insert(5, 0x1_0000_0000, 0x100);
        s.insert(6, 0x2_0000_0000, 0x1000_0000); // big: 256 MiB
        assert_eq!(s.latest(&[(0x1_0000_0080, 4)], 0), Some(5));
        assert_eq!(s.latest(&[(0x2_0800_0000, 4)], 0), Some(6));
        assert_eq!(s.latest(&[(0x3_0000_0000, 4)], 0), None);
        // Completed batches no longer count.
        s.prune(5);
        assert_eq!(s.latest(&[(0x1_0000_0080, 4)], 5), None);
        assert_eq!(s.latest(&[(0x2_0800_0000, 4)], 5), Some(6));
        s.prune(6);
        assert_eq!(s.latest(&[(0x2_0800_0000, 4)], 6), None);
        assert!(s.big.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(r: &Ranges) -> Vec<(u64, u64)> {
        r.iter().collect()
    }

    #[test]
    fn coalesces_overlapping_and_touching() {
        let mut r = Ranges::default();
        r.insert(100, 10); // [100,110)
        r.insert(120, 10); // [120,130)
        assert_eq!(collect(&r), vec![(100, 10), (120, 10)]);
        r.insert(110, 10); // touches both -> [100,130)
        assert_eq!(collect(&r), vec![(100, 30)]);
        r.insert(105, 5); // contained
        assert_eq!(collect(&r), vec![(100, 30)]);
        r.insert(90, 200); // covers all -> [90,290)
        assert_eq!(collect(&r), vec![(90, 200)]);
        r.insert(400, 0); // empty: ignored
        assert_eq!(r.len(), 1);
        r.insert(300, 10);
        r.insert(320, 10);
        r.insert(295, 30); // [295,325) overlaps both -> [295,330)
        assert_eq!(collect(&r), vec![(90, 200), (295, 35)]);
    }

    #[test]
    fn overlap_queries() {
        let mut r = Ranges::default();
        r.insert(100, 10);
        r.insert(200, 50);
        assert!(!r.overlaps(0, 100)); // ends where the first starts
        assert!(r.overlaps(0, 101));
        assert!(r.overlaps(109, 1));
        assert!(!r.overlaps(110, 90)); // the gap exactly
        assert!(r.overlaps(110, 91));
        assert!(r.overlaps(150, 1000));
        assert!(!r.overlaps(250, 10));
        assert!(!r.overlaps(105, 0));
        assert!(r.overlaps_any(&[(0, 1), (240, 20)]));
        assert!(!r.overlaps_any(&[(0, 1), (250, 20)]));
        // A huge range at the top of the address space.
        r.insert(u64::MAX - 10, 100);
        assert!(r.overlaps(u64::MAX - 1, 1));
    }

    #[test]
    fn containment() {
        let mut r = Ranges::default();
        r.insert(100, 10);
        r.insert(110, 10); // touching: one range [100,120)
        r.insert(200, 50);
        assert!(r.contains(100, 20));
        assert!(r.contains(105, 5));
        assert!(!r.contains(95, 10)); // starts before
        assert!(!r.contains(115, 10)); // runs past the end
        assert!(!r.contains(150, 1)); // the gap
        assert!(!r.contains(110, 100)); // spans the gap
        assert!(r.contains(249, 1));
        assert!(!r.contains(250, 1));
        assert!(r.contains(7, 0)); // empty
    }

    #[test]
    fn overlap_tests_match_brute_force_with_big_ranges() {
        // xorshift: deterministic pseudo-random ranges, small (bytes to a
        // few pages) and big (over FILTER_BIG_PAGES pages of 64 KiB).
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for round in 0..20 {
            let mut r = Ranges::new();
            let mut set: Vec<(u64, u64)> = Vec::new();
            for k in 0..200 {
                let a = 0x10_0000_0000 + (next() % (1 << 34)) / 4 * 4;
                let n = if next() % 16 == 0 { (64 + next() % 4096) << 16 } else { 4 + next() % (3 << 16) };
                if k % 23 == 22 && !set.is_empty() {
                    // Remove part of an inserted range.
                    let (s, e) = set[(next() % set.len() as u64) as usize];
                    let cut = (s + (e - s) / 3, (e - s) / 3 + 1);
                    r.remove(cut.0, cut.1);
                    let (cs, ce) = (cut.0, cut.0 + cut.1);
                    let mut out = Vec::new();
                    for &(s, e) in &set {
                        if e <= cs || ce <= s {
                            out.push((s, e));
                        } else {
                            if s < cs {
                                out.push((s, cs));
                            }
                            if ce < e {
                                out.push((ce, e));
                            }
                        }
                    }
                    set = out;
                } else {
                    r.insert(a, n);
                    set.push((a, a + n));
                }
            }
            for _ in 0..2000 {
                let a = 0x10_0000_0000 + (next() % (1 << 34)) / 4 * 4;
                let n = if next() % 8 == 0 { (1 + next() % 128) << 16 } else { 4 + next() % 256 };
                let want = set.iter().any(|&(s, e)| s < a + n && a < e);
                assert_eq!(r.overlaps(a, n), want, "round {round}: [{a:#x}, +{n:#x})");
            }
            // Queries inside the inserted ranges always overlap.
            for &(s, e) in &set {
                assert!(r.overlaps(s, 4.min(e - s)), "round {round}: inside [{s:#x}, {e:#x})");
            }
        }
    }

    #[test]
    fn remove_and_intersect() {
        let mut r = Ranges::default();
        r.insert(100, 100); // [100,200)
        r.insert(300, 50); // [300,350)
        assert_eq!(r.intersect(150, 200), vec![(150, 50), (300, 50)]);
        assert_eq!(r.intersect(0, 100), vec![]);
        assert_eq!(r.intersect(320, 10), vec![(320, 10)]);
        r.remove(120, 10); // splits the first
        assert_eq!(collect(&r), vec![(100, 20), (130, 70), (300, 50)]);
        r.remove(190, 120); // trims the second, cuts the third's start
        assert_eq!(collect(&r), vec![(100, 20), (130, 60), (310, 40)]);
        r.remove(0, 1000);
        assert!(r.is_empty());
        r.insert(10, 10);
        r.remove(10, 0); // empty: no change
        assert_eq!(collect(&r), vec![(10, 10)]);
    }

    #[test]
    fn latest_writer_by_sequence() {
        let mut s = SeqRanges::default();
        s.insert(5, 100, 10);
        s.insert(7, 105, 10);
        s.insert(9, 500, 10);
        assert_eq!(s.latest(&[(100, 1)], 0), Some(5));
        assert_eq!(s.latest(&[(108, 1)], 0), Some(7));
        assert_eq!(s.latest(&[(108, 1)], 7), None); // 5 and 7 completed
        assert_eq!(s.latest(&[(0, 1000)], 0), Some(9));
        assert_eq!(s.latest(&[(200, 10)], 0), None);
        s.prune(7);
        assert_eq!(s.latest(&[(100, 20)], 0), None);
        assert_eq!(s.ranges(), 1);
    }
}
