//! Unions of guest byte ranges for the command processors' dependency
//! checks: overlapping and touching ranges are coalesced, so a batch that
//! writes the same buffers thousands of times (the title's GPU-driven culling
//! writes its 112 MB index buffer and argument slots once per visible item)
//! keeps a few entries, and an overlap test is one ordered lookup.

use std::collections::BTreeMap;

/// Union of half-open ranges [start, end), kept disjoint and coalesced.
#[derive(Default, Clone, Debug)]
pub struct Ranges {
    by_start: BTreeMap<u64, u64>,
}

impl Ranges {
    /// Adds [a, a + n).
    pub fn insert(&mut self, a: u64, n: u64) {
        if n == 0 {
            return;
        }
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

    /// Whether [a, a + n) overlaps the union.
    pub fn overlaps(&self, a: u64, n: u64) -> bool {
        if n == 0 {
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
}

/// Written ranges by batch sequence number: the latest batch that writes
/// into a range, for batches that have not completed.
#[derive(Default, Debug)]
pub struct SeqRanges {
    by_seq: BTreeMap<u64, Ranges>,
}

impl SeqRanges {
    pub const fn new() -> SeqRanges {
        SeqRanges { by_seq: BTreeMap::new() }
    }

    pub fn insert(&mut self, seq: u64, a: u64, n: u64) {
        if n != 0 {
            self.by_seq.entry(seq).or_default().insert(a, n);
        }
    }

    pub fn insert_all(&mut self, seq: u64, r: &Ranges) {
        if !r.is_empty() {
            self.by_seq.entry(seq).or_default().extend(r);
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
    }

    /// The latest batch after `done` that writes into any of `reads`.
    pub fn latest(&self, reads: &[(u64, u64)], done: u64) -> Option<u64> {
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
    }

    /// Total disjoint ranges over all batches (diagnostics).
    pub fn ranges(&self) -> usize {
        self.by_seq.values().map(Ranges::len).sum()
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
