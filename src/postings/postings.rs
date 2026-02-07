use crate::docset::DocSet;

/// Postings (also called inverted list)
///
/// For a given term, it is the list of doc ids of the doc
/// containing the term. Optionally, for each document,
/// it may also give access to the term frequency
/// as well as the list of term positions.
///
/// Its main implementation is `SegmentPostings`,
/// but other implementations mocking `SegmentPostings` exist,
/// for merging segments or for testing.
pub trait Postings: DocSet + 'static {
    /// The number of times the term appears in the document.
    fn term_freq(&self) -> u32;

    /// Returns the positions offsetted with a given value.
    /// It is not necessary to clear the `output` before calling this method.
    /// The output vector will be resized to the `term_freq`.
    fn positions_with_offset(&mut self, offset: u32, output: &mut Vec<u32>) {
        output.clear();
        self.append_positions_with_offset(offset, output);
    }

    /// Returns the positions offsetted with a given value.
    /// Data will be appended to the output.
    fn append_positions_with_offset(&mut self, offset: u32, output: &mut Vec<u32>);

    /// Returns the positions of the term in the given document.
    /// The output vector will be resized to the `term_freq`.
    fn positions(&mut self, output: &mut Vec<u32>) {
        self.positions_with_offset(0u32, output);
    }

    /// Returns the character byte offsets (offset_from, offset_to) for each occurrence of the
    /// term in the current document. Clears output first.
    ///
    /// Only available when the field was indexed with
    /// [`IndexRecordOption::WithFreqsAndPositionsAndOffsets`].
    fn offsets(&mut self, output: &mut Vec<(u32, u32)>) {
        output.clear();
        self.append_offsets(output);
    }

    /// Appends byte offsets to the output without clearing.
    /// This is the append-variant of `offsets()`, analogous to `append_positions_with_offset`.
    fn append_offsets(&mut self, _output: &mut Vec<(u32, u32)>) {}

    /// Appends `(position, offset_from, offset_to)` tuples for the current document.
    /// Keeps positions and byte offsets correlated in a single sorted collection.
    /// Falls back to positions with zero offsets when offsets are not indexed.
    fn append_positions_and_offsets(&mut self, offset: u32, output: &mut Vec<(u32, u32, u32)>) {
        let base = output.len();
        let mut positions = Vec::new();
        self.append_positions_with_offset(offset, &mut positions);
        let mut byte_offsets = Vec::new();
        self.append_offsets(&mut byte_offsets);
        for (i, &pos) in positions.iter().enumerate() {
            let (from, to) = byte_offsets.get(i).copied().unwrap_or((0, 0));
            output.push((pos, from, to));
        }
        output[base..].sort_unstable_by_key(|&(pos, _, _)| pos);
    }
}

impl Postings for Box<dyn Postings> {
    fn term_freq(&self) -> u32 {
        (**self).term_freq()
    }

    fn append_positions_with_offset(&mut self, offset: u32, output: &mut Vec<u32>) {
        (**self).append_positions_with_offset(offset, output);
    }

    fn append_offsets(&mut self, output: &mut Vec<(u32, u32)>) {
        (**self).append_offsets(output);
    }

    fn append_positions_and_offsets(&mut self, offset: u32, output: &mut Vec<(u32, u32, u32)>) {
        (**self).append_positions_and_offsets(offset, output);
    }
}
