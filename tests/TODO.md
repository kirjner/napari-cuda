Planned unit tests for AVCC/VT refactor (to add in a later pass)

- avcC parsing
  - parse_avcc: valid avcC with 1 SPS/1 PPS returns expected nal_length_size and arrays
  - parse_avcc: rejects truncated headers and lengths

- Splitters and conversions
  - split_annexb: detects and splits start-code-delimited NALs correctly
  - split_avcc_by_len: splits for 1,2,3,4 byte length sizes; handles malformed gracefully
  - annexb_to_avcc and avcc_to_annexb are inverses for simple AU cases

- SPS/PPS extraction helpers
  - find_sps_pps picks NAL type 7/8 from mixed sets
  - extract_sps_pps_from_blob works for AnnexB and AVCC blobs; heuristic fallback covered

- AccessUnit semantics
  - AccessUnit dataclass construction and basic attributes
  - Keyframe gating logic in packer-based flows (first keyframe required)

- VT decode initialization (macOS-only, skipped elsewhere)
  - VTLiveDecoder initializes from avcC and decodes a minimal AU sequence
  - Flush/drain behavior yields expected counts

