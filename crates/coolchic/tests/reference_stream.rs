use coolchic::reference::{codec, stream};

const CASES: &[(&str, &[u8], &[u8])] = &[
    (
        "lop",
        include_bytes!("fixtures/full-reference/lop.cool"),
        include_bytes!("fixtures/full-reference/lop.rgba"),
    ),
    (
        "mop",
        include_bytes!("fixtures/full-reference/mop.cool"),
        include_bytes!("fixtures/full-reference/mop.rgba"),
    ),
    (
        "hop",
        include_bytes!("fixtures/full-reference/hop.cool"),
        include_bytes!("fixtures/full-reference/hop.rgba"),
    ),
    (
        "vhop",
        include_bytes!("fixtures/full-reference/vhop.cool"),
        include_bytes!("fixtures/full-reference/vhop.rgba"),
    ),
    (
        "common",
        include_bytes!("fixtures/full-reference/common.cool"),
        include_bytes!("fixtures/full-reference/common.rgba"),
    ),
    (
        "resize",
        include_bytes!("fixtures/full-reference/resize.cool"),
        include_bytes!("fixtures/full-reference/resize.rgba"),
    ),
    (
        "linear",
        include_bytes!("fixtures/full-reference/linear.cool"),
        include_bytes!("fixtures/full-reference/linear.rgba"),
    ),
];

#[test]
fn full_official_bitstreams_decode_and_reencode_exactly() {
    for &(name, bytes, expected) in CASES {
        let image = codec::decode(bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((image.width, image.height), (13, 7));
        assert_eq!(image.rgba, expected, "{name} exact official RGBA");
        let header = stream::header(bytes).unwrap();
        let network = stream::neural_network(&header).unwrap();
        let grids = codec::decode_latents(&network, &header.descriptors, header.latent).unwrap();
        let grids: Vec<_> = grids
            .iter()
            .map(|g| g.data.iter().map(|&v| v as i32).collect())
            .collect();
        assert_eq!(
            codec::encode(&network, &header.descriptors, &grids).unwrap(),
            bytes,
            "{name} byte-exact neural/header/range re-encoding"
        );
    }
}

#[test]
fn truncated_extended_and_malformed_streams_are_rejected_without_panics() {
    for &(_, bytes, _) in CASES {
        for end in (0..bytes.len()).step_by(23) {
            assert!(codec::decode(&bytes[..end]).is_err());
        }
        let mut extra = bytes.to_vec();
        extra.extend([0; 4]);
        assert!(codec::decode(&extra).is_err());
        let mut damaged = bytes.to_vec();
        damaged[0] = 255;
        assert!(codec::decode(&damaged).is_err());
        // Header failures and mutated small architectures must be handled,
        // not an assertion panic in an allocation or matrix kernel.
        for index in (0..bytes.len().min(64)).step_by(3) {
            let mut damaged = bytes.to_vec();
            damaged[index] ^= 0x40;
            if let Ok(header) = stream::header(&damaged)
                && header.architecture.height <= 128
                && header.architecture.width <= 128
            {
                assert!(std::panic::catch_unwind(|| codec::decode(&damaged)).is_ok());
            }
        }
    }
}
