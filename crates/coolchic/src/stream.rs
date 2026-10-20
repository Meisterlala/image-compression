//! Restricted, reference-compatible 5.0 headers and neural parameter serialization.
use crate::{
    Error, Result,
    bits::{Reader, Writer},
    model::Model,
};

const COUNTS: [usize; 8] = [12, 2, 0, 0, 6, 4, 18, 6];
const Q_INDEX: [u32; 8] = [0, 0, 0, 0, 0, 0, 0, 8];

fn finish(mut w: Writer, size_offset: usize) -> Vec<u8> {
    w.patch(size_offset, w.bits.len().div_ceil(8) as u32, 16);
    w.bytes()
}

pub(crate) fn pack(model: &Model, latent: &[u8]) -> Vec<u8> {
    let groups = model.parameter_groups();
    let mut orders = [0; 8];
    for (i, group) in groups.iter().enumerate() {
        orders[i] = (0..=12)
            .min_by_key(|&k| {
                group
                    .iter()
                    .map(|&x| {
                        let mut w = Writer::default();
                        w.signed_golomb(x, k);
                        w.bits.len()
                    })
                    .sum::<usize>()
            })
            .unwrap();
    }
    let mut nn = Writer::default();
    for (group, &k) in groups.iter().zip(&orders) {
        for &x in group {
            nn.signed_golomb(x, k);
        }
    }
    let padding = (8 - nn.bits.len() % 8) % 8;
    nn.bits.splice(0..0, std::iter::repeat_n(false, padding));
    let nn = nn.bytes();
    assert!(nn.len() < 1 << 14);

    let mut w = Writer::default();
    w.put(1, 12); // One image
    w.put(1, 12); // One intra
    w.put(0, 12); // No P frames
    w.put(0, 16); // Header byte count
    w.put(0, 12); // Intra at display index 0
    let mut out = finish(w, 36);

    let mut w = Writer::default();
    w.put(0, 12);
    w.put(0, 2);
    w.put(0, 2);
    w.put(0, 4); // I, RGB, 8 bit
    w.put(0, 16);
    out.extend(finish(w, 20));

    let mut w = Writer::default();
    w.put(0, 1);
    w.put(1, 3); // No synthesis stabiliser; one layer
    w.put(4, 4);
    w.put(1, 4); // Nearest-neighbor transpose filter / identity preconcat
    w.put(0, 5);
    w.put(6, 6);
    w.put(0, 1);
    w.put(0, 3); // No IFCE, linear six-context ARM
    w.put(model.height as u32, 14);
    w.put(model.width as u32, 14);
    w.put(0, 4);
    w.put(2, 4); // Three latents: full, half, quarter
    w.put(3, 5);
    w.put(0, 1);
    w.put(0, 1); // No hyperlatent/common randomness
    w.put(0, 2); // Nearest final interpolation (full-resolution output)
    for q in Q_INDEX {
        w.put(q, 5);
    }
    for k in orders {
        w.put(k, 4);
    }
    w.put(nn.len() as u32, 14);
    w.put(padding as u32, 3);
    w.put(latent.len() as u32, 28);
    let offset = w.bits.len();
    w.put(0, 16);
    w.put(3, 7);
    w.put(1, 4);
    w.put(0, 1);
    w.put(0, 1); // 3-1-linear-none
    out.extend(finish(w, offset));
    out.extend(nn);
    out.extend(latent);
    out
}

pub(crate) fn unpack(data: &[u8]) -> Result<(Model, &[u8])> {
    let mut r = Reader::new(data);
    r.expect(1, 12)?;
    r.expect(1, 12)?;
    r.expect(0, 12)?;
    let size = r.get(16)? as usize;
    r.expect(0, 12)?;
    let mut r = Reader::new(r.finish_header(size)?);
    r.expect(0, 12)?;
    r.expect(0, 2)?;
    r.expect(0, 2)?;
    r.expect(0, 4)?;
    let size = r.get(16)? as usize;
    let mut r = Reader::new(r.finish_header(size)?);
    r.expect(0, 1)?;
    r.expect(1, 3)?;
    r.expect(4, 4)?;
    r.expect(1, 4)?;
    r.expect(0, 5)?;
    r.expect(6, 6)?;
    r.expect(0, 1)?;
    r.expect(0, 3)?;
    let height = r.get(14)? as usize;
    let width = r.get(14)? as usize;
    crate::check_size(width, height)?;
    r.expect(0, 4)?;
    r.expect(2, 4)?;
    r.expect(3, 5)?;
    r.expect(0, 1)?;
    r.expect(0, 1)?;
    r.expect(0, 2)?;
    for q in Q_INDEX {
        r.expect(q, 5)?;
    }
    let mut orders = [0; 8];
    for k in &mut orders {
        *k = r.get(4)?;
        if *k > 12 {
            return Err(Error("Invalid neural exp-Golomb order"));
        }
    }
    let nn_size = r.get(14)? as usize;
    let pad = r.get(3)? as usize;
    let latent_size = r.get(28)? as usize;
    let size = r.get(16)? as usize;
    r.expect(3, 7)?;
    r.expect(1, 4)?;
    r.expect(0, 1)?;
    r.expect(0, 1)?;
    let data = r.finish_header(size)?;
    if data.len() != nn_size + latent_size || latent_size == 0 || !latent_size.is_multiple_of(4) {
        return Err(Error("Invalid Cool-chic payload length"));
    }
    let mut r = Reader::new(&data[..nn_size]);
    r.expect(0, pad)?;
    let mut groups: [Vec<i32>; 8] = std::array::from_fn(|_| Vec::new());
    for (i, group) in groups.iter_mut().enumerate() {
        for _ in 0..COUNTS[i] {
            let value = r.signed_golomb(orders[i])?;
            if value.unsigned_abs() > 1 << 20 {
                return Err(Error("Neural parameter exceeds profile limits"));
            }
            group.push(value);
        }
    }
    if r.pos != nn_size * 8 {
        return Err(Error("Unexpected trailing neural parameters"));
    }
    Ok((Model::from_groups(width, height, groups)?, &data[nn_size..]))
}
