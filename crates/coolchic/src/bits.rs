//! MSB-first reference headers and signed order-k exponential-Golomb codes.
use crate::{Error, Result};

#[derive(Default)]
pub(crate) struct Writer {
    pub bits: Vec<bool>,
}
impl Writer {
    pub fn put(&mut self, value: u32, n: usize) {
        assert!(n == 32 || value < (1 << n));
        self.bits
            .extend((0..n).rev().map(|i| value & (1 << i) != 0));
    }
    pub fn bytes(mut self) -> Vec<u8> {
        self.bits.resize(self.bits.len().div_ceil(8) * 8, false);
        self.bits
            .chunks_exact(8)
            .map(|b| b.iter().fold(0, |a, &v| (a << 1) | u8::from(v)))
            .collect()
    }
    pub fn patch(&mut self, offset: usize, value: u32, n: usize) {
        for i in 0..n {
            self.bits[offset + i] = value & (1 << (n - i - 1)) != 0;
        }
    }
    pub fn signed_golomb(&mut self, x: i32, k: u32) {
        let unsigned = if x <= 0 {
            -2 * i64::from(x)
        } else {
            2 * i64::from(x) - 1
        } as u64;
        let v = unsigned + (1 << k);
        let high = 63 - v.leading_zeros();
        self.bits
            .extend(std::iter::repeat_n(false, (high - k) as usize));
        for i in (0..=high).rev() {
            self.bits.push(v & (1 << i) != 0);
        }
    }
}

pub(crate) struct Reader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}
impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    pub fn get(&mut self, n: usize) -> Result<u32> {
        if n > 32 || self.pos + n > self.data.len() * 8 {
            return Err(Error("Truncated Cool-chic bitstream"));
        }
        let mut value = 0;
        for i in self.pos..self.pos + n {
            value = (value << 1) | u32::from((self.data[i / 8] >> (7 - i % 8)) & 1);
        }
        self.pos += n;
        Ok(value)
    }
    pub fn expect(&mut self, value: u32, n: usize) -> Result<()> {
        if self.get(n)? != value {
            return Err(Error(
                "Unsupported Cool-chic profile (only the experimental Rust RGB image profile is supported)",
            ));
        }
        Ok(())
    }
    pub fn finish_header(&mut self, n: usize) -> Result<&'a [u8]> {
        if n != self.pos.div_ceil(8) || n > self.data.len() {
            return Err(Error("Invalid Cool-chic header length"));
        }
        while self.pos < n * 8 {
            self.expect(0, 1)?;
        }
        Ok(&self.data[n..])
    }
    pub fn signed_golomb(&mut self, k: u32) -> Result<i32> {
        let mut zeros = 0;
        while self.get(1)? == 0 {
            zeros += 1;
            if zeros + k > 30 {
                return Err(Error("Cool-chic neural parameter is out of range"));
            }
        }
        let suffix = self.get((zeros + k) as usize)?;
        let x = (1u32 << (zeros + k)) + suffix - (1 << k);
        Ok(if x & 1 == 1 {
            (x / 2 + 1) as i32
        } else {
            -(x as i32 / 2)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn golomb_orders_and_signs() {
        for k in 0..=12 {
            let mut w = Writer::default();
            for x in -1000..=1000 {
                w.signed_golomb(x, k);
            }
            let bytes = w.bytes();
            let mut r = Reader::new(&bytes);
            for x in -1000..=1000 {
                assert_eq!(r.signed_golomb(k).unwrap(), x);
            }
        }
    }
}
