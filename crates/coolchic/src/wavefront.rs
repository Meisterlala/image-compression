//! Reference 9x9-ARM causal wavefront ordering, including narrow-grid raster order.
use crate::Result;

pub(crate) fn visit(
    width: usize,
    height: usize,
    mut f: impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    if width <= 9 {
        for y in 0..height {
            for x in 0..width {
                f(x, y)?;
            }
        }
    } else {
        for diagonal in 0..width + (height - 1) * 10 {
            let first = if diagonal < width {
                0
            } else {
                (diagonal - width) / 10 + 1
            };
            let last = (diagonal / 10).min(height - 1);
            for y in first..=last {
                f(diagonal - y * 10, y)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_cell_once_including_narrow_and_odd_grids() {
        for w in [1, 2, 8, 9, 10, 11, 19, 31] {
            for h in [1, 2, 5, 17] {
                let mut seen = vec![false; w * h];
                visit(w, h, |x, y| {
                    assert!(x < w && y < h);
                    assert!(!seen[y * w + x]);
                    seen[y * w + x] = true;
                    Ok(())
                })
                .unwrap();
                assert!(seen.iter().all(|v| *v));
            }
        }
    }
}
