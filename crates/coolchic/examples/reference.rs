//! Development-only native entry point for the full reference path.
use coolchic::reference::{codec, stream};
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        return Err("Usage: reference decode|shift INPUT.cool OUTPUT".into());
    }
    let bytes = fs::read(&args[2])?;
    match args[1].as_str() {
        "decode" => {
            fs::write(&args[3], codec::decode(&bytes)?.rgba)?;
        }
        "shift" => {
            let header = stream::header(&bytes)?;
            let network = stream::neural_network(&header)?;
            let grids = codec::decode_latents(&network, &header.descriptors, header.latent)?;
            let grids: Vec<_> = grids
                .iter()
                .map(|g| {
                    g.data
                        .iter()
                        .map(|&v| (v as i32 + 1).clamp(-64, 63))
                        .collect()
                })
                .collect();
            fs::write(
                &args[3],
                codec::encode(&network, &header.descriptors, &grids)?,
            )?;
        }
        _ => return Err("Unknown reference command".into()),
    }
    Ok(())
}
