//! Packs the encoder AND decoder tensors; upstream's demo bundles are decoder-only.
use std::{env, fs, path::PathBuf};
use zenjpegai::{
    header::OperatingPoint,
    model::{self, ModelBundle},
    nn::fast::{Engine, Tier},
    weights::packed::Recorder,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: pack-models <checkpoint-dir> <model-id> <notice-file> <output.zjb> <sop|bop|hop>"
                .into(),
        );
    }
    let dir = PathBuf::from(&args[1]);
    let id: usize = args[2].parse()?;
    let op = match args[5].as_str() {
        "sop" => OperatingPoint::Sop,
        "bop" => OperatingPoint::Bop,
        "hop" => OperatingPoint::Hop,
        _ => return Err("operating point must be sop, bop or hop".into()),
    };
    // SOP reuses BOP analysis but has its own lightweight synthesis networks.
    let analysis_op = if op == OperatingPoint::Sop {
        OperatingPoint::Bop
    } else {
        op
    };
    let mut source = ModelBundle::new();
    for ccs in 0..2 {
        for path in [
            model::common_path(id, ccs)?,
            model::analysis_path(id, ccs, analysis_op)?,
            model::synthesis_path(id, ccs, op)?,
        ] {
            source.insert(path.clone(), fs::read(dir.join(path))?);
        }
    }
    let recorder = Recorder::new(source);
    let eng = Engine::with(Tier::Scalar, false);
    for ccs in 0..2 {
        model::load_common(&recorder, id, ccs, &eng)?;
        model::load_hyper_encoder(&recorder, id, ccs, &eng)?;
        // Encoder rate estimation loads the learned hyper-entropy density, which
        // the decoder's common loader deliberately does not access.
        model::with_checkpoint(&recorder, &model::common_path(id, ccs)?, |checkpoint| {
            for name in checkpoint
                .tensor_names()
                .filter(|name| name.starts_with("hyper_entropy."))
            {
                checkpoint.raw(name)?;
            }
            Ok(())
        })?;
    }
    model::load_analysis_primary(&recorder, id, analysis_op, &eng)?;
    model::load_analysis_secondary(&recorder, id, analysis_op, &eng)?;
    model::load_synthesis_primary(&recorder, id, op, &eng)?;
    model::load_synthesis_secondary(&recorder, id, op, &eng)?;
    let bytes = recorder.pack(&fs::read_to_string(&args[3])?)?;
    fs::write(&args[4], &bytes)?;
    println!("{op:?} model {id}: {} bytes", bytes.len());
    Ok(())
}
