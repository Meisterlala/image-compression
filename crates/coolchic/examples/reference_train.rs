//! Development-only invocation of the same bounded-step browser trainer.
use coolchic::reference::{
    architecture::OperatingPoint,
    training::{Settings, Trainer},
};
use std::{env, fs, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 8 {
        return Err("Usage: reference_train INPUT.rgba WIDTH HEIGHT ITERATIONS LAMBDA lop|mop|hop|vhop OUTPUT.cool".into());
    }
    let point = match args[6].as_str() {
        "lop" => OperatingPoint::Lop,
        "mop" => OperatingPoint::Mop,
        "hop" => OperatingPoint::Hop,
        "vhop" => OperatingPoint::Vhop,
        _ => return Err("Unknown operating point".into()),
    };
    let settings = Settings {
        iterations: args[4].parse()?,
        lambda: args[5].parse()?,
        point,
        ..Settings::default()
    };
    let start = Instant::now();
    let mut trainer = Trainer::new(
        &fs::read(&args[1])?,
        args[2].parse()?,
        args[3].parse()?,
        settings,
    )?;
    let mut logged = usize::MAX;
    loop {
        match trainer.step() {
            Ok(true) => break,
            Ok(false) => {}
            Err(error) => return Err(format!("{}: {error}", trainer.status()).into()),
        }
        if trainer.completed.is_multiple_of(100) && trainer.completed != logged {
            eprintln!("{}", trainer.status());
            logged = trainer.completed;
        }
    }
    let completed = trainer.completed;
    let bytes = trainer.finish()?;
    fs::write(&args[7], &bytes)?;
    eprintln!(
        "{completed} training iterations, {} bytes, {:.2}s",
        bytes.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
