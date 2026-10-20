//! Small native interoperability harness; no image I/O dependencies.
//! encode <input.rgba> <width> <height> <quality> <effort> <output.cool>
//! decode <input.cool> <output.rgba>
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("encode") if a.len() == 8 => {
            let pixels = std::fs::read(&a[2])?;
            let bytes = coolchic::encode(&pixels, a[3].parse()?, a[4].parse()?, a[5].parse()?, a[6].parse()?)?;
            std::fs::write(&a[7], bytes)?;
        }
        Some("decode") if a.len() == 4 => {
            let image = coolchic::decode(&std::fs::read(&a[2])?)?;
            println!("{} {}", image.width, image.height);
            std::fs::write(&a[3], image.rgba)?;
        }
        _ => return Err("Usage: rgba encode <input.rgba> <width> <height> <quality> <effort> <output.cool> | rgba decode <input.cool> <output.rgba>".into()),
    }
    Ok(())
}
