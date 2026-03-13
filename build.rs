use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;

fn main() {
    let raw = std::fs::read("src/tokenizer.json").expect("src/tokenizer.json not found");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest = std::path::Path::new(&out_dir).join("tokenizer.json.gz");

    let file = std::fs::File::create(&dest).expect("Failed to create compressed tokenizer");
    let mut encoder = GzEncoder::new(file, Compression::best());
    encoder.write_all(&raw).expect("Failed to write compressed tokenizer");
    encoder.finish().expect("Failed to finish gzip encoding");

    println!("cargo:rerun-if-changed=src/tokenizer.json");
}
