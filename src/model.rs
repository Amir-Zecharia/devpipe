use anyhow::{Context, Result};
use candle_core::{quantized::gguf_file, Device};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::io::Read;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const EMBEDDED_TOKENIZER_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tokenizer.json.gz"));

/// Decompress the gzip-compressed embedded tokenizer bytes.
fn decompress_tokenizer() -> Result<Vec<u8>> {
    let mut decoder = flate2::read::GzDecoder::new(EMBEDDED_TOKENIZER_GZ);
    let mut buf = Vec::new();
    decoder
        .read_to_end(&mut buf)
        .context("Failed to decompress embedded tokenizer")?;
    Ok(buf)
}

/// Select the best available compute device (Metal on macOS, CPU elsewhere).
pub fn select_device() -> Device {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).unwrap_or_else(|_| {
            eprintln!("Metal not available, falling back to CPU");
            Device::Cpu
        })
    }
    #[cfg(not(feature = "metal"))]
    {
        Device::Cpu
    }
}

/// Download a GGUF model file from a HuggingFace repo, returning its local cache path.
pub async fn download_model(repo_id: &str, filename: &str) -> Result<PathBuf> {
    use hf_hub::{api::tokio::Api, Repo, RepoType};
    let api = Api::new().context("Failed to create HuggingFace API client")?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
    let path = repo
        .get(filename)
        .await
        .with_context(|| format!("Failed to download {filename} from {repo_id}"))?;
    Ok(path)
}

/// Load quantized GGUF model weights from a local file.
pub fn load_model_weights(model_path: &Path, device: &Device) -> Result<ModelWeights> {
    let mut file = std::fs::File::open(model_path)
        .with_context(|| format!("Failed to open model file: {}", model_path.display()))?;
    let model = gguf_file::Content::read(&mut file).context("Failed to parse GGUF file")?;
    let weights = ModelWeights::from_gguf(model, &mut file, device)
        .context("Failed to load model weights")?;
    Ok(weights)
}

/// Load a tokenizer from the same directory as the model file, or fall back to embedded tokenizer.
pub fn load_tokenizer(model_path: &Path) -> Result<Tokenizer> {
    let dir = model_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let tok_path = dir.join("tokenizer.json");
    if tok_path.exists() {
        Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", tok_path, e))
    } else {
        let bytes = decompress_tokenizer()?;
        Tokenizer::from_bytes(&bytes)
            .map_err(|e| anyhow::anyhow!("Failed to load embedded tokenizer: {}", e))
    }
}

/// Read input text from a file path or stdin.
pub fn read_input(input: &Option<PathBuf>) -> Result<String> {
    match input {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display())),
        None => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .context("Failed to read from stdin")?;
            Ok(buf)
        }
    }
}
