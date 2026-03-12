use anyhow::Result;
use std::path::PathBuf;

use crate::compress;
use crate::generate;

/// Pipeline mode: chain compress and generate steps.
#[allow(clippy::too_many_arguments)]
pub async fn run_pipe(
    mode: &str,
    input: &Option<PathBuf>,
    keep_ratio: f32,
    groq_model: &str,
    groq_max_tokens: u32,
    compress_model: &str,
    compress_model_file: &str,
    stats: bool,
) -> Result<()> {
    match mode {
        "compress-generate" => {
            // Step 1: compress the input text
            eprintln!("[pipe] Step 1: Compressing input...");
            let compressed = compress::run_compress(
                input,
                keep_ratio,
                compress_model,
                compress_model_file,
                stats,
                false, // auto
                None,  // target_tokens
                false, // perplexity
            )
            .await?;

            if compressed.is_empty() {
                eprintln!("[pipe] Compressed output is empty, nothing to generate.");
                return Ok(());
            }

            // Step 2: use compressed text as the prompt for spec generation
            eprintln!("[pipe] Step 2: Generating spec from compressed text...");
            let spec =
                generate::generate_spec_string(&compressed, groq_model, groq_max_tokens).await?;
            print!("{}", spec);

            Ok(())
        }
        "generate-compress" => {
            // Step 1: read input text as prompt, generate spec
            let prompt = crate::model::read_input(input)?;
            if prompt.is_empty() {
                eprintln!("[pipe] Input prompt is empty, nothing to generate.");
                return Ok(());
            }

            eprintln!("[pipe] Step 1: Generating spec...");
            let spec = generate::generate_spec_string(&prompt, groq_model, groq_max_tokens).await?;

            // Step 2: compress the generated spec
            eprintln!("[pipe] Step 2: Compressing generated spec...");

            // Write spec to a temp file so compress can read it
            let tmp_dir = std::env::temp_dir();
            let tmp_path = tmp_dir.join("devpipe_pipe_tmp.txt");
            std::fs::write(&tmp_path, &spec)?;

            let compressed = compress::run_compress(
                &Some(tmp_path.clone()),
                keep_ratio,
                compress_model,
                compress_model_file,
                stats,
                false,
                None,
                false,
            )
            .await?;

            // Clean up temp file
            let _ = std::fs::remove_file(&tmp_path);

            print!("{}", compressed);

            Ok(())
        }
        other => {
            anyhow::bail!(
                "Unknown pipe mode: '{}'. Valid modes: compress-generate, generate-compress",
                other
            );
        }
    }
}
