mod compress;
mod generate;
mod model;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "devpipe",
    about = "Unified CLI: compress text with LLM surprisal scoring + generate specs via Groq API",
    long_about = "Unified CLI: compress text with LLM surprisal scoring + generate specs via Groq API.\n\nExamples:\n  devpipe compress input.txt --stats\n  devpipe compress input.txt --auto\n  devpipe generate \"Payment service\" -o spec.md\n  devpipe generate-compress \"Auth system\" --keep-ratio 0.5",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress text by removing low-information tokens using LLM surprisal scoring
    #[command(long_about = "Compress text by removing low-information tokens using LLM surprisal scoring.\n\nScores each token by surprisal (how unexpected it is given context), then keeps\nonly the most informative ones.\n\nExamples:\n  devpipe compress input.txt --stats\n  devpipe compress input.txt --auto --stats\n  devpipe compress input.txt --target-tokens 200\n  cat input.txt | devpipe compress --keep-ratio 0.5")]
    Compress {
        /// Input text or file path (reads from stdin if not provided)
        input: Option<String>,

        /// Fraction of tokens to keep (0.0-1.0), ranked by surprisal
        #[arg(long, default_value_t = 0.7)]
        keep_ratio: f32,

        /// Print compression stats (token counts, ratio, distribution) to stderr
        #[arg(long)]
        stats: bool,

        /// Auto-detect optimal keep_ratio via elbow detection on the surprisal curve
        #[arg(long)]
        auto: bool,

        /// Keep exactly N output tokens (finds the right surprisal threshold automatically)
        #[arg(long)]
        target_tokens: Option<usize>,

        /// Use Groq API for compression instead of local model (faster, requires GROQ_API_KEY)
        #[arg(long)]
        groq: bool,
    },

    /// Generate a technical specification from a brief prompt using Groq
    #[command(long_about = "Generate a technical specification from a brief prompt using Groq.\n\nSends a structured prompt to the Groq API and returns a developer-ready spec\nwith architecture, data models, implementation plan, and edge cases.\n\nExamples:\n  devpipe generate \"Payment processing microservice\"\n  devpipe generate \"Auth system with OAuth2\" -o spec.md --json")]
    Generate {
        /// A short description of the feature or tool to generate a spec for
        prompt: String,

        /// Output file path (prints to stdout if not provided)
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Output as JSON with metadata (model, token count, prompt)
        #[arg(long)]
        json: bool,

        /// Print generation stats (model, tokens, timing) to stderr
        #[arg(long)]
        stats: bool,
    },

    /// Generate a spec from a prompt, then compress the output
    #[command(
        name = "generate-compress",
        long_about = "Generate a spec from a prompt, then compress the output.\n\nFirst generates a technical specification via the Groq API, then compresses\nit using LLM surprisal scoring to keep only the most informative tokens.\n\nExamples:\n  devpipe generate-compress \"Payment processing microservice\"\n  devpipe generate-compress \"Auth system\" --keep-ratio 0.5 --stats\n  devpipe generate-compress \"CLI tool\" --auto -o spec.md"
    )]
    GenerateCompress {
        /// A short description of the feature or tool to generate a spec for
        prompt: String,

        /// Fraction of tokens to keep (0.0-1.0), ranked by surprisal
        #[arg(long, default_value_t = 0.7)]
        keep_ratio: f32,

        /// Print stats (generation + compression) to stderr
        #[arg(long)]
        stats: bool,

        /// Auto-detect optimal keep_ratio via elbow detection
        #[arg(long)]
        auto: bool,

        /// Keep exactly N output tokens
        #[arg(long)]
        target_tokens: Option<usize>,

        /// Use Groq API for compression instead of local model (faster, requires GROQ_API_KEY)
        #[arg(long)]
        groq: bool,

        /// Output file path (prints to stdout if not provided)
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Output as JSON with metadata
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress {
            input,
            keep_ratio,
            stats,
            auto,
            target_tokens,
            groq,
        } => {
            let start = std::time::Instant::now();
            // Resolve input: file path, raw text, or stdin
            let (text, file_input) = match &input {
                Some(s) if std::path::Path::new(s).exists() => {
                    (None, Some(PathBuf::from(s)))
                }
                Some(s) => (Some(s.clone()), None),
                None => (None, None), // will read from stdin
            };
            let output = if groq {
                let text = text.unwrap_or_else(|| model::read_input(&file_input).unwrap());
                let in_tokens = text.split_whitespace().count();
                let result = generate::compress_via_groq(&text, keep_ratio).await?;
                if stats {
                    let out_tokens = result.split_whitespace().count();
                    let saved = in_tokens.saturating_sub(out_tokens);
                    let compression = (1.0 - (out_tokens as f32 / in_tokens as f32)) * 100.0;
                    eprintln!("Tokens in: ~{in_tokens} | Tokens out: ~{out_tokens} | Saved: ~{saved} | Compression: {compression:.1}%");
                }
                result
            } else if let Some(text) = text {
                // Raw text: write to temp file for local model
                let tmp_path = std::env::temp_dir().join("devpipe_compress_input.txt");
                std::fs::write(&tmp_path, &text)?;
                let tmp_input = Some(tmp_path.clone());
                let result = compress::run_compress(
                    &tmp_input,
                    keep_ratio,
                    "bartowski/Llama-3.2-1B-Instruct-GGUF",
                    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                    stats,
                    auto,
                    target_tokens,
                    false,
                )
                .await?;
                let _ = std::fs::remove_file(&tmp_path);
                result
            } else {
                compress::run_compress(
                    &file_input,
                    keep_ratio,
                    "bartowski/Llama-3.2-1B-Instruct-GGUF",
                    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                    stats,
                    auto,
                    target_tokens,
                    false,
                )
                .await?
            };
            print!("{}", output);
            if stats {
                eprintln!("Total time: {:.1}s", start.elapsed().as_secs_f64());
            }
        }

        Commands::Generate {
            prompt,
            output,
            json,
            stats,
        } => {
            let start = std::time::Instant::now();
            let spec = generate::generate_spec_string(
                &prompt,
                generate::DEFAULT_GROQ_MODEL,
                generate::DEFAULT_MAX_TOKENS,
            )
            .await?;
            let elapsed = start.elapsed();

            if json {
                let json_output = serde_json::json!({
                    "spec": spec,
                    "model": generate::DEFAULT_GROQ_MODEL,
                    "prompt": prompt,
                    "tokens": spec.split_whitespace().count(),
                    "elapsed_ms": elapsed.as_millis(),
                });
                let formatted = serde_json::to_string_pretty(&json_output)?;
                match output {
                    Some(path) => {
                        std::fs::write(&path, &formatted)?;
                        eprintln!("Written to {}", path.display());
                    }
                    None => print!("{}", formatted),
                }
            } else {
                match output {
                    Some(path) => {
                        std::fs::write(&path, &spec)?;
                        eprintln!("Written to {}", path.display());
                    }
                    None => print!("{}", spec),
                }
            }

            if stats {
                let token_count = spec.split_whitespace().count();
                eprintln!(
                    "Model: {} | Tokens: ~{} | Total time: {:.1}s",
                    generate::DEFAULT_GROQ_MODEL,
                    token_count,
                    elapsed.as_secs_f64()
                );
            }
        }

        Commands::GenerateCompress {
            prompt,
            keep_ratio,
            stats,
            auto,
            target_tokens,
            groq,
            output,
            json,
        } => {
            let total_start = std::time::Instant::now();
            // Step 1: Generate spec from prompt
            eprintln!("[1/2] Generating spec...");
            let gen_start = std::time::Instant::now();
            let spec = generate::generate_spec_string(
                &prompt,
                generate::DEFAULT_GROQ_MODEL,
                generate::DEFAULT_MAX_TOKENS,
            )
            .await?;
            let gen_elapsed = gen_start.elapsed();

            if stats {
                eprintln!(
                    "Generation: {} | ~{} tokens | {:.1}s",
                    generate::DEFAULT_GROQ_MODEL,
                    spec.split_whitespace().count(),
                    gen_elapsed.as_secs_f64()
                );
            }

            // Step 2: Compress the spec
            eprintln!("[2/2] Compressing spec...");
            let compress_start = std::time::Instant::now();
            let spec_tokens = spec.split_whitespace().count();
            let compressed = if groq {
                let result = generate::compress_via_groq(&spec, keep_ratio).await?;
                if stats {
                    let out_tokens = result.split_whitespace().count();
                    let saved = spec_tokens.saturating_sub(out_tokens);
                    let compression = (1.0 - (out_tokens as f32 / spec_tokens as f32)) * 100.0;
                    eprintln!("Tokens in: ~{spec_tokens} | Tokens out: ~{out_tokens} | Saved: ~{saved} | Compression: {compression:.1}%");
                }
                result
            } else {
                let tmp_dir = std::env::temp_dir();
                let tmp_path = tmp_dir.join("devpipe_gen_compress.txt");
                std::fs::write(&tmp_path, &spec)?;
                let tmp_input = Some(tmp_path.clone());
                let result = compress::run_compress(
                    &tmp_input,
                    keep_ratio,
                    "bartowski/Llama-3.2-1B-Instruct-GGUF",
                    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                    stats,
                    auto,
                    target_tokens,
                    false,
                )
                .await?;
                let _ = std::fs::remove_file(&tmp_path);
                result
            };

            if json {
                let json_output = serde_json::json!({
                    "compressed_spec": compressed,
                    "model": generate::DEFAULT_GROQ_MODEL,
                    "prompt": prompt,
                    "original_tokens": spec.split_whitespace().count(),
                    "compressed_tokens": compressed.split_whitespace().count(),
                    "generation_ms": gen_elapsed.as_millis(),
                    "compress_ms": compress_start.elapsed().as_millis(),
                });
                let formatted = serde_json::to_string_pretty(&json_output)?;
                match output {
                    Some(path) => {
                        std::fs::write(&path, &formatted)?;
                        eprintln!("Written to {}", path.display());
                    }
                    None => print!("{}", formatted),
                }
            } else {
                match output {
                    Some(path) => {
                        std::fs::write(&path, &compressed)?;
                        eprintln!("Written to {}", path.display());
                    }
                    None => print!("{}", compressed),
                }
            }
            if stats {
                let compress_elapsed = compress_start.elapsed();
                eprintln!(
                    "Compress time: {:.1}s | Generate time: {:.1}s | Total: {:.1}s",
                    compress_elapsed.as_secs_f64(),
                    gen_elapsed.as_secs_f64(),
                    total_start.elapsed().as_secs_f64()
                );
            }
        }
    }

    Ok(())
}
