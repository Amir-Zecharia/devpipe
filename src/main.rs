mod analyze;
mod compress;
mod generate;
mod hook;
mod model;
mod pipe;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "devpipe",
    about = "Unified CLI: compress text with LLM surprisal scoring + generate specs via Groq API",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress text by removing low-information tokens using LLM log-probabilities
    Compress {
        /// Input file path (reads from stdin if not provided)
        input: Option<PathBuf>,

        /// Fraction of tokens to keep (0.0-1.0), ranked by surprisal
        #[arg(long, default_value_t = 0.7)]
        keep_ratio: f32,

        /// HuggingFace repo ID or local path to GGUF model
        #[arg(long, default_value = "bartowski/Llama-3.2-1B-Instruct-GGUF")]
        model: String,

        /// Specific GGUF filename within the HF repo
        #[arg(long, default_value = "Llama-3.2-1B-Instruct-Q4_K_M.gguf")]
        model_file: String,

        /// Print compression stats to stderr
        #[arg(long)]
        stats: bool,

        /// Automatically determine keep_ratio using elbow detection on surprisal scores
        #[arg(long)]
        auto: bool,

        /// Keep exactly this many output tokens (binary-searches for the right threshold)
        #[arg(long)]
        target_tokens: Option<usize>,

        /// Compute and print perplexity of the input text, then exit (no compression)
        #[arg(long)]
        perplexity: bool,

        /// Token budget: a number (e.g. 4096) or model name (e.g. gpt-4o) to look up context window
        #[arg(long)]
        budget: Option<String>,
    },

    /// Generate a comprehensive technical specification from a brief prompt using Groq
    Generate {
        /// A short description of the feature or tool to generate a spec for
        prompt: String,

        /// Output file path (prints to stdout if not provided)
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Groq model to use
        #[arg(long, short, default_value = generate::DEFAULT_GROQ_MODEL)]
        model: String,

        /// Maximum tokens for the generated response
        #[arg(long, default_value_t = generate::DEFAULT_MAX_TOKENS)]
        max_tokens: u32,
    },

    /// Chain compress and generate steps in a pipeline
    Pipe {
        /// Pipeline mode: compress-generate or generate-compress
        mode: String,

        /// Input file path (reads from stdin if not provided)
        input: Option<PathBuf>,

        /// Fraction of tokens to keep during compression
        #[arg(long, default_value_t = 0.7)]
        keep_ratio: f32,

        /// Groq model to use for generation
        #[arg(long, default_value = generate::DEFAULT_GROQ_MODEL)]
        groq_model: String,

        /// Maximum tokens for the generated response
        #[arg(long, default_value_t = generate::DEFAULT_MAX_TOKENS)]
        groq_max_tokens: u32,

        /// HuggingFace repo ID or local path to GGUF model for compression
        #[arg(long, default_value = "bartowski/Llama-3.2-1B-Instruct-GGUF")]
        compress_model: String,

        /// Specific GGUF filename within the HF repo for compression
        #[arg(long, default_value = "Llama-3.2-1B-Instruct-Q4_K_M.gguf")]
        compress_model_file: String,

        /// Print compression stats to stderr
        #[arg(long)]
        stats: bool,
    },

    /// Analyze text by showing a surprisal heatmap with color-coded tokens
    Analyze {
        /// Input file path (reads from stdin if not provided)
        input: Option<PathBuf>,
        /// HuggingFace repo ID or local path to GGUF model
        #[arg(long, default_value = "bartowski/Llama-3.2-1B-Instruct-GGUF")]
        model: String,
        /// Specific GGUF filename within the HF repo
        #[arg(long, default_value = "Llama-3.2-1B-Instruct-Q4_K_M.gguf")]
        model_file: String,
        /// Print statistics to stderr
        #[arg(long)]
        stats: bool,
        /// Output as JSON instead of colored text
        #[arg(long)]
        json: bool,
    },

    /// Print Claude Code UserPromptSubmit hook configuration JSON
    Hook {
        /// Keep ratio to use in the generated hook command
        #[arg(long, default_value_t = 0.8)]
        keep_ratio: f32,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress {
            input,
            keep_ratio,
            model,
            model_file,
            stats,
            auto,
            target_tokens,
            perplexity,
            budget,
        } => {
            // Resolve --budget to target_tokens
            let target_tokens = if let Some(ref b) = budget {
                Some(compress::resolve_budget(b)?)
            } else {
                target_tokens
            };
            let output = compress::run_compress(
                &input,
                keep_ratio,
                &model,
                &model_file,
                stats,
                auto,
                target_tokens,
                perplexity,
            )
            .await?;
            print!("{}", output);
        }

        Commands::Generate {
            prompt,
            output,
            model,
            max_tokens,
        } => {
            generate::run_generate(&prompt, &output, &model, max_tokens).await?;
        }

        Commands::Pipe {
            mode,
            input,
            keep_ratio,
            groq_model,
            groq_max_tokens,
            compress_model,
            compress_model_file,
            stats,
        } => {
            pipe::run_pipe(
                &mode,
                &input,
                keep_ratio,
                &groq_model,
                groq_max_tokens,
                &compress_model,
                &compress_model_file,
                stats,
            )
            .await?;
        }

        Commands::Analyze {
            input,
            model,
            model_file,
            stats,
            json,
        } => {
            analyze::run_analyze(&input, &model, &model_file, stats, json).await?;
        }

        Commands::Hook { keep_ratio } => {
            hook::run_hook(keep_ratio)?;
        }
    }

    Ok(())
}
