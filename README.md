# devpipe

[![CI](https://github.com/Amir-Zecharia/devpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/Amir-Zecharia/devpipe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LLM-powered text compression and spec generation from the command line.

## Features

- **Surprisal-based compression** — remove low-information tokens using a local quantized Llama model
- **Spec generation** — produce detailed technical specifications via the Groq API (Llama 3.3 70B)
- **Pipelines** — chain compression and generation in either order
- **Auto-tuning** — elbow detection for optimal compression ratio, or target a specific token count
- **Perplexity scoring** — measure input text perplexity without compressing
- **Claude Code integration** — generate hook configs for automatic prompt compression
- **Fully local compression** — no API calls needed; model is downloaded and cached on first run

## Prerequisites

- **Rust 1.75+** (`rustup` recommended)
- **macOS with Apple Silicon** recommended (Metal GPU acceleration, no extra setup)
- **Internet access** on first run (downloads ~700 MB GGUF model from HuggingFace, cached locally)
- **`GROQ_API_KEY`** for the `generate` and `pipe` commands (free at [console.groq.com](https://console.groq.com))

## Installation

### From source

```bash
git clone https://github.com/Amir-Zecharia/devpipe
cd devpipe
cargo install --path .
```

### From crates.io (planned)

```bash
cargo install devpipe
```

> **macOS 26+ note:** If you hit a `cstdint` build error, set:
> ```bash
> export CXXFLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1"
> ```

## Usage

### Compress text

```bash
# Compress a file, keeping 70% of tokens (default)
devpipe compress input.txt --stats

# Compress from stdin with a custom ratio
cat input.txt | devpipe compress --keep-ratio 0.5

# Auto-detect optimal ratio via elbow detection
devpipe compress input.txt --auto --stats

# Keep exactly 200 output tokens
devpipe compress input.txt --target-tokens 200

# Compute perplexity without compressing
devpipe compress input.txt --perplexity

# Use a different model
devpipe compress input.txt \
  --model bartowski/Llama-3.2-3B-Instruct-GGUF \
  --model-file Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

### Generate a technical spec

```bash
export GROQ_API_KEY=your_key_here

# Generate to stdout
devpipe generate "Payment processing microservice"

# Generate to file
devpipe generate "A real-time collaborative code editor" -o spec.md

# Use a different Groq model
devpipe generate "Auth system with OAuth2" -m llama-3.3-70b-versatile
```

### Pipeline (chain operations)

```bash
# Compress large context, then generate a spec from it
devpipe pipe compress-generate large_dump.txt --keep-ratio 0.5

# Generate a spec, then compress the output
devpipe pipe generate-compress prompt.txt --keep-ratio 0.8
```

### Generate a Claude Code hook config

```bash
devpipe hook --keep-ratio 0.8
```

Prints a JSON config you can merge into `~/.claude/settings.json` to auto-compress prompts before they reach the model.

## How It Works

**Compression** uses *surprisal scoring*: each token is assigned a score representing how unlikely it was given the preceding context. High-surprisal tokens carry novel information; low-surprisal tokens are predictable filler. `devpipe compress` runs a quantized Llama model locally via `llama.cpp`, scores every token, and discards the bottom fraction — keeping only the most informative ones.

**Spec generation** sends a structured prompt to the Groq API (Llama 3.3 70B) and returns a comprehensive, developer-ready technical specification.

**Pipelines** chain both steps: compress a large codebase dump then generate a spec from the compressed context, or generate a spec then compress it for downstream consumption.

## Architecture

| Module | Purpose |
|---|---|
| `main.rs` | CLI entry point (clap derive) and subcommand dispatch |
| `compress.rs` | Surprisal scoring, token filtering, elbow detection, perplexity |
| `generate.rs` | Groq API client and structured prompt for spec generation |
| `pipe.rs` | Pipeline orchestration (compress-generate, generate-compress) |
| `hook.rs` | Claude Code hook config JSON generation |
| `model.rs` | GGUF model downloading and loading via HuggingFace Hub |

## CLI Reference

### `compress` options

| Flag | Default | Description |
|---|---|---|
| `[input]` | stdin | Input file path |
| `--keep-ratio` | `0.7` | Fraction of tokens to keep (0.0-1.0) |
| `--model` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | HuggingFace repo ID |
| `--model-file` | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | GGUF filename |
| `--stats` | off | Print token counts and compression ratio to stderr |
| `--auto` | off | Auto-detect keep ratio via elbow detection |
| `--target-tokens` | - | Keep exactly this many output tokens |
| `--perplexity` | off | Compute and print perplexity, then exit |

### `generate` options

| Flag | Default | Description |
|---|---|---|
| `<prompt>` | required | Short description for the spec |
| `-o, --output` | stdout | Output file path |
| `-m, --model` | `llama-3.3-70b-versatile` | Groq model |
| `--max-tokens` | `8192` | Max response tokens |

### `pipe` options

| Flag | Default | Description |
|---|---|---|
| `<mode>` | required | `compress-generate` or `generate-compress` |
| `[input]` | stdin | Input file path |
| `--keep-ratio` | `0.7` | Compression keep ratio |
| `--groq-model` | `llama-3.3-70b-versatile` | Groq model for generation |
| `--groq-max-tokens` | `8192` | Max response tokens for generation |
| `--compress-model` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | GGUF model for compression |
| `--compress-model-file` | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | GGUF filename for compression |
| `--stats` | off | Print compression stats to stderr |

### `hook` options

| Flag | Default | Description |
|---|---|---|
| `--keep-ratio` | `0.8` | Keep ratio for the generated hook command |

## Contributing

Contributions welcome. Please open an issue first to discuss changes.

1. Fork the repository
2. Create your branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Push and open a Pull Request

## License

MIT — see [LICENSE](LICENSE).
