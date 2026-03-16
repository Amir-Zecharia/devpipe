# devpipe

[![CI](https://github.com/Amir-Zecharia/devpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/Amir-Zecharia/devpipe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LLM-powered text compression and spec generation from the command line.

## Features

- **Surprisal-based compression** — remove low-information tokens using a local quantized Llama model or via Groq API
- **Spec generation** — produce detailed technical specifications via the Groq API
- **Generate + compress pipeline** — generate a spec then compress the output in a single command

## Prerequisites

- **Rust 1.75+** (`rustup` recommended)
- **macOS with Apple Silicon** recommended (Metal GPU acceleration, no extra setup)
- **Internet access** on first run (downloads ~700 MB GGUF model from HuggingFace, cached locally)
- **`GROQ_API_KEY`** for the `generate` and `generate-compress` commands (free at [console.groq.com](https://console.groq.com))

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

# Use Groq API for compression instead of local model
devpipe compress input.txt --groq --stats
```

### Generate a technical spec

```bash
export GROQ_API_KEY=your_key_here

# Generate to stdout
devpipe generate "Payment processing microservice"

# Generate to file
devpipe generate "A real-time collaborative code editor" -o spec.md

# Output as JSON with metadata
devpipe generate "Auth system with OAuth2" --json

# Print generation stats
devpipe generate "Payment service" --stats
```

### Generate then compress

```bash
# Generate a spec and compress the output
devpipe generate-compress "Payment processing microservice"

# With custom keep ratio and stats
devpipe generate-compress "Auth system" --keep-ratio 0.5 --stats

# Auto-detect optimal compression ratio
devpipe generate-compress "CLI tool" --auto

# Use Groq API for compression step
devpipe generate-compress "API gateway" --groq --stats

# Output to file as JSON
devpipe generate-compress "Auth system" -o spec.md --json
```

## How It Works

**Compression** uses *surprisal scoring*: each token is assigned a score representing how unlikely it was given the preceding context. High-surprisal tokens carry novel information; low-surprisal tokens are predictable filler. `devpipe compress` runs a quantized Llama model locally via `llama.cpp`, scores every token, and discards the bottom fraction — keeping only the most informative ones. Alternatively, use `--groq` to compress via the Groq API.

**Spec generation** sends a structured prompt to the Groq API (Llama 3.3 70B) and returns a comprehensive, developer-ready technical specification.

**Generate-compress** chains both steps: generate a spec from a prompt, then compress the output for downstream consumption.

## Architecture

| Module | Purpose |
|---|---|
| `main.rs` | CLI entry point (clap derive) and subcommand dispatch |
| `compress.rs` | Surprisal scoring, token filtering, elbow detection |
| `generate.rs` | Groq API client, spec generation, and Groq-based compression |
| `model.rs` | GGUF model downloading and loading via HuggingFace Hub |

## CLI Reference

### `compress` options

| Flag | Default | Description |
|---|---|---|
| `[input]` | stdin | Input text or file path |
| `--keep-ratio` | `0.7` | Fraction of tokens to keep (0.0-1.0) |
| `--stats` | off | Print token counts and compression ratio to stderr |
| `--auto` | off | Auto-detect keep ratio via elbow detection |
| `--target-tokens` | - | Keep exactly this many output tokens |
| `--groq` | off | Use Groq API for compression instead of local model |

### `generate` options

| Flag | Default | Description |
|---|---|---|
| `<prompt>` | required | Short description for the spec |
| `-o, --output` | stdout | Output file path |
| `--json` | off | Output as JSON with metadata |
| `--stats` | off | Print generation stats to stderr |

### `generate-compress` options

| Flag | Default | Description |
|---|---|---|
| `<prompt>` | required | Short description for the spec |
| `--keep-ratio` | `0.7` | Fraction of tokens to keep (0.0-1.0) |
| `--stats` | off | Print stats to stderr |
| `--auto` | off | Auto-detect keep ratio via elbow detection |
| `--target-tokens` | - | Keep exactly this many output tokens |
| `--groq` | off | Use Groq API for compression |
| `-o, --output` | stdout | Output file path |
| `--json` | off | Output as JSON with metadata |

## License

MIT — see [LICENSE](LICENSE).
