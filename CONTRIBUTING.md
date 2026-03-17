# Contributing to devpipe

Thanks for your interest in contributing! Here's everything you need to get started.

## Prerequisites

- **Rust 1.75+** — install via [rustup](https://rustup.rs)
- **macOS with Apple Silicon** recommended for Metal GPU acceleration
- **`GROQ_API_KEY`** — for testing `generate` and `generate-compress` commands (free at [console.groq.com](https://console.groq.com))

> **macOS 26+ note:** If you hit a `cstdint` build error during compilation, set:
> ```bash
> export CXXFLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1"
> ```

## Getting Started

```bash
git clone https://github.com/Amir-Zecharia/devpipe
cd devpipe
cargo build
```

The first build will download a ~700 MB GGUF model from HuggingFace and cache it locally. Subsequent builds are fast.

## Development Workflow

Before submitting a PR, make sure all checks pass locally:

```bash
# Format
cargo fmt

# Lint (zero warnings)
cargo clippy -- -D warnings

# Test
cargo test

# Full release build
cargo build --release
```

CI runs all four steps on both Ubuntu and macOS (with Metal).

## Project Structure

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI entry point — subcommand definitions and dispatch |
| `src/compress.rs` | Surprisal scoring, token filtering, elbow detection |
| `src/generate.rs` | Groq API client and spec generation |
| `src/model.rs` | GGUF model downloading and loading via HuggingFace Hub |
| `build.rs` | Compresses `src/tokenizer.json` into the binary at build time |
| `tests/cli_tests.rs` | Integration tests for CLI commands |

## Adding a New Command

1. Add a new variant to the `Commands` enum in `src/main.rs`
2. Handle it in the `match cli.command` block
3. Add any new logic in a new or existing module under `src/`
4. Update `README.md` and `SKILL.md` to reflect the new command
5. Add CLI integration tests in `tests/cli_tests.rs`

## Guidelines

- **No dead code** — if a function isn't used, remove it or gate it with `#[cfg(test)]`
- **No unused dependencies** — remove any crate from `Cargo.toml` that isn't imported
- **Docs must match the code** — update `README.md`, `SKILL.md`, and `Cargo.toml` description when commands or flags change
- **Keep `src/tokenizer.json` tracked** — it's required by `build.rs` and must stay in the repo
- **Don't commit generated artifacts** — files like `spec.md` (tool output samples) should be gitignored

## Submitting a PR

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run the full check suite (see above)
4. Open a PR — CI will run automatically on Ubuntu and macOS

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
