# Contributing to devpipe

## Setup

Requires Rust 1.75+ and a `GROQ_API_KEY` (free at [console.groq.com](https://console.groq.com)).

```bash
git clone https://github.com/Amir-Zecharia/devpipe
cd devpipe
cargo build
```

> macOS 26+: if you hit a `cstdint` error, set `CXXFLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1"`

## Before submitting a PR

```bash
cargo fmt
cargo clippy -- -D warnings
cargo test
```

CI runs these on Ubuntu and macOS (Metal).

## Guidelines

- No dead code or unused dependencies
- Keep `README.md`, `SKILL.md`, and `Cargo.toml` description in sync with any command/flag changes
- Don't commit generated output files

## License

Contributions are licensed under [MIT](LICENSE).
