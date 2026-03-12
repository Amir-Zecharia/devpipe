---
name: devpipe
description: >
  Unified CLI for compressing text using LLM surprisal scoring and generating
  technical specifications via Groq API. Supports pipeline chaining and Claude Code hooks.
  TRIGGER when: user mentions "compress text", "reduce tokens", "remove low-information tokens",
  "generate spec", "technical specification", "devpipe", "compress and generate",
  "token compression", "surprisal", or asks to compress prompts before sending to an LLM.
version: 1.0.0
allowed-tools: Bash(*) Read Write Edit Glob Grep
---

# devpipe

Unified developer productivity CLI that compresses text by removing low-information tokens
(LLM surprisal scoring) and generates technical specifications via Groq API.

## Capabilities

### 1. Text Compression (`devpipe compress`)

Remove predictable (low-surprisal) tokens from text to reduce token count while preserving meaning.

```bash
# Compress a file, keeping 70% of tokens
devpipe compress input.txt --keep-ratio 0.7 --stats

# Compress from stdin
cat large_file.txt | devpipe compress --keep-ratio 0.5

# Auto-detect optimal ratio via elbow detection
devpipe compress input.txt --auto --stats

# Target exact output token count
devpipe compress input.txt --target-tokens 500

# Compute perplexity score
devpipe compress input.txt --perplexity
```

### 2. Spec Generation (`devpipe generate`)

Generate a comprehensive, developer-ready technical specification from a brief prompt.

```bash
export GROQ_API_KEY=your_key_here
devpipe generate "A real-time collaborative code editor" -o spec.md
devpipe generate "Payment processing microservice" --model llama-3.3-70b-versatile
```

### 3. Pipeline Chaining (`devpipe pipe`)

Chain compress and generate operations together.

```bash
# Compress large context, then generate spec from compressed text
devpipe pipe compress-generate large_codebase_dump.txt --keep-ratio 0.5

# Generate spec, then compress the output to save tokens
devpipe pipe generate-compress prompt.txt --keep-ratio 0.8
```

### 4. Claude Code Hook (`devpipe hook`)

Generate hook config for automatic prompt compression in Claude Code.

```bash
devpipe hook --keep-ratio 0.8
# Paste output into ~/.claude/settings.json
```

## Installation

```bash
cd ~/devpipe && cargo install --path .
```

## Prerequisites

- Rust 1.75+ (for building)
- `GROQ_API_KEY` env var (for `generate` and `pipe` commands)
- Internet access on first run (downloads ~700 MB GGUF model, cached locally)
- macOS with Apple Silicon recommended (Metal GPU acceleration)
