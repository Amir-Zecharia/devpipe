# devpipe

Unified developer productivity CLI that compresses text by removing low-information tokens (LLM surprisal scoring) and generates technical specifications via Groq API. Can chain both operations in a pipeline.

## Trigger
Use this skill when the user asks to:
- Compress text or code to reduce token count
- Remove low-information tokens from text
- Generate a technical specification from a prompt
- Compress then generate a spec (or vice versa)
- Show surprisal heatmap for text
- Calculate perplexity of text

## Instructions

### Compress text

```bash
# From file
devpipe compress path/to/file.txt --keep-ratio 0.7 --stats

# From stdin
echo "long text" | devpipe compress --keep-ratio 0.7

# Auto-detect optimal keep ratio
devpipe compress input.txt --auto --stats

# Keep exact number of tokens
devpipe compress input.txt --target-tokens 200

# Compute perplexity
devpipe compress input.txt --perplexity
```

### Generate a technical spec

Requires `GROQ_API_KEY` environment variable.

```bash
devpipe generate "A real-time collaborative editor" -o spec.md
```

### Pipeline (chain operations)

```bash
# Compress input, then use compressed text as spec prompt
devpipe pipe compress-generate input.txt --keep-ratio 0.7

# Generate spec from prompt, then compress the output
devpipe pipe generate-compress prompt.txt --keep-ratio 0.8

# Same workflows using cross-command flags (no pipe subcommand needed)
devpipe compress input.txt --generate
devpipe generate "Payment processing microservice" --compress
devpipe generate "Payment processing microservice" --compress --keep-ratio 0.5
```

### Surprisal heatmap

```bash
devpipe compress input.txt --heatmap --stats
devpipe compress input.txt --heatmap --json
```

### Generate Claude Code hook config

```bash
devpipe compress --emit-hook --keep-ratio 0.8
```

Paste the output JSON into `~/.claude/settings.json` to auto-compress prompts.

### Installation

If `devpipe` is not installed:
```bash
cd ~/devpipe && cargo install --path .
```

## Notes
- Compression uses a local quantized Llama model via candle (Metal GPU on macOS).
- First run downloads a ~700 MB GGUF model from HuggingFace (cached locally).
- Spec generation uses Groq API (free tier, no credit card needed).
- On macOS 26+, you may need: `export CXXFLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1"`
