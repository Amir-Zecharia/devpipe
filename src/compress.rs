use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::path::PathBuf;

use crate::model;

/// Score each token by its surprisal (-log_prob) using the model's next-token predictions.
pub fn compute_surprisals(
    model: &mut ModelWeights,
    token_ids: &[u32],
    device: &Device,
) -> Result<Vec<f32>> {
    let n = token_ids.len();
    let mut surprisals = vec![f32::INFINITY; n]; // token 0 always kept

    if n <= 1 {
        return Ok(surprisals);
    }

    // Process tokens one at a time for autoregressive scoring.
    // candle's ModelWeights maintains an internal KV-cache: each forward()
    // call appends new K/V vectors and attends over all prior positions,
    // avoiding redundant recomputation.
    for i in 0..n - 1 {
        let input = Tensor::new(&[token_ids[i]], device)
            .context("Failed to create input tensor")?
            .unsqueeze(0)?;

        let logits = model
            .forward(&input, i)
            .context("Model forward pass failed")?;

        // logits shape: (1, vocab_size) — flatten to 1D
        let logits = logits.flatten_all()?;
        let log_probs = candle_nn::ops::log_softmax(&logits, 0)
            .context("Failed to compute log-softmax")?;
        let log_probs_vec: Vec<f32> = log_probs.to_vec1()?;

        let next_tok = token_ids[i + 1] as usize;
        if next_tok >= log_probs_vec.len() {
            anyhow::bail!(
                "Token ID {} at position {} exceeds vocabulary size {} -- tokenizer/model mismatch?",
                next_tok, i + 1, log_probs_vec.len()
            );
        }
        surprisals[i + 1] = -log_probs_vec[next_tok];
    }

    Ok(surprisals)
}

/// Select which token indices to keep based on surprisal scores and a keep ratio.
pub fn select_tokens(surprisals: &[f32], keep_ratio: f32) -> Vec<usize> {
    let n = surprisals.len();
    if keep_ratio >= 1.0 {
        return (0..n).collect();
    }
    if keep_ratio <= 0.0 {
        return vec![0];
    }

    let keep_count = ((n as f32) * keep_ratio).ceil() as usize;
    let keep_count = keep_count.clamp(1, n);

    // Sort indices by surprisal descending, keep top keep_count
    let mut indexed: Vec<(usize, f32)> = surprisals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<usize> = indexed.iter().take(keep_count).map(|(i, _)| *i).collect();
    kept.sort_unstable(); // restore original order
    kept
}

/// Auto mode: find the natural elbow in the sorted surprisal distribution.
///
/// Algorithm: sort finite surprisals ascending, compute the discrete second derivative
/// (rate of change of the slope), and pick the index where it peaks -- that's the point
/// where values start rising steeply, i.e. the boundary between redundant and informative.
/// Returns a keep_ratio in (0, 1].
pub fn elbow_keep_ratio(surprisals: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = surprisals
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .collect();
    if sorted.len() < 3 {
        return 1.0;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();

    // Second derivative: d2[i] = sorted[i+1] - 2*sorted[i] + sorted[i-1]
    // Peak in d2 = sharpest upward bend = elbow
    let mut max_d2 = f32::NEG_INFINITY;
    let mut elbow_idx = n / 2; // fallback to midpoint
    for i in 1..n - 1 {
        let d2 = sorted[i + 1] - 2.0 * sorted[i] + sorted[i - 1];
        if d2 > max_d2 {
            max_d2 = d2;
            elbow_idx = i;
        }
    }

    // Tokens with surprisal >= sorted[elbow_idx] are "informative" -- keep them.
    let threshold = sorted[elbow_idx];
    let keep_count = surprisals.iter().filter(|&&s| s >= threshold).count();
    (keep_count as f32 / surprisals.len() as f32).clamp(0.3, 1.0)
}

/// Target-tokens mode: binary-search for the surprisal threshold that yields exactly
/// `target` output tokens (or as close as possible without going over).
pub fn target_tokens_keep_ratio(surprisals: &[f32], target: usize) -> f32 {
    let n = surprisals.len();
    let target = target.clamp(1, n);

    let mut sorted: Vec<f32> = surprisals
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    if sorted.is_empty() {
        return 1.0;
    }

    // Binary search: find the lowest threshold such that tokens kept >= target
    let mut lo = 0usize;
    let mut hi = sorted.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let threshold = sorted[mid];
        let kept = surprisals.iter().filter(|&&s| s >= threshold).count();
        if kept >= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    let threshold = if lo == 0 {
        sorted[0]
    } else {
        sorted[lo.min(sorted.len()) - 1]
    };
    let keep_count = surprisals.iter().filter(|&&s| s >= threshold).count();
    (keep_count as f32 / n as f32).clamp(0.01, 1.0)
}

/// Resolve a --budget value to a token count.
/// Accepts a plain number (e.g. "4096") or a model name (e.g. "gpt-4o").
pub fn resolve_budget(budget: &str) -> Result<usize> {
    if let Ok(n) = budget.parse::<usize>() {
        return Ok(n);
    }
    let tokens = match budget.to_lowercase().as_str() {
        "gpt-4o" | "gpt-4o-mini" => 128_000,
        "gpt-4" | "gpt-4-turbo" => 128_000,
        "gpt-3.5" | "gpt-3.5-turbo" => 16_385,
        "claude-3-opus" | "claude-3-sonnet" | "claude-3-haiku" => 200_000,
        "claude-3.5-sonnet" | "claude-3.5-haiku" => 200_000,
        "claude-opus-4" | "claude-sonnet-4" => 200_000,
        "llama-3-8b" | "llama-3-70b" => 8_192,
        "llama-3.1-8b" | "llama-3.1-70b" | "llama-3.1-405b" => 128_000,
        "llama-3.2-1b" | "llama-3.2-3b" => 128_000,
        "mistral-7b" | "mistral" => 32_768,
        "mixtral" | "mixtral-8x7b" => 32_768,
        "gemini-pro" | "gemini-1.5-pro" => 1_000_000,
        "gemini-1.5-flash" => 1_000_000,
        "deepseek-v2" | "deepseek-coder" => 128_000,
        "phi-3" | "phi-3-mini" => 128_000,
        "command-r" => 128_000,
        "command-r-plus" => 128_000,
        _ => anyhow::bail!(
            "Unknown model '{}'. Use a number (e.g. --budget 4096) or a known model name (gpt-4o, claude-3-opus, llama-3.1-8b, etc.)",
            budget
        ),
    };
    Ok(tokens)
}

/// Core sync compress logic: given a resolved model path, load model, tokenize, score, select, decode.
#[allow(clippy::too_many_arguments)]
fn run_compress_inner(
    input: &Option<PathBuf>,
    keep_ratio: f32,
    model_path: &std::path::Path,
    stats: bool,
    auto: bool,
    target_tokens: Option<usize>,
    perplexity: bool,
) -> Result<String> {
    let text = model::read_input(input)?;

    if text.is_empty() {
        return Ok(String::new());
    }

    let device = model::select_device();

    // Load model weights
    let mut weights = model::load_model_weights(model_path, &device)?;

    // Load tokenizer
    let tokenizer = model::load_tokenizer(model_path)?;

    // Tokenize input
    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let n = token_ids.len();

    if n == 0 {
        return Ok(String::new());
    }

    // Short-circuit if keeping everything (skip when model is needed for scoring)
    let needs_model = perplexity || auto || target_tokens.is_some();
    if keep_ratio >= 1.0 && !needs_model {
        if stats {
            eprintln!("Tokens in: {n} | Tokens out: {n} | Saved: 0 | Compression: 0.0%");
        }
        return Ok(text);
    }

    // Compute surprisals
    let surprisals = compute_surprisals(&mut weights, &token_ids, &device)?;

    // Perplexity mode: exp(mean surprisal over tokens 1..N)
    if perplexity {
        let scored: Vec<f32> = surprisals
            .iter()
            .skip(1)
            .copied()
            .filter(|s| s.is_finite())
            .collect();
        if scored.is_empty() {
            eprintln!("Perplexity: N/A (too few tokens)");
        } else {
            let mean_surprisal = scored.iter().sum::<f32>() / scored.len() as f32;
            let ppl = mean_surprisal.exp();
            if stats {
                eprintln!("Tokens scored: {} | Mean surprisal: {mean_surprisal:.4} | Perplexity: {ppl:.4}", scored.len());
            }
            return Ok(format!("{ppl:.4}"));
        }
        return Ok(String::new());
    }

    // Determine keep ratio from mode flags (target_tokens > auto > explicit keep_ratio)
    let effective_ratio = if let Some(target) = target_tokens {
        let ratio = target_tokens_keep_ratio(&surprisals, target);
        if stats {
            eprintln!("Auto ratio (target {target} tokens): {ratio:.3}");
        }
        ratio
    } else if auto {
        let ratio = elbow_keep_ratio(&surprisals);
        if stats {
            eprintln!("Auto ratio (elbow detection): {ratio:.3}");
        }
        ratio
    } else {
        keep_ratio
    };

    // Select tokens to keep
    let kept_indices = select_tokens(&surprisals, effective_ratio);
    let k = kept_indices.len();

    // Decode kept tokens
    let kept_ids: Vec<u32> = kept_indices.iter().map(|&i| token_ids[i]).collect();
    let output = tokenizer
        .decode(&kept_ids, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

    if stats {
        let saved = n - k;
        let compression = (1.0 - (k as f32 / n as f32)) * 100.0;
        eprintln!(
            "Tokens in: {n} | Tokens out: {k} | Saved: {saved} | Compression: {compression:.1}%"
        );
    }

    Ok(output)
}

/// Main compress orchestration: resolve model (async), then delegate to sync inner function.
#[allow(clippy::too_many_arguments)]
pub async fn run_compress(
    input: &Option<PathBuf>,
    keep_ratio: f32,
    model_repo: &str,
    model_file: &str,
    stats: bool,
    auto: bool,
    target_tokens: Option<usize>,
    perplexity: bool,
) -> Result<String> {
    // Resolve model path (only async part)
    let model_path = if model_repo.contains('/') {
        model::download_model(model_repo, model_file).await?
    } else {
        PathBuf::from(model_repo)
    };

    run_compress_inner(input, keep_ratio, &model_path, stats, auto, target_tokens, perplexity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_tokens_keep_all() {
        let surprisals = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(select_tokens(&surprisals, 1.0), vec![0, 1, 2, 3]);
    }

    #[test]
    fn select_tokens_keep_none_returns_first() {
        let surprisals = vec![1.0, 2.0, 3.0];
        // keep_ratio <= 0 should return just index 0 (highest surprisal kept)
        let result = select_tokens(&surprisals, 0.0);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn select_tokens_keeps_highest_surprisal() {
        // Token at index 2 has highest surprisal, should be kept first
        let surprisals = vec![1.0, 3.0, 10.0, 2.0];
        let result = select_tokens(&surprisals, 0.25); // keep 1 token
        assert!(result.contains(&2));
    }

    #[test]
    fn select_tokens_preserves_order() {
        let surprisals = vec![1.0, 5.0, 2.0, 4.0];
        let result = select_tokens(&surprisals, 0.5); // keep 2
        assert!(result.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn select_tokens_single() {
        let surprisals = vec![5.0];
        assert_eq!(select_tokens(&surprisals, 0.5), vec![0]);
    }

    #[test]
    fn elbow_keep_ratio_too_few_tokens() {
        assert_eq!(elbow_keep_ratio(&[1.0, 2.0]), 1.0);
        assert_eq!(elbow_keep_ratio(&[]), 1.0);
    }

    #[test]
    fn elbow_keep_ratio_returns_valid_range() {
        let surprisals = vec![0.1, 0.2, 0.3, 0.5, 1.0, 5.0, 10.0, 50.0];
        let ratio = elbow_keep_ratio(&surprisals);
        assert!(ratio > 0.0 && ratio <= 1.0, "ratio was {ratio}");
    }

    #[test]
    fn elbow_keep_ratio_filters_infinity() {
        let surprisals = vec![f32::INFINITY, 0.5, 1.0, 2.0, 5.0];
        let ratio = elbow_keep_ratio(&surprisals);
        assert!(ratio > 0.0 && ratio <= 1.0);
    }

    #[test]
    fn target_tokens_keep_ratio_clamps() {
        let surprisals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Requesting more tokens than available should return ~1.0
        let ratio = target_tokens_keep_ratio(&surprisals, 100);
        assert!((ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn target_tokens_keep_ratio_single() {
        let surprisals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ratio = target_tokens_keep_ratio(&surprisals, 1);
        assert!(ratio > 0.0 && ratio <= 1.0);
        let kept = select_tokens(&surprisals, ratio);
        assert!(!kept.is_empty());
    }

    #[test]
    fn target_tokens_all_infinite() {
        let surprisals = vec![f32::INFINITY, f32::INFINITY];
        // All infinite filtered out, sorted is empty → returns 1.0
        assert_eq!(target_tokens_keep_ratio(&surprisals, 1), 1.0);
    }

    #[test]
    fn select_tokens_known_inputs_outputs() {
        // 5 tokens, keep 60% = ceil(3) = 3 tokens; highest surprisals at indices 1,3,4
        let surprisals = vec![1.0, 8.0, 2.0, 7.0, 9.0];
        let kept = select_tokens(&surprisals, 0.6);
        assert_eq!(kept, vec![1, 3, 4]);
    }

    #[test]
    fn resolve_budget_number() {
        assert_eq!(resolve_budget("4096").unwrap(), 4096);
        assert_eq!(resolve_budget("128000").unwrap(), 128000);
    }

    #[test]
    fn resolve_budget_model_name() {
        assert_eq!(resolve_budget("gpt-4o").unwrap(), 128_000);
        assert_eq!(resolve_budget("claude-3-opus").unwrap(), 200_000);
        assert_eq!(resolve_budget("llama-3.1-8b").unwrap(), 128_000);
    }

    #[test]
    fn resolve_budget_unknown_model() {
        assert!(resolve_budget("unknown-model-xyz").is_err());
    }

    #[test]
    fn resolve_budget_case_insensitive() {
        assert_eq!(resolve_budget("GPT-4O").unwrap(), 128_000);
        assert_eq!(resolve_budget("Claude-3-Opus").unwrap(), 200_000);
    }

    #[test]
    fn select_tokens_half_preserves_order() {
        let surprisals = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let kept = select_tokens(&surprisals, 0.5); // keep 4
                                                    // Must be sorted ascending (original order preserved)
        for w in kept.windows(2) {
            assert!(w[0] < w[1], "Order not preserved: {:?}", kept);
        }
        // The 4 highest surprisals are at indices 4(5.0), 5(9.0), 7(6.0), 2(4.0)
        assert_eq!(kept, vec![2, 4, 5, 7]);
    }

    #[test]
    fn elbow_keep_ratio_simple_distribution() {
        // Low plateau then sharp rise: elbow should detect the jump
        let surprisals = vec![0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 5.0, 10.0, 50.0];
        let ratio = elbow_keep_ratio(&surprisals);
        assert!(ratio > 0.0 && ratio <= 1.0, "ratio was {ratio}");
        // The high-surprisal tokens (informative) should be kept
        assert!(ratio < 1.0, "should not keep everything");
    }

    #[test]
    fn elbow_keep_ratio_uniform_distribution() {
        // Uniform: no clear elbow, should still return valid ratio
        let surprisals = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let ratio = elbow_keep_ratio(&surprisals);
        assert!(ratio > 0.0 && ratio <= 1.0);
    }
}
