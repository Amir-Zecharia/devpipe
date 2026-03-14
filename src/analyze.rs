use anyhow::Result;
use std::path::PathBuf;

use crate::compress;
use crate::model;

/// Run the analyze command: compute surprisals and display a color-coded heatmap.
pub async fn run_analyze(
    input: &Option<PathBuf>,
    model_repo: &str,
    model_file: &str,
    stats: bool,
    json_output: bool,
) -> Result<()> {
    let text = model::read_input(input)?;
    if text.is_empty() {
        return Ok(());
    }

    // Resolve model path
    let model_path = if model_repo.contains('/') {
        model::download_model(model_repo, model_file).await?
    } else {
        PathBuf::from(model_repo)
    };

    let device = model::select_device();
    let mut weights = model::load_model_weights(&model_path, &device)?;
    let tokenizer = model::load_tokenizer(&model_path)?;

    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let n = token_ids.len();

    if n == 0 {
        return Ok(());
    }

    // Compute surprisals
    let surprisals = compress::compute_surprisals(&mut weights, &token_ids, &device)?;

    // Compute percentile thresholds from finite surprisals
    let mut finite_surprisals: Vec<f32> = surprisals
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .collect();
    finite_surprisals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let (p25, p50, p75) = if finite_surprisals.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let len = finite_surprisals.len();
        (
            finite_surprisals[len / 4],
            finite_surprisals[len / 2],
            finite_surprisals[3 * len / 4],
        )
    };

    // Decode individual tokens
    let mut tokens_data: Vec<(String, f32, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        let token_text = tokenizer
            .decode(&[token_ids[i]], false)
            .unwrap_or_else(|_| format!("<token-{}>", token_ids[i]));
        let surprisal = surprisals[i];
        let percentile = if !surprisal.is_finite() || finite_surprisals.is_empty() {
            100.0 // position 0 (INFINITY) treated as max
        } else {
            let pos = finite_surprisals.partition_point(|&x| x < surprisal);
            (pos as f32 / finite_surprisals.len() as f32) * 100.0
        };
        tokens_data.push((token_text, surprisal, percentile));
    }

    if json_output {
        // JSON mode
        let json_tokens: Vec<serde_json::Value> = tokens_data
            .iter()
            .map(|(text, surprisal, percentile)| {
                serde_json::json!({
                    "text": text,
                    "surprisal": if surprisal.is_finite() { serde_json::json!(surprisal) } else { serde_json::json!("Infinity") },
                    "percentile": percentile
                })
            })
            .collect();
        let output = serde_json::json!({
            "tokens": json_tokens,
            "totalTokens": n,
            "thresholds": { "p25": p25, "p50": p50, "p75": p75 }
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        // Colored output mode
        for (text, surprisal, percentile) in &tokens_data {
            let color = if !surprisal.is_finite() {
                "\x1b[37m" // white for position 0
            } else if *percentile < 25.0 {
                "\x1b[90m" // gray - low information
            } else if *percentile < 50.0 {
                "\x1b[34m" // blue
            } else if *percentile < 75.0 {
                "\x1b[33m" // yellow
            } else {
                "\x1b[1;31m" // bright red - high information
            };
            print!("{}{}\x1b[0m", color, text);
        }
        println!(); // final newline
    }

    if stats {
        let finite: Vec<f32> = surprisals
            .iter()
            .skip(1)
            .copied()
            .filter(|s| s.is_finite())
            .collect();
        if !finite.is_empty() {
            let mean = finite.iter().sum::<f32>() / finite.len() as f32;
            let max = finite.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = finite.iter().cloned().fold(f32::INFINITY, f32::min);
            let median = {
                let mut sorted = finite.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                sorted[sorted.len() / 2]
            };
            eprintln!(
                "Tokens: {} | Mean surprisal: {:.4} | Median: {:.4} | Min: {:.4} | Max: {:.4}",
                n, mean, median, min, max
            );
            eprintln!(
                "Thresholds — P25: {:.4} | P50: {:.4} | P75: {:.4}",
                p25, p50, p75
            );
            // Simple histogram
            let buckets = [
                (
                    "  Gray (< P25)",
                    finite.iter().filter(|&&s| s < p25).count(),
                ),
                (
                    "  Blue (P25-P50)",
                    finite.iter().filter(|&&s| s >= p25 && s < p50).count(),
                ),
                (
                    "Yellow (P50-P75)",
                    finite.iter().filter(|&&s| s >= p50 && s < p75).count(),
                ),
                (
                    "   Red (>= P75)",
                    finite.iter().filter(|&&s| s >= p75).count(),
                ),
            ];
            eprintln!("Distribution:");
            let max_count = buckets.iter().map(|(_, c)| *c).max().unwrap_or(1);
            for (label, count) in &buckets {
                let bar_len = (*count as f32 / max_count as f32 * 30.0) as usize;
                let bar: String = "█".repeat(bar_len);
                eprintln!("  {} │ {} {}", label, bar, count);
            }
        }
    }

    Ok(())
}
