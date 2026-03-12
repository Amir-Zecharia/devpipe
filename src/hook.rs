use anyhow::{Context, Result};

/// Print Claude Code UserPromptSubmit hook configuration JSON.
pub fn run_hook(keep_ratio: f32) -> Result<()> {
    let exe = std::env::current_exe().context("Failed to get current executable path")?;
    let exe_str = exe.to_string_lossy();
    let config = serde_json::json!({
        "hooks": {
            "UserPromptSubmit": [{
                "matcher": "",
                "hooks": [{
                    "type": "command",
                    "command": format!("{} compress --keep-ratio {} --stats", exe_str, keep_ratio)
                }]
            }]
        }
    });
    println!("{}", serde_json::to_string_pretty(&config)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    fn build_hook_config(keep_ratio: f32) -> serde_json::Value {
        let exe = std::env::current_exe().unwrap_or_default();
        let exe_str = exe.to_string_lossy();
        serde_json::json!({
            "hooks": {
                "UserPromptSubmit": [{
                    "matcher": "",
                    "hooks": [{
                        "type": "command",
                        "command": format!("{} compress --keep-ratio {} --stats", exe_str, keep_ratio)
                    }]
                }]
            }
        })
    }

    #[test]
    fn hook_config_has_correct_structure() {
        let config = build_hook_config(0.7);
        assert!(config["hooks"]["UserPromptSubmit"].is_array());
        let hooks = &config["hooks"]["UserPromptSubmit"][0]["hooks"];
        assert_eq!(hooks[0]["type"], "command");
    }

    #[test]
    fn hook_config_contains_keep_ratio() {
        let config = build_hook_config(0.42);
        let cmd = config["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap();
        assert!(cmd.contains("0.42"));
        assert!(cmd.contains("compress"));
    }

    #[test]
    fn hook_config_matcher_is_empty_string() {
        let config = build_hook_config(0.5);
        let matcher = config["hooks"]["UserPromptSubmit"][0]["matcher"]
            .as_str()
            .unwrap();
        assert_eq!(matcher, "");
    }

    #[test]
    fn hook_config_command_includes_stats_flag() {
        let config = build_hook_config(0.9);
        let cmd = config["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap();
        assert!(cmd.contains("--stats"));
    }

    #[test]
    fn hook_config_is_valid_json_roundtrip() {
        let config = build_hook_config(0.5);
        let serialized = serde_json::to_string(&config).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed, config);
    }
}
