use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;

pub const SYSTEM_PROMPT: &str = r#"You are a senior Product Manager and Software Architect with 15+ years of experience.
Your task is to transform a brief feature or product description into a comprehensive, developer-ready technical specification document.

The specification MUST be structured in Markdown and MUST include ALL of the following sections:

1. **Overview** — A clear, concise description of the feature/tool, its purpose, and its value to users.
2. **Core Requirements** — A numbered list of functional and non-functional requirements. Be specific and measurable.
3. **Technical Architecture** — The proposed system design, components, services, and their interactions. Include diagrams in Mermaid syntax where helpful.
4. **Data Models** — All relevant data structures, schemas, or database models. Use tables or code blocks to define fields, types, and constraints.
5. **Implementation Plan** — A phased, step-by-step breakdown of the work. Each phase should have a clear goal and list of tasks.
6. **Edge Cases & Considerations** — A thorough list of edge cases, error scenarios, security concerns, scalability considerations, and potential pitfalls.

Be thorough, precise, and actionable. A developer should be able to start implementation directly from this document."#;

pub const DEFAULT_GROQ_MODEL: &str = "llama-3.3-70b-versatile";
pub const DEFAULT_MAX_TOKENS: u32 = 8192;

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ChatMessage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

/// Get the config file path: ~/.config/devpipe/config.json
fn config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
        .join("devpipe")
        .join("config.json")
}

/// Load the API key from config file.
fn load_api_key() -> Option<String> {
    let path = config_path();
    let content = std::fs::read_to_string(&path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&content).ok()?;
    config["groq_api_key"].as_str().map(|s| s.to_string())
}

/// Save the API key to config file.
fn save_api_key(key: &str) -> Result<()> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create config directory")?;
    }
    let config = serde_json::json!({ "groq_api_key": key });
    std::fs::write(&path, serde_json::to_string_pretty(&config)?)
        .context("Failed to save config file")?;
    eprintln!("API key saved to {}", path.display());
    Ok(())
}

/// Get the Groq API key: env var > config file > prompt user.
fn get_api_key() -> Result<String> {
    // 1. Check environment variable
    if let Ok(key) = std::env::var("GROQ_API_KEY") {
        if !key.is_empty() {
            return Ok(key);
        }
    }

    // 2. Check config file
    if let Some(key) = load_api_key() {
        if !key.is_empty() {
            return Ok(key);
        }
    }

    // 3. Prompt the user
    eprintln!("No Groq API key found.");
    eprintln!("Get your free API key at: https://console.groq.com");
    eprintln!();
    eprint!("Enter your Groq API key: ");
    std::io::stderr().flush()?;

    let mut key = String::new();
    std::io::stdin().read_line(&mut key)?;
    let key = key.trim().to_string();

    if key.is_empty() {
        anyhow::bail!("No API key provided. Cannot continue.");
    }

    // Save for next time
    save_api_key(&key)?;

    Ok(key)
}

/// Call the Groq API and return the generated spec text.
pub async fn generate_spec_string(prompt: &str, model: &str, max_tokens: u32) -> Result<String> {
    let api_key = get_api_key()?;

    let user_content = format!(
        "Generate a comprehensive technical specification for the following:\n\n{}",
        prompt
    );

    let request_body = ChatRequest {
        model: model.to_string(),
        max_tokens,
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_content,
            },
        ],
    };

    let client = reqwest::Client::new();
    let response = client
        .post("https://api.groq.com/openai/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await
        .context("Failed to connect to the Groq API")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("Groq API returned error (HTTP {}): {}", status, body);
    }

    let chat_response: ChatResponse = response
        .json()
        .await
        .context("Failed to parse Groq API response")?;

    let content = chat_response
        .choices
        .into_iter()
        .next()
        .context("Groq API returned no choices")?
        .message
        .content;

    Ok(content)
}

/// Main generate orchestration: call Groq API, write output file.
pub async fn run_generate(
    prompt: &str,
    output: &Option<PathBuf>,
    model: &str,
    max_tokens: u32,
) -> Result<()> {
    eprintln!("Prompt: {}", prompt);
    eprintln!("Model:  {}", model);

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .context("Failed to set progress spinner template")?,
    );
    spinner.set_message("Generating specification with Groq...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    let spec_content = generate_spec_string(prompt, model, max_tokens).await?;

    spinner.finish_and_clear();

    match output {
        Some(path) => {
            std::fs::write(path, &spec_content)
                .with_context(|| format!("Failed to write output file: {}", path.display()))?;
            eprintln!(
                "Specification generated successfully!\nFile: {}\nSize: {} characters",
                path.display(),
                spec_content.len()
            );
        }
        None => {
            // No output file specified: print to stdout
            print!("{}", spec_content);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_contains_all_sections() {
        for section in &[
            "Overview",
            "Core Requirements",
            "Technical Architecture",
            "Data Models",
            "Implementation Plan",
            "Edge Cases",
        ] {
            assert!(
                SYSTEM_PROMPT.contains(section),
                "SYSTEM_PROMPT missing section: {section}"
            );
        }
    }

    #[test]
    fn default_constants() {
        assert_eq!(DEFAULT_GROQ_MODEL, "llama-3.3-70b-versatile");
        assert_eq!(DEFAULT_MAX_TOKENS, 8192);
    }

    #[test]
    fn chat_request_body_structure() {
        let request = ChatRequest {
            model: "test-model".to_string(),
            max_tokens: 1024,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: SYSTEM_PROMPT.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "Build a todo app".to_string(),
                },
            ],
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["max_tokens"], 1024);
        assert_eq!(json["messages"].as_array().unwrap().len(), 2);
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][1]["role"], "user");
        assert_eq!(json["messages"][1]["content"], "Build a todo app");
    }

    #[test]
    fn chat_request_serializes_to_valid_json() {
        let request = ChatRequest {
            model: DEFAULT_GROQ_MODEL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "sys".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "prompt".to_string(),
                },
            ],
        };
        let serialized = serde_json::to_string(&request).unwrap();
        assert!(serialized.contains("\"model\""));
        assert!(serialized.contains("\"max_tokens\""));
        assert!(serialized.contains("\"messages\""));
    }

    #[test]
    fn chat_response_deserializes() {
        let json = r#"{
            "choices": [{
                "message": { "content": "Generated spec here" }
            }]
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content, "Generated spec here");
    }
}
