use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn help_exits_zero() {
    Command::cargo_bin("devpipe")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("compress"));
}

#[test]
fn version_prints_version() {
    Command::cargo_bin("devpipe")
        .unwrap()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("devpipe"));
}

#[test]
fn compress_help_shows_options() {
    Command::cargo_bin("devpipe")
        .unwrap()
        .args(["compress", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("keep-ratio"))
        .stdout(predicate::str::contains("stats"))
        .stdout(predicate::str::contains("auto"))
        .stdout(predicate::str::contains("target-tokens"));
}

#[test]
fn generate_help_shows_options() {
    Command::cargo_bin("devpipe")
        .unwrap()
        .args(["generate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("PROMPT"))
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--json"))
        .stdout(predicate::str::contains("--stats"));
}
