use serde_json::json;
use std::fs;
use tempfile::tempdir;

use crate::tokenizer::{Tokenizer, gpt2_bytes_to_unicode};

fn encode_bytes(bytes: &[u8], enc: &std::collections::HashMap<u8, char>) -> String {
    bytes
        .iter()
        .map(|b| enc.get(b).copied().expect("byte missing from encoder"))
        .collect()
}

fn build_test_tokenizer() -> Tokenizer {
    let tmp_dir = tempdir().expect("failed to create tempdir for tokenizer fixtures");
    let vocab_path = tmp_dir.path().join("vocab.json");
    let merges_path = tmp_dir.path().join("merges.txt");
    let special_tokens_path = tmp_dir.path().join("special_tokens.json");

    let enc = gpt2_bytes_to_unicode();

    let vocab_json = json!({
        encode_bytes(b" ", &enc): 0,
        encode_bytes(b"a", &enc): 1,
        encode_bytes(b"b", &enc): 2,
        encode_bytes(b"ab", &enc): 3,
    });

    let vocab_serialized = serde_json::to_string(&vocab_json).expect("failed to serialize vocab json");
    fs::write(&vocab_path, vocab_serialized).expect("failed to write vocab json");
    fs::write(&merges_path, "a b\n").expect("failed to write merges file");
    let special_tokens_json = json!({
        "<|eot|>": 4
    });
    let special_tokens_serialized =
        serde_json::to_string(&special_tokens_json).expect("failed to serialize special tokens json");
    fs::write(&special_tokens_path, special_tokens_serialized)
        .expect("failed to write special tokens file");

    Tokenizer::from_files(
        &vocab_path,
        &merges_path,
        &special_tokens_path,
    )
        .expect("failed to build tokenizer fixture")
}

#[test]
fn encode_decode_roundtrip_from_files() {
    let tokenizer = build_test_tokenizer();
    let text = "a ab b";
    let ids = tokenizer.encode(text.to_string()).expect("encode should succeed");
    let decoded = tokenizer.decode(ids).expect("decode should succeed");
    assert_eq!(decoded, text);
}

#[test]
fn merges_respect_special_token_boundaries() {
    let tokenizer = build_test_tokenizer();
    let text = "a<|eot|>b";
    let ids = tokenizer.encode(text.to_string()).expect("encode should succeed");
    assert_eq!(ids.len(), 3, "special token should prevent merges across boundary");

    let decoded = tokenizer.decode(ids.clone()).expect("decode should succeed");
    assert_eq!(decoded, text);

    let text2 = "ab<|eot|>ab";
    let ids2 = tokenizer.encode(text2.to_string()).expect("encode should succeed");
    assert_eq!(ids2.len(), 3, "merges should apply independently on each side of special token");
    assert_eq!(tokenizer.decode(ids2).expect("decode should succeed"), text2);
}
