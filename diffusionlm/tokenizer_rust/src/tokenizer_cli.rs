#![cfg(feature = "cli")] 
use std::println;

use tokenizer_rs::tokenizer::Tokenizer;

fn main() { 
    let tokenizer = Tokenizer::new();
    let vocab_filepath = "../../data/gpt2_vocab.json";
    let merges_filepath = "../../data/gpt2_merges.txt";
    let special_tokens = vec!["<|endoftext|>".to_string(), "<|mask|>".to_string()];
    Tokenizer::from_files(vocab_filepath, merges_filepath, special_tokens);
 }