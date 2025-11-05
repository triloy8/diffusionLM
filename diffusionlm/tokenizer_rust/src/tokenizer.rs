// use pyo3::prelude::*;

use std::path::Path;
use std::collections::HashMap;

// #[pyclass]
pub struct Tokenizer;

// #[pymethods]
impl Tokenizer {
    // #[new]
    pub fn new(vocab: HashMap<u32, Vec<u8>>, merges: Vec<(Vec<u8>, Vec<u8>)>, special_tokens: Vec<String>) -> Self {
        Tokenizer
    }

    // #[staticmethod]
    pub fn from_files<P: AsRef<Path>>(vocab_filepath: P, merges_filepath: P, special_tokens: Vec<String>) ->  Tokenizer {
        // gpt2 unicode encoder/decoder
        let encoder: HashMap<u8, char> = gpt2_bytes_to_unicode();
        let decoder: HashMap<char, u8> = encoder.iter().map(|(&id, &ch)| (ch, id)).collect();

        // vocab
        let raw_gpt2_vocab = std::fs::read_to_string(vocab_filepath).expect("Failed to read raw vocab file");
        // println!("{raw_gpt2_vocab}");
        let mut gpt2_vocab: HashMap<String, u32> = serde_json::from_str::<HashMap<String, u32>>(&raw_gpt2_vocab).expect("Failed to parse to json vocab file");
        for special_token in &special_tokens {
            gpt2_vocab.insert(special_token.clone(), gpt2_vocab.values().len() as u32);
        }

        let mut vocab: HashMap<u32, Vec<u8>> = HashMap::new();
        for (gpt2_vocab_word, gpt2_vocab_id) in gpt2_vocab {
            let mut bytes_vec = Vec::<u8>::new();
            for char in gpt2_vocab_word.chars() {
                let bytes = decoder.get(&char).unwrap();
                bytes_vec.push(*bytes);
            }
            vocab.insert(gpt2_vocab_id, bytes_vec);
        }

        // merges
        let gpt2_merges = std::fs::read_to_string(merges_filepath).expect("Failed to read merges file");
        let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for line in gpt2_merges.lines(){
            let cleaned_line = line.trim_end();
            
            if cleaned_line.is_empty(){
                continue
            }

            let parts: Vec<&str> = cleaned_line.split(" ").collect();
            if parts.len() == 2 {
                let mut m1: Vec<u8> = Vec::new();
                for char in parts[0].to_string().chars(){
                    let byte = decoder.get(&char).unwrap();
                    m1.push(*byte)
                }
                let mut m2: Vec<u8> = Vec::new();
                for char in parts[1].to_string().chars(){
                    let byte = decoder.get(&char).unwrap();
                    m2.push(*byte)
                }
                merges.push((m1, m2));
            }
        }

        Tokenizer::new(vocab, merges, special_tokens)
    }

    pub fn encode(&self) {
        todo!()
    }

    pub fn encode_iterable(&self) {
        todo!()
    }

    pub fn decode(&self) {
        todo!()
    }
}

fn gpt2_bytes_to_unicode() -> HashMap<u8, char> {
    let allowed: std::iter::Chain<std::iter::Chain<std::ops::RangeInclusive<char>, std::ops::RangeInclusive<char>>, std::ops::RangeInclusive<char>> = ('!'..='~')
        .chain('¡'..='¬')
        .chain('®'..='ÿ');

    let mut chars: Vec<char> = allowed.clone().collect();
    let mut codes: Vec<u8> = allowed.map(|c| c as u8).collect();
    let mut n = 0;

    for i in 0..=255{
        if !codes.contains(&i){
            codes.push(i);
            chars.push(char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }

    codes.into_iter().zip(chars).collect()
}