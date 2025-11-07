// use pyo3::prelude::*;

use std::path::Path;
use std::collections::HashMap;
use regex::Regex;

// #[pyclass]
pub struct Tokenizer{
    re_spec: Regex,
    re_pat: Regex,
    gpt2_encoder: HashMap<u8, char>,
    gpt2_decoder: HashMap<char, u8>,
    vocab_encoder: HashMap<usize, String>, 
    vocab_decoder: HashMap<String, usize>, 
    merges: HashMap<(String, String), usize>, 
    special_tokens: Vec<String>,
}

// #[pymethods]
impl Tokenizer {
    // #[staticmethod]
    pub fn from_files<P: AsRef<Path>>(vocab_filepath: P, merges_filepath: P, special_tokens: Vec<String>) ->  Tokenizer {
        // gpt2 unicode encoder/decoder
        let gpt2_encoder: HashMap<u8, char> = gpt2_bytes_to_unicode();
        let gpt2_decoder: HashMap<char, u8> = gpt2_encoder.iter().map(|(&id, &ch)| (ch, id)).collect();

        // vocab
        let raw_gpt2_vocab = std::fs::read_to_string(vocab_filepath).expect("Failed to read raw vocab file");
        let mut gpt2_vocab: HashMap<String, usize> = serde_json::from_str::<HashMap<String, usize>>(&raw_gpt2_vocab).expect("Failed to parse to json vocab file");
        for special_token in &special_tokens {
            gpt2_vocab.insert(special_token.clone(), gpt2_vocab.values().len());
        }

        let mut vocab_encoder: HashMap<usize, String> = HashMap::new();
        let mut vocab_decoder: HashMap<String, usize> = HashMap::new();
        for (gpt2_vocab_word, gpt2_vocab_id) in gpt2_vocab {
            vocab_encoder.insert(gpt2_vocab_id, gpt2_vocab_word.clone());
            vocab_decoder.insert(gpt2_vocab_word, gpt2_vocab_id);
        }

        // merges
        let gpt2_merges = std::fs::read_to_string(merges_filepath).expect("Failed to read merges file");
        let mut merges: HashMap<(String, String), usize> = HashMap::new();
        for (i, line) in gpt2_merges.lines().enumerate() {
            let cleaned_line = line.trim_end();
            
            if cleaned_line.is_empty(){
                continue
            }

            let parts: Vec<&str> = cleaned_line.split(" ").collect();
            if parts.len() == 2 {
                merges.insert((parts[0].to_string(), parts[1].to_string()), i);
            }
        }

        // regex
        let special_tokens_pattern = special_tokens
        .iter()
        .map(|tok| regex::escape(tok))
        .collect::<Vec<_>>()
        .join("|");

        let re_spec = Regex::new(&format!("({special_tokens_pattern})")).expect("Failed to validate special tokens regex");

        // the old gpt2 pattern, the newer one is not supported on regex bc look-around is not implemented
        let pat: &str = r"(?:'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?:\S|\z))";
        let re_pat = Regex::new(pat).expect("Failed to create PAT regex");

        Tokenizer{
            re_spec,
            re_pat,
            gpt2_encoder,
            gpt2_decoder,
            vocab_encoder, 
            vocab_decoder, 
            merges, 
            special_tokens
        }
    }

    pub fn encode(&self, text: String) -> Vec<String>{
        // pretoken
        if self.special_tokens.is_empty() {
            return vec![text];
        }

        let parts: Vec<String> = self.re_spec.split(&text).map(|s| s.to_owned()).collect();

        let mut pretoken_list: Vec<String> = Vec::new();

        for part in parts {
            if self.special_tokens.contains(&part) {
                pretoken_list.push(part);
            } else if part.is_empty() {
                continue;
            } else {
                for m in self.re_pat.find_iter(&part) {
                    pretoken_list.push(m.as_str().to_string());
                }
            }
        }

        // merges
        // encodings 
        vec!["placeholder".to_string()]
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