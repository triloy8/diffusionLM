// use pyo3::prelude::*;

use std::path::Path;
use std::collections::HashMap;

// #[pyclass]
pub struct Tokenizer;

// #[pymethods]
impl Tokenizer {
    // #[new]
    pub fn new() -> Self {
        Tokenizer
    }

    // #[staticmethod]
    pub fn from_files<P: AsRef<Path>>(vocab_filepath: P, merges_filepath: P, special_tokens: Vec<String>) ->  Result<(), std::io::Error> {
        let encoder: HashMap<u32, char> = gpt2_bytes_to_unicode();
        let decoder: HashMap<char, u32> = encoder.iter().map(|(&id, &ch)| (ch, id)).collect();
        // for (key, value) in &encoder {
        //     println!("{key} => {value}");
        // }

        // vocab
        let raw_gpt2_vocab = std::fs::read_to_string(vocab_filepath)?;
        // println!("{raw_gpt2_vocab}");
        let mut gpt2_vocab: HashMap<String, u32> = serde_json::from_str::<HashMap<String, u32>>(&raw_gpt2_vocab)?;
        for special_token in special_tokens {
            gpt2_vocab.insert(special_token, gpt2_vocab.values().len() as u32);
        }
        let vocab: HashMap<u32, &[u8]> = gpt2_vocab
        .iter()
        .map(|(word, &id)| (id, word.as_bytes())).collect();
        
        // if let Err(gpt2_vocab) = serde_json::from_str::<HashMap<String, u32>>(&raw_gpt2_vocab){
        //     println!{"{gpt2_vocab}"};
        // }
        // for (key, value) in &gpt2_vocab {
        //     println!("{key} => {value}");
        // }
        // for (key, &value) in &vocab {
        //     println!("{key} => {:?}", &value);
        // }

        // merges
        let gpt2_merges = std::fs::read_to_string(merges_filepath)?;
        // println!("{gpt2_merges}");
        

        Ok(())
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

fn gpt2_bytes_to_unicode() -> HashMap<u32, char> {
    let allowed: std::iter::Chain<std::iter::Chain<std::ops::RangeInclusive<char>, std::ops::RangeInclusive<char>>, std::ops::RangeInclusive<char>> = ('!'..='~')
        .chain('¡'..='¬')
        .chain('®'..='ÿ');

    let mut chars: Vec<char> = allowed.clone().collect();
    let mut codes: Vec<u32> = allowed.map(|c| c as u32).collect();
    let mut n = 0;

    for i in 0..256{
        if !codes.contains(&i){
            codes.push(i);
            chars.push(char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }

    codes.into_iter().zip(chars).collect()
}