// use pyo3::prelude::*;

use std::path::Path;
use std::collections::HashMap;
use regex::Regex;

// #[pyclass]
pub struct Tokenizer{
    re_spec: Option<Regex>,
    re_pat: Regex,
    gpt2_encoder: HashMap<u8, char>,
    gpt2_decoder: HashMap<char, u8>,
    vocab_encoder: HashMap<String, usize>, 
    vocab_decoder: HashMap<usize, String>, 
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

        let mut vocab_encoder: HashMap<String, usize> = HashMap::new();
        let mut vocab_decoder: HashMap<usize, String> = HashMap::new();
        for (gpt2_vocab_word, gpt2_vocab_id) in gpt2_vocab {
            vocab_encoder.insert(gpt2_vocab_word.clone(), gpt2_vocab_id);
            vocab_decoder.insert(gpt2_vocab_id, gpt2_vocab_word);
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

        let re_spec = if special_tokens.is_empty() {
            None
        } else {
            // special token regex
            let special_tokens_pattern = special_tokens
            .iter()
            .map(|tok| regex::escape(tok))
            .collect::<Vec<_>>()
            .join("|");

            Some(Regex::new(&format!("({special_tokens_pattern})")).expect("Failed to validate special tokens regex"))
        };

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

    pub fn encode(&self, text: String) -> Vec<usize>{
        // pretoken
        let parts: Vec<String> = if let Some(regex) = &self.re_spec {
            regex.split(&text).map(|s| s.to_owned()).collect()
        } else {
            vec![text]
        };

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
        let mut pretoken_list_merged: Vec<Vec<String>> = Vec::new();
        for pretoken in pretoken_list {
            if !self.special_tokens.contains(&pretoken) {
                let mut pretoken_gpt2: Vec<String> = Vec::new();
                for b in pretoken.as_bytes() {
                    let ch = self.gpt2_encoder.get(b).expect("Byte not in gpt2_encoder");
                    pretoken_gpt2.push(ch.to_string());
                }
                loop { 
                    if pretoken_gpt2.len() < 2 {
                        break;
                    }

                    #[derive(Clone)]
                    struct MergeCandidate {
                        position: usize,
                        rank: usize,
                        pair: (String, String),
                    }

                    let mut mergeable:Vec<MergeCandidate> = Vec::new(); 
                    for position in 0..(pretoken_gpt2.len()-1){
                        let p0: String = pretoken_gpt2[position].clone();
                        let p1: String = pretoken_gpt2[position+1].clone();
                        let key = (p0.clone(), p1.clone());
                        if let Some(rank) = self.merges.get(&key){
                            mergeable.push(MergeCandidate { 
                                position, 
                                rank: *rank,
                                pair: (p0, p1) 
                            })
                        }
                    }

                    if mergeable.is_empty() {
                        break;
                    }

                    let best = mergeable
                    .iter()
                    .min_by_key(|c| c.rank)
                    .expect("Expected at least one candidate");

                    let position = best.position.clone();
                    let pair = best.pair.clone();

                    let mut new_vec: Vec<String> = Vec::new();
                    for i in 0..position{
                        new_vec.push(pretoken_gpt2[i].clone());
                    }
                    let merged = format!("{}{}", pair.0, pair.1);
                    new_vec.push(merged);   
                    for i in (position+2)..pretoken_gpt2.len(){
                        new_vec.push(pretoken_gpt2[i].clone());
                    }
                    
                    pretoken_gpt2 = new_vec;
                }
                pretoken_list_merged.push(pretoken_gpt2);

            } else {
                pretoken_list_merged.push(vec![pretoken]);
            }
        }

        // encodings 
        let mut encoding: Vec<usize> = Vec::new();
        for pretoken_merge in pretoken_list_merged{
            for merge in pretoken_merge {
                let id = self.vocab_encoder.get(&merge).expect("Merge not found in vocab_encoder");
                encoding.push(*id);
            }
        }

        encoding

    }

    pub fn encode_iterable<'a, I>(&'a self, iter: I) -> impl Iterator<Item = usize> + 'a 
    where
        I: IntoIterator<Item = &'a str> + 'a,
    {
        iter.into_iter().flat_map(|line| self.encode(line.to_string()))
    }

    pub fn decode(&self, ids: Vec<usize>) -> String {
        let gpt2_encoded_string = ids.iter()
        .map(|id| self.vocab_decoder.get(&id).expect("Decode id not found in vocab").as_str())
        .collect::<Vec<_>>()
        .join("");

        let mut utf8_bytes = Vec::new();
        for char in gpt2_encoded_string.chars(){
            utf8_bytes.push(*self.gpt2_decoder.get(&char).expect("Decoded char not found in gpt2 decoder"));
        }

        let decoded_string = String::from_utf8_lossy(&utf8_bytes).into_owned();

        decoded_string
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