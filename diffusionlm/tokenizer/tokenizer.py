from diffusionlm.tokenizer.pretokenize import gpt2_bytes_to_unicode, PAT
from typing import Iterable, Iterator
import json
import regex as re

class Tokenizer:

    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        
        self.vocab_decoder = {
            vocab_index: "".join([self.gpt2_byte_encoder[byte] for byte in vocab_bytes])
            for vocab_index, vocab_bytes in vocab.items()
        }
        self.vocab_encoder = {v: k for k, v in self.vocab_decoder.items()}

        self.merges = {
            (
                "".join([self.gpt2_byte_encoder[c] for c in merge[0]]),
                "".join([self.gpt2_byte_encoder[c] for c in merge[1]])
                
            ):i
            for i, merge in enumerate(merges)
        }
        self.special_tokens = special_tokens or []
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens_path: str,
    ):

        # vocab w/ special tokens
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        vocab = {
            gpt2_vocab_index: bytes([cls.gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        gpt2_special_tokens = {}
        with open(special_tokens_path, encoding="utf-8") as special_tokens_f:
            gpt2_special_tokens = json.load(special_tokens_f)
        for special_token, token_id in gpt2_special_tokens.items():
            byte_encoded_special_token = special_token.encode("utf-8")
            vocab[int(token_id)] = byte_encoded_special_token

        # merges
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        merges = [
            (
                bytes([cls.gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([cls.gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]

        return cls(vocab, merges, list(gpt2_special_tokens.keys()))

    def encode(self, 
               text: str) -> list[int]:

        # pretokenize
        if self.special_tokens:
            special_pat = "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True))
            special_pat_split = f"({special_pat})"
            parts = re.split(special_pat_split, text)
        else:
            parts = [text]
        
        pretoken_list = []
        for part in parts:
            if part in self.special_tokens:
                pretoken_list.append(part)
            elif not part:
              continue
            else:
                pretoken_list.extend(m.group(0) for m in re.finditer(PAT, part))

        # applying merges
        pretoken_list_merged = []
        for pretoken in pretoken_list:
            if pretoken not in self.special_tokens:
                pretoken_gpt2 = list([self.gpt2_byte_encoder[pretoken_i] for pretoken_i in pretoken.encode("utf-8")])
                while True:
                    pretoken_gpt2_pairs = [(pretoken_gpt2[i], pretoken_gpt2[i+1]) for i in range(len(pretoken_gpt2) - 1)]
                    mergeable = [(position, self.merges[pair], pair) for position, pair in enumerate(pretoken_gpt2_pairs) if pair in self.merges.keys()]
                    if not mergeable:
                        break
                    position, i, pair = min(mergeable, key=lambda x: x[1])
                    pretoken_gpt2 = pretoken_gpt2[:position] + [pair[0] + pair[1]] + pretoken_gpt2[position+2:]
                pretoken_list_merged.append(tuple(pretoken_gpt2))
            else:
                pretoken_list_merged.append((pretoken,))

        # encoding to bytes encoding
        bytes_encoding = [self.vocab_encoder[merge] for pretoken_tuple in pretoken_list_merged for merge in pretoken_tuple]

        return bytes_encoding

    def encode_iterable(self, 
                        iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            ids = self.encode(line)
            for _id in ids:
                yield _id

    def decode(self, 
               ids: list[int]) -> str:
        
        
        decoded_string = bytes(
                                [
                                    self.gpt2_byte_decoder[gpt2_byte_encoded] 
                                    for gpt2_byte_encoded in "".join([self.vocab_decoder[id] for id in ids])
                                ]
                            ).decode("utf-8", errors='replace')

        return decoded_string
