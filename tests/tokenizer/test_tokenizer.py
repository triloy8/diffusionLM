import pytest

from diffusionlm.tokenizer.tokenizer import Tokenizer
from diffusionlm.tokenizer import gpt2_bytes_to_unicode


def make_minimal_tokenizer():
    # Minimal byte-level vocab covering space, 'a', 'b', 'ab', and a special token
    vocab = {
        0: b" ",
        1: b"a",
        2: b"b",
        3: b"ab",  # merged token
        4: b"<|eot|>",
    }
    merges = [
        (b"a", b"b"),
    ]
    special_tokens = ["<|eot|>"]
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def test_roundtrip_basic_ascii_and_spaces():
    tok = make_minimal_tokenizer()
    text = "a b ab"
    ids = tok.encode(text)
    out = tok.decode(ids)
    assert out == text


def test_special_tokens_are_atomic_and_roundtrip():
    tok = make_minimal_tokenizer()
    text = "a<|eot|>b"
    ids = tok.encode(text)
    # Expect exactly three tokens: 'a', special, 'b'
    assert len(ids) == 3
    dec = tok.decode(ids)
    assert dec == text


def test_merge_application_on_small_example():
    tok = make_minimal_tokenizer()
    ids = tok.encode("ab")
    # Should have a single token id corresponding to b"ab"
    assert len(ids) == 1


def test_encode_iterable_yields_consistent_stream():
    tok = make_minimal_tokenizer()
    lines = ["a b", "ab"]
    list_ids = list(tok.encode_iterable(lines))
    # Equivalent to concatenated encoding
    expected = tok.encode("".join(lines))
    assert list_ids == expected


def test_deterministic_merge_order():
    # Craft merges where (b,c) has higher priority than (a,b)
    vocab = {
        0: b"a",
        1: b"b",
        2: b"c",
        3: b"bc",
        4: b"ab",
    }
    merges = [
        (b"b", b"c"),  # rank 0
        (b"a", b"b"),  # rank 1
    ]
    tok = Tokenizer(vocab=vocab, merges=merges, special_tokens=[])
    ids = tok.encode("abc")
    # Expect 'a' + 'bc' (since (b,c) has higher priority)
    # Thus, two tokens
    assert len(ids) == 2


def test_tokenizer_from_files_roundtrip(tmp_path):
    enc = gpt2_bytes_to_unicode()
    # Build a tiny GPT-2-like vocab JSON: token_string -> index
    def enc_str(s: bytes) -> str:
        return "".join(enc[b] for b in s)

    vocab_json = {
        enc_str(b" "): 0,
        enc_str(b"a"): 1,
        enc_str(b"b"): 2,
        enc_str(b"ab"): 3,
    }
    merges_txt = "a b\n"

    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    vocab_path.write_text(__import__("json").dumps(vocab_json))
    merges_path.write_text(merges_txt)

    tok = Tokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=["<|eot|>"])
    text = "a ab b"
    ids = tok.encode(text)
    out = tok.decode(ids)
    assert out == text


def test_no_merge_across_special_tokens():
    # Even if merges allow (a,b), they must not cross a special token boundary
    vocab = {
        0: b"a",
        1: b"b",
        2: b"ab",
        3: b"<|eot|>",
    }
    merges = [
        (b"a", b"b"),
    ]
    special = ["<|eot|>"]
    tok = Tokenizer(vocab=vocab, merges=merges, special_tokens=special)

    text = "a<|eot|>b"
    ids = tok.encode(text)
    # Expect exactly three tokens: 'a', special, 'b' (no merge across special)
    assert len(ids) == 3
    assert tok.decode(ids) == text

    # Inner merges still apply on either side of the special token
    text2 = "ab<|eot|>ab"
    ids2 = tok.encode(text2)
    # 'ab' merges on both sides plus the special → 3 tokens total
    assert len(ids2) == 3
    assert tok.decode(ids2) == text2
