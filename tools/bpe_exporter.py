# The MIT License (MIT)
#
# Copyright (c) 2023 Xiaoyang Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import argparse
import sys
import struct
import configparser
from copy import copy
from functools import partial
from typing import Dict, List, Tuple, Callable
from os import path

class TestCaseExporter:
    SAMPLE_TEXT = [
        "The Touhou Project (Japanese: 東方Project, Hepburn: Tōhō Purojekuto), ",
        "also known simply as Touhou (東方, literally ",
        '''"Eastern" or "Oriental"), is a bullet hell shoot 'em up video game series created by one-man ''',
        "independent Japanese",
        "doujin soft developer Team Shanghai Alice.",
        """Since 1995,[1][2] the team's member, Jun'ya "ZUN" Ōta, has independently""",
        "developed programming, graphics, writing, and music for the series, self-publishing 18 mainline games and six",
        "spin-offs as of August 2022. ",
        "ZUN has also produced related print works and music albums, and collaborated with",
        "developer Twilight Frontier on seven official Touhou spin-offs, most being fighting games.[3]",
        "🐱"
    ]

    @classmethod
    def run(cls, tokenize_func: Callable[[str], List[str]], output_file: str):
        """Invoke sentencepiece sample export with arguments"""
        with open(output_file, 'w', encoding="utf-8") as fp:
            for text in cls.SAMPLE_TEXT:
                text_encoded = " ".join(tokenize_func(text))
                fp.write(f"{text}\t{text_encoded}\n")

class Token:
    FLAG_UNK = 1
    FLAG_CONTROL = 2
    FLAG_BYTE = 4
    FLAG_UNUSED = 8

    @classmethod
    def unused(cls, token_id: int) -> 'Token':
        """create an unused token"""
        return Token(token_id, cls.FLAG_UNUSED, b"<UNUSED>", 0)

    def __init__(self, 
                 token_id: int,
                 flag: int = 0,
                 piece: bytes = b"",
                 piece_display: str = "",
                 weight: float = None) -> None:
        self.token_id = token_id
        self.flag = flag
        self.piece = piece
        self.piece_display = piece_display
        self.weight = weight

    def is_unused(self) -> bool:
        return self.FLAG_UNUSED & self.flag != 0

class TokenizerConfig:
    def __init__(self,
                 add_prefix_space: bool,
                 split_by_unicode: bool) -> None:
        self.add_prefix_space = add_prefix_space
        self.split_by_unicode = split_by_unicode
        self.model_file = None

    def update_config(self, config: configparser.ConfigParser, section: str) -> None:
        config[section] = {}
        config[section]["type"] = "bpe"
        config[section]["model_file"] = self.model_file
        config[section]["add_prefix_space"] = str(self.add_prefix_space).lower()
        config[section]["split_by_unicode"] = str(self.split_by_unicode).lower()

class LibllmTokenizer:
    MAGIC_NUMBER = 0x55aa

    def __init__(self):
        self._vocab = []
        self._bin_model_file = None
        self.config = TokenizerConfig(add_prefix_space=True, split_by_unicode=True)
    
    def add_token(self, token: Token) -> None:
        if token.token_id != len(self._vocab):
            raise Exception("token_id and vocab_size mismatch")
        self._vocab.append(token)

    def save(self, filename: str) -> None:
        """save the sentencepiece model as libllm format"""
        self._bin_model_file = path.basename(filename)
        print(f"save libllm model: {filename}")

        with open(filename, 'wb') as fp:
            fp.write(b'LLsp')
            fp.write(struct.pack('<l', len(self._vocab)))
            fp.write(struct.pack('<h', self.MAGIC_NUMBER))
            for token in self._vocab:
                piece_display = Util.truncate_display(token.piece_display)

                fp.write(struct.pack('<b', token.flag))
                fp.write(struct.pack('<B', len(token.piece)))
                fp.write(token.piece)
                fp.write(struct.pack('<B', len(piece_display)))
                fp.write(piece_display)
                fp.write(struct.pack('<f', token.weight))
            fp.write(struct.pack('<h', self.MAGIC_NUMBER))

    def save_text_model(self, filename: str) -> None:
        """save the sentencepiece model as libllm format"""
        with open(filename, 'w', encoding="utf-8") as fp:
            for token in self._vocab:
                piece_display = self.truncate_display(token.piece_display)
                piece_display = piece_display.decode("utf-8")
                piece = token.piece.decode()
                fp.write(f"{token.token_id}\t0x{token.flag:02x}\t{token.weight}\t{piece}\t{piece_display}\n")

    def get_vocab(self):
        return self._vocab
    
    def get_config(self) -> TokenizerConfig:
        if self._bin_model_file is None:
            raise Exception("get_config() called before save()")\

        self.config.model_file = self._bin_model_file
        return self.config

class SentencePieceModelReader:
    def __init__(self, name: str) -> None:
        from transformers import AutoTokenizer
        from sentencepiece import SentencePieceProcessor
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self._sp = SentencePieceProcessor(model_file=tokenizer.vocab_file)

    def to_libllm_model(self) -> LibllmTokenizer:
        """convert the sentencepiece model to libllm tokenizer format."""
        libllm_model = LibllmTokenizer()
        for token_id in range(self._sp.vocab_size()):
            flag = 0
            piece = b''
            piece_display = ""
            if self._sp.IsUnknown(token_id):
                flag = flag | Token.FLAG_UNK
                piece_display = self._sp.IdToPiece(token_id)
            if self._sp.IsControl(token_id):
                flag = flag | Token.FLAG_CONTROL
                piece_display = self._sp.IdToPiece(token_id)
            if self._sp.IsUnused(token_id):
                flag = flag | Token.FLAG_UNUSED
                piece_display = self._sp.IdToPiece(token_id)
            if self._sp.IsByte(token_id):
                flag = flag | Token.FLAG_BYTE
                b = int(self._sp.IdToPiece(token_id)[1: -1], 16)
                b = struct.pack('B', b)
                piece = b
                piece_display = self._sp.IdToPiece(token_id)
            if flag == 0:
                piece = self._sp.IdToPiece(token_id).replace("\u2581", " ").encode('utf-8')
                piece_display = self._sp.IdToPiece(token_id)

            piece = piece
            piece_display = piece_display
            
            libllm_model.add_token(Token(token_id, flag, piece, piece_display, self._sp.GetScore(token_id)))
        
        return libllm_model
    
    def encode_as_pieces(self, s: str) -> List[str]:
        return self._sp.EncodeAsPieces(s)

class TiktokenModelReader:
    """model reader for the BPE tokenizer from transformers"""

    def __init__(self, tiktoken_enc) -> None:
        self._tokenizer = tiktoken_enc

    def to_libllm_model(self) -> List[Token]:
        """convert the BPE model to lytok tokenizer format."""
        libllm_model = LibllmTokenizer()
        for piece, rank in self._tokenizer._mergeable_ranks.items():
            libllm_model.add_token(Token(rank, 0, piece, Util.piece_to_display(piece), -rank))

        # special tokens
        for token, token_id in self._tokenizer._special_tokens.items():
            libllm_model.add_token(Token(token_id, Token.FLAG_CONTROL, b'', token, 0))

        libllm_model.config.add_prefix_space = False
        return libllm_model


class TransformersBpeModelReader:
    """model reader for the BPE tokenizer from transformers"""

    def __init__(self, tokenizer) -> None:
        self._bpe = tokenizer

    def _to_byte(self, s: str) -> bytes:
        """convert unicode string to bytes according to the char to byte mapping table"""
        b = b''
        byte_decoder = self._bpe.byte_decoder
        for ch in s:
            assert ch in byte_decoder, "invalid character"
            byte_ord = byte_decoder[ch]
            b += byte_ord.to_bytes(1, 'little')
        
        return b

    def to_libllm_model(self) -> List[Token]:
        """convert the BPE model to lytok tokenizer format."""
        pieces: Dict[bytes, Token] = {}

        # text and flag
        for piece, piece_id in self._bpe.encoder.items():
            b_piece = self._to_byte(piece)
            pieces[piece] = Token(piece_id, 0, b_piece, Util.piece_to_display(b_piece))
        
        # weight
        for piece_pair, rank in self._bpe.bpe_ranks.items():
            piece = piece_pair[0] + piece_pair[1]
            if pieces[piece].weight is not None:
                print(f'pair for {piece} already exists')
            pieces[piece].weight = -rank

        # vocab
        vocab: List[Token] = []
        for token_id in range(self._bpe.vocab_size):
            vocab.append(Token.unused(token_id))
        
        for token in pieces.values():
            if not vocab[token.token_id].is_unused():
                raise Exception(f"duplicated token id {token.token_id}")
            vocab[token.token_id] = token

        # special symbols
        for token_id, piece in zip(self._bpe.all_special_ids, self._bpe.all_special_tokens):
            while token_id >= len(vocab):
                vocab.append(Token.unused(token_id))

            vocab[token_id] = Token(
                token_id,
                Token.FLAG_CONTROL,
                piece=b"",
                piece_display=piece,
                weight=0)
        if self._bpe.unk_token_id is not None:
            vocab[self._bpe.unk_token_id].flag |= Token.FLAG_UNK

        # update weight None to 0
        libllm_model = LibllmTokenizer()
        for token in vocab:
            if token.weight is None:
                token.weight = 0
            libllm_model.add_token(token)

        return libllm_model

class Util:
    @classmethod
    def escape_string(cls, s):
        e = ""
        for ch in s:
            if ord(ch) <= 32:
                ch = f"\\x{ord(ch):02x}"
            e += ch
        return e

    @classmethod
    def piece_to_display(cls, piece):
        assert isinstance(piece, bytes)
        try:
            piece = piece.decode("utf-8")
        except:
            piece = str(piece)[2:-1]

        escaped_piece = ""
        for ch in piece:
            if ord(ch) < 32:
                ch = f"\\x{ord(ch):02x}"
            elif ch == " ":
                ch = "\u2581"
            escaped_piece += ch

        return escaped_piece

    @classmethod
    def truncate_display(cls, s: str) -> bytes:
        bs = s.encode('utf-8')
        if len(bs) <= 255:
            return bs
        
        _trunk_repr = lambda s: s.encode('utf-8') + b"...(truncated)"

        bs = _trunk_repr(s)
        while len(bs) >= 256:
            s = s[: -1]
            bs = _trunk_repr(s)

        return bs

def read_spm_model(model_path_or_name: str) -> LibllmTokenizer:
    print("read sentencepiece model from " + model_path_or_name)
    spm_model = SentencePieceModelReader(model_path_or_name)
    libllm_model = spm_model.to_libllm_model()
    print(f"vocab_size={len(libllm_model.get_vocab())}")
    return libllm_model

def read_transformers_bpe_model(transformers_bpe) -> LibllmTokenizer:
    bpe_model = TransformersBpeModelReader(transformers_bpe)
    libllm_model = bpe_model.to_libllm_model()
    print(f"vocab_size={len(libllm_model.get_vocab())}")
    return libllm_model

def read_tiktoken_model(tiktoken_enc) -> LibllmTokenizer:
    tiktoken_model = TiktokenModelReader(tiktoken_enc)
    libllm_model = tiktoken_model.to_libllm_model()
    print(f"vocab_size={len(libllm_model.get_vocab())}")
    return libllm_model
