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
import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
import configparser
import zipfile
import io
import sys
from os import path
from model_exporter import Context, ModelExporter, TensorWriter, Quant
from bpe_exporter import read_spm_model, read_transformers_fast_bpe_model
from torch import nn

class LlamaExporter(ModelExporter):
    def __init__(self, writer: TensorWriter) -> None:
        super().__init__(writer)

    def _export(self, ctx: Context, model):
        base_model = model.base_model

        self.export_embedding(ctx.with_subname("embd"), base_model.embed_tokens)
        self._export_rms_norm(ctx.with_subname("norm"), base_model.norm)
        self._export_rope(ctx.with_subname("rope"), base_model.layers[0].self_attn.rotary_emb)
        for idx, block in enumerate(base_model.layers):
            self._export_block(ctx.with_subname("block" + str(idx)), block)
        self._write(ctx.with_subname("out_proj.weight"), model.lm_head.weight)

    def _export_rms_norm(self, ctx: Context, module):
        self._write(ctx.with_subname("weight").with_quant(Quant.NONE), module.weight)

    def _export_block(self, ctx: Context, model_block):
        self._export_rms_norm(ctx.with_subname("input_norm"), model_block.input_layernorm)
        self._export_attn(ctx.with_subname("attn"), model_block.self_attn)
        self._export_rms_norm(ctx.with_subname("post_attn_norm"), model_block.post_attention_layernorm)
        self._export_mlp(ctx.with_subname("mlp"), model_block.mlp)

    def _export_rope(self, ctx: Context, rope_embd):
        original_max_seq_len = rope_embd.original_max_seq_len
        position_ids = torch.arange(original_max_seq_len).unsqueeze(0)
        x = torch.ones(1)

        cos_cached, sin_cached = rope_embd(x, position_ids)

        rope = torch.stack((cos_cached, sin_cached))
        self._write(ctx.with_quant(Quant.NONE), rope)

    def _export_attn(self, ctx: Context, attn_block):
        q_proj = attn_block.q_proj.weight
        k_proj = attn_block.k_proj.weight
        v_proj = attn_block.v_proj.weight

        qkv_proj = torch.cat((q_proj, k_proj, v_proj), dim=0)
        self._write(ctx.with_subname("qkv_proj.weight"), qkv_proj)
        self._write(ctx.with_subname("out_proj.weight"), attn_block.o_proj.weight)

    def _export_mlp(self, ctx: Context, attn_block):
        w_gate = attn_block.gate_proj.weight
        w_up = attn_block.up_proj.weight
        w_gate_up = torch.cat((w_gate, w_up), dim=0)
        self._write(ctx.with_subname("gate_up_proj.weight"), w_gate_up)
        self._write(ctx.with_subname("down_proj.weight"), attn_block.down_proj.weight)

    @classmethod
    def generate_config(cls, llama_config) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config["llama"] = {}

        print("llama_config.rope_scaling =", llama_config.rope_scaling)

        assert llama_config.pretraining_tp == 1
        assert llama_config.hidden_act == "silu"
        
        section = config["llama"]
        section["hidden_size"] = str(llama_config.hidden_size)
        section["num_heads"] = str(llama_config.num_attention_heads)
        section["num_key_value_heads"] = str(llama_config.num_key_value_heads)
        section["intermediate_size"] = str(llama_config.intermediate_size)
        section["norm_eps"] = str(llama_config.rms_norm_eps)
        section["num_layers"] = str(llama_config.num_hidden_layers)
        section["vocab_size"] = str(llama_config.vocab_size)
        section["max_ctx_length"] = str(llama_config.max_position_embeddings)
        section["bot_token_id"] = "1"
        section["eot_token_id"] = "2"

        return config

    @classmethod
    def export(cls, llama_model, fp):
        model = llama_model
        config = llama_model.config

        ctx = Context("llama")
        with TensorWriter(fp) as writer:
            exporter = LlamaExporter(writer)
            exporter._export(ctx, model)

        ini_config = cls.generate_config(config)
        ini_config["model"] = {}
        ini_config["model"]["type"] = "llama"
        ini_config["model"]["model_file"] = path.basename(MODEL_BIN)

        return ini_config

def run_llama_chat(huggingface_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(huggingface_name, use_fast=False)

    prompt = "hi<s>"
    messages = [
        {"role": "user", "content": prompt}
    ]
    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(huggingface_name, device_map="cpu")
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    print(generated_ids)

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_BIN = "model.bin"
MODEL_INI = "model.ini"
TOKENIZER_BIN = "tokenizer.bin"
TOKENIZER_INI = "tokenizer.ini"
TEST_CASE_BIN = "test_case.bin"

# reference input_ids and logits for these sentences are exported alongside the model, so libllm
# can be checked end-to-end against huggingface.
TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Marisa Kirisame (霧雨 魔理沙) is an ordinary human magician who specializes in light and heat magic and currently resides in the Forest of Magic. She is considered to be the deuteragonist of the Touhou Project series along with the main protagonist, Reimu Hakurei.",
]

TEST_SENTENCES += ["""Alice Margatroid
Personality
Alice is generally indifferent toward others and prefers an indoor, solitary lifestyle. She rarely leaves the Forest of Magic and knows relatively little about the world outside it. Although she usually presents herself confidently, she also has a timid side.
She has a strong interest in magic and enjoys collecting magical items and books. This sometimes causes conflict with Marisa Kirisame, who has similar interests.
Despite being a youkai, Alice is relatively friendly toward humans and is considered to have a low danger level. If someone becomes lost in the Forest of Magic and reaches her house, she is willing to offer them shelter. However, her home is filled with dolls, and her quiet personality can make visitors uncomfortable.
Alice can be hospitable when she chooses. She has served tea and cake to the Three Fairies of Light and offered them friendship after learning their identities. She has also helped Sakuya Izayoi find her way out of the Forest of Magic.
Alice was originally human and does not normally attack humans in the way some youkai do. However, she enjoys fighting and readily accepts challenges.
In combat, Alice prefers intelligence, tactics, and careful control over overwhelming force. She deliberately avoids using all of her strength and generally tries to fight at a level only slightly above her opponent. She dislikes simply overpowering an enemy. Even when losing, she may refuse to reveal her full strength because she believes losing after using everything she has would leave her with nowhere further to go.
Abilities
Alice is a magician whose specialty is magic involving dolls.
Her physical and magical power are considered relatively average, but she possesses exceptional dexterity and extremely refined control over her dolls. She can manipulate a large number of dolls simultaneously and make them behave almost like living beings.
Her dolls are not actually alive. Alice controls them through magical threads rather than physically moving them with her hands.
The dolls can perform most actions that humans can perform. They can also manipulate other dolls. Alice commonly uses them for household tasks such as cooking, laundry, cleaning, maintenance, and other chores.
Alice personally creates her dolls rather than having existing dolls manufacture new ones.
During battle, Alice usually fights through large groups of dolls. Destroyed dolls can quickly be replaced, allowing her to overwhelm opponents through numbers and positioning.
Some dolls contain gunpowder or can be magically detonated. Alice may use dolls that are no longer needed as explosive attacks.
Because controlling many dolls requires concentration, Alice herself tends to move relatively slowly during combat. Directly attacking Alice can therefore be an effective way to disrupt her fighting style.
Most of her danmaku attacks are performed by her dolls. When Alice attacks directly, her attacks tend to be simpler and weaker.
Her dolls have occasionally been shown speaking or displaying apparent emotions. However, they are still being controlled by Alice rather than acting independently, meaning these conversations can effectively be considered Alice performing both sides herself.
Autonomous Doll Research
Alice's long-term objective is to create a completely autonomous doll capable of thinking and moving according to its own will.
She has not yet succeeded and acknowledges that she still has much to learn.
Her research includes experiments concerning the relationship between bodies and souls. She believes that the connection between body and soul is related to the same principle that allows her magical strings to control dolls.
One experiment involved attaching humanoid straw dolls to trees in order to investigate how actions performed on a representation could affect the associated subject.
Despite pursuing autonomous dolls as an important research goal, Alice admits that dolls directly controlled by her are currently more practical and useful than dolls capable of independent thought.
Occupation
Alice normally lives in the Forest of Magic and rarely leaves it.
She generally has little interest in the activities of other youkai and usually does not involve herself in incidents. However, if an incident appears to be going unresolved, she may investigate it herself.
Alice is primarily a doll maker and magician.
She has stated that she does not normally sell her dolls and does not consider herself a public performer. Because she creates her own dolls, she also has little reason to purchase them.
Other accounts describe her performing doll shows for crowds during festivals, suggesting that she occasionally presents her puppetry publicly.
She has also been seen outside the Forest of Magic, including in the Human Village.
Possessions
Dolls
Alice owns a very large collection of dolls of many different types.
Because the Forest of Magic is humid, the dolls require frequent maintenance to prevent damage.
Many of Alice's spell cards name dolls after locations or nationalities.
Notable dolls include:
Shanghai Doll
A named doll capable of firing a straight, piercing laser.
Hourai Doll
A named doll that uses a stronger version of Shanghai Doll's attack.
Goliath Doll
A particularly large doll created by Alice. It was still undergoing testing when it appeared.
Grimoire
Alice possesses a magical book or grimoire.
In Mystic Square, she carried a black book labeled "Grimoire of Alice" and claimed that it contained ultimate magic.
In later appearances, she is usually shown carrying an unlabeled black-covered book.
Alice considers the book valuable and has reacted strongly when she believed others were attempting to steal it.
Species
Alice is a magician as a species rather than merely a human who practices magic.
She was originally human and later became a magician youkai through magic associated with abandoning the need for food.
Her transformation into a magician is relatively recent.
Because she spent much of her life as a human, she still retains human habits such as eating and sleeping even though these activities are no longer strictly necessary.
Core Traits
Species: Magician youkai, formerly human
Residence: Forest of Magic
Primary ability: Magic and doll manipulation
Specialization: Puppetry, magical threads, simultaneous control of many dolls
Combat style: Strategic, technical, indirect, doll-based
Temperament: Solitary, reserved, confident but somewhat timid
Attitude toward humans: Generally peaceful and hospitable
Interests: Magic, magical items, books, doll making
Primary research goal: Creation of a fully autonomous doll
Major weakness: Heavy concentration on doll control leaves Alice herself comparatively vulnerable
Notable possessions: Large doll collection, Shanghai Doll, Hourai Doll, Goliath Doll, magical grimoire"""]

def export_test_cases(model, tokenizer, fp):
    ctx = Context("test_case")
    with TensorWriter(fp) as writer:
        assert next(model.parameters()).device.type == "cpu"
        assert next(model.parameters()).dtype == torch.float32
        for idx, sentence in enumerate(TEST_SENTENCES):
            input_ids = tokenizer(sentence, return_tensors="pt").input_ids
            with torch.inference_mode():
                logits_cpu_fp32 = model(input_ids).logits
            assert logits_cpu_fp32.device.type == "cpu"
            assert logits_cpu_fp32.dtype == torch.float32

            case_ctx = ctx.with_subname(str(idx)).with_quant(Quant.NONE)
            writer.write_tensor(case_ctx.with_subname("input_ids"), input_ids[0])
            writer.write_tensor(
                case_ctx.with_subname("logits_cpu_fp32"),
                logits_cpu_fp32[0],
                preserve_dtype=True)
            del logits_cpu_fp32

        model.to(device="cuda", dtype=torch.float16)
        assert next(model.parameters()).device.type == "cuda"
        assert next(model.parameters()).dtype == torch.float16
        for idx, sentence in enumerate(TEST_SENTENCES):
            input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to("cuda")
            with torch.inference_mode():
                logits_cuda_fp16 = model(input_ids).logits
            assert logits_cuda_fp16.device.type == "cuda"
            assert logits_cuda_fp16.dtype == torch.float16

            case_ctx = ctx.with_subname(str(idx)).with_quant(Quant.NONE)
            writer.write_tensor(
                case_ctx.with_subname("logits_cuda_fp16"),
                logits_cuda_fp16[0])
            del logits_cuda_fp16

if __name__ == '__main__':
    from transformers import AutoTokenizer
    from transformers.models.llama import LlamaForCausalLM


    parser = argparse.ArgumentParser(description='export llama model from huggingface to libllm format.')
    parser.add_argument('-huggingface_name', type=str, help='the llama model name in huggingface.', default=MODEL_NAME)
    parser.add_argument('-output', type=str, help='output file name.', default="llama.llmpkg")
    parser.add_argument(
        '-test_output',
        type=str,
        help='output file name of the end-to-end test cases.',
        default="llama3.2-3b-instruct-fp16_test.llmpkg")
    parser.add_argument('-llama_version', type=int, help='llama model version.', default=3)
    parser.add_argument('-run', action="store_true")
    args = parser.parse_args()

    if args.run:
        run_llama_chat(args.huggingface_name)
        sys.exit(0)

    tokenizer = AutoTokenizer.from_pretrained(args.huggingface_name, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(
        args.huggingface_name,
        trust_remote_code=True,
        torch_dtype=torch.float32)
    model = model.eval()

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as package:
        if args.llama_version == 3:
            libllm_tokenizer = read_transformers_fast_bpe_model(args.huggingface_name)
        else:
            libllm_tokenizer = read_spm_model(args.huggingface_name)

        with package.open(MODEL_BIN, "w", force_zip64=True) as fp:
            config = LlamaExporter.export(model, fp)

        if args.llama_version == 3:
            config["llama"]["bot_token_id"] = str(tokenizer.bos_token_id)
            config["llama"]["eot_token_id"] = str(tokenizer.eos_token_id)
            config["llama"]["eot_id"] = str(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        with package.open(MODEL_INI, "w", force_zip64=True) as fp:
            config.write(io.TextIOWrapper(fp))

        with package.open(TOKENIZER_BIN, "w", force_zip64=True) as fp:
            libllm_tokenizer.save(fp)
        
        with package.open(TOKENIZER_INI, "w", force_zip64=True) as fp:
            libllm_tokenizer.get_config().to_ini(TOKENIZER_BIN).write(io.TextIOWrapper(fp))

    with zipfile.ZipFile(args.test_output, "w", compression=zipfile.ZIP_STORED) as package:
        with package.open(TEST_CASE_BIN, "w", force_zip64=True) as fp:
            export_test_cases(model, tokenizer, fp)
