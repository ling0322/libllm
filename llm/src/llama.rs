// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! The Llama model.
//!
//! The stack is the usual one: an embedding, a run of decode layers each with grouped-query
//! attention and a SwiGLU feed-forward, a final norm, and a projection to the vocabulary. What is
//! particular to this implementation is that attention reads and writes a paged KV cache, so a
//! layer never sees a contiguous per-sequence cache: it writes the keys of the tokens it was just
//! given into the slots the batch names, then attends over the blocks the batch lists.

use crate::flint::{functional as F, DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::forward_batch::PreparedBatch;
use crate::ini::IniSection;
use crate::kv_cache::{KVCacheManager, KVCacheSpec};
use crate::layers::{Embedding, Linear, RmsNorm};
use crate::prompt::{Message, Prompt};
use crate::tokenizer::Tokenizer;
use crate::var_builder::VarBuilder;
use crate::zip_file::ZipFile;

/// The shape of one Llama model, as its package's configuration gives it.
#[derive(Clone, Copy, Debug)]
pub struct LlamaConfig {
    pub hidden_size: i32,
    pub num_heads: i32,
    /// Fewer than `num_heads` under grouped-query attention, which is what lets the cache hold
    /// fewer keys than there are query heads.
    pub num_key_value_heads: i32,
    pub intermediate_size: i32,
    pub norm_eps: f32,
    pub num_layers: i32,
    pub vocab_size: i32,
    pub max_context_length: i32,
    pub qkv_proj_bias: bool,
}

impl LlamaConfig {
    /// Read the configuration out of the section named after the model type.
    pub fn from_section(section: &IniSection) -> Result<LlamaConfig> {
        let num_heads: i32 = section.get("num_heads")?;

        Ok(LlamaConfig {
            hidden_size: section.get("hidden_size")?,
            num_heads,
            // A model that does not say otherwise gives every query head its own key head.
            num_key_value_heads: section.get_or("num_key_value_heads", num_heads)?,
            intermediate_size: section.get("intermediate_size")?,
            norm_eps: section.get("norm_eps")?,
            num_layers: section.get("num_layers")?,
            vocab_size: section.get("vocab_size")?,
            max_context_length: section.get("max_ctx_length")?,
            qkv_proj_bias: section.get_bool_or("qkv_proj_bias", false)?,
        })
    }

    /// The width of one attention head.
    pub fn head_dim(&self) -> Result<i32> {
        if self.num_heads <= 0 || self.hidden_size % self.num_heads != 0 {
            return Err(Error::model(format!(
                "hidden_size {} does not divide into {} heads",
                self.hidden_size, self.num_heads
            )));
        }
        Ok(self.hidden_size / self.num_heads)
    }
}

/// The feed-forward network of one decode layer.
///
/// The gate and the up projection are stored as one matrix and split by the SwiGLU, which halves
/// the number of matrix multiplications the layer does.
#[derive(Debug)]
struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn build(config: &LlamaConfig, vb: &VarBuilder) -> Result<Mlp> {
        let (d, di) = (config.hidden_size, config.intermediate_size);
        Ok(Mlp {
            gate_up_proj: Linear::build(d, di * 2, false, &vb.with_name("gate_up_proj"))?,
            down_proj: Linear::build(di, d, false, &vb.with_name("down_proj"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.gate_up_proj.forward(input)?;
        let x = F::swiglu(&x)?;
        self.down_proj.forward(&x)
    }
}

/// Grouped-query attention over a paged KV cache.
#[derive(Debug)]
struct Attention {
    qkv_proj: Linear,
    out_proj: Linear,
    /// The cosines and sines of every position, shared by every layer.
    rotary_cache: Tensor,
    hidden_size: i32,
    num_head: i32,
    num_key_value_head: i32,
    head_dim: i32,
    /// Which pool of the cache this layer reads and writes.
    layer_index: i32,
}

impl Attention {
    fn build(
        config: &LlamaConfig,
        vb: &VarBuilder,
        rotary_cache: Tensor,
        layer_index: i32,
    ) -> Result<Attention> {
        let head_dim = config.head_dim()?;
        let d = config.hidden_size;
        // Queries, keys and values are projected by one matrix and split apart afterwards.
        let qkv_proj_dim = head_dim * config.num_key_value_heads * 2 + d;

        Ok(Attention {
            qkv_proj: Linear::build(
                d,
                qkv_proj_dim,
                config.qkv_proj_bias,
                &vb.with_name("qkv_proj"),
            )?,
            out_proj: Linear::build(d, d, false, &vb.with_name("out_proj"))?,
            rotary_cache,
            hidden_size: d,
            num_head: config.num_heads,
            num_key_value_head: config.num_key_value_heads,
            head_dim,
            layer_index,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        batch: &PreparedBatch,
        cache: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let dim = input.dim()?;
        if dim != 2 {
            return Err(Error::model(format!(
                "attention takes a packed <float>(totalTokens, hidden), got a {dim}-D tensor"
            )));
        }

        let qkv = self.qkv_proj.forward(input)?;
        let q_len = batch.total_q_len();
        if qkv.shape_at(0)? != q_len {
            return Err(Error::model(format!(
                "batch describes {q_len} packed tokens but the activation holds {}",
                qkv.shape_at(0)?
            )));
        }

        let kv_hidden_size = self.head_dim * self.num_key_value_head;
        let mut q =
            qkv.slice(-1, 0, self.hidden_size)?
                .view(&[q_len, self.num_head, self.head_dim])?;
        let mut k = qkv
            .slice(-1, self.hidden_size, self.hidden_size + kv_hidden_size)?
            .view(&[q_len, self.num_key_value_head, self.head_dim])?;
        let v = qkv
            .slice(
                -1,
                self.hidden_size + kv_hidden_size,
                self.hidden_size + 2 * kv_hidden_size,
            )?
            .view(&[q_len, self.num_key_value_head, self.head_dim])?;

        F::rotary_embedding(batch.position_ids(), &mut q, &mut k, &self.rotary_cache)?;

        let (max_q_len, max_k_len) = (batch.max_q_len(), batch.max_k_len());
        let (key_cache, value_cache) = cache.caches_mut(self.layer_index)?;

        // The keys of this batch have to be in the pool before attention reads them: under a
        // causal mask a query attends to its own key.
        F::store_kv_cache(&k, &v, key_cache, value_cache, batch.slot_mapping())?;

        let x = F::paged_attention(
            &q,
            &F::PagedKvCache {
                key_cache,
                value_cache,
                block_table: batch.block_table(),
                cu_seqlens_q: batch.cu_seqlens_q(),
                seqlens_k: batch.seqlens_k(),
                max_q_len,
                max_k_len,
            },
            true,
        )?;

        let x = x.contiguous()?.view(&[q_len, self.hidden_size])?;
        self.out_proj.forward(&x)
    }
}

/// One decode layer: attention and feed-forward, each around a residual connection.
#[derive(Debug)]
struct DecodeLayer {
    input_norm: RmsNorm,
    attn: Attention,
    post_attn_norm: RmsNorm,
    mlp: Mlp,
}

impl DecodeLayer {
    fn build(
        config: &LlamaConfig,
        vb: &VarBuilder,
        rotary_cache: Tensor,
        layer_index: i32,
    ) -> Result<DecodeLayer> {
        Ok(DecodeLayer {
            input_norm: RmsNorm::build(
                config.hidden_size,
                config.norm_eps,
                &vb.with_name("input_norm"),
            )?,
            attn: Attention::build(config, &vb.with_name("attn"), rotary_cache, layer_index)?,
            post_attn_norm: RmsNorm::build(
                config.hidden_size,
                config.norm_eps,
                &vb.with_name("post_attn_norm"),
            )?,
            mlp: Mlp::build(config, &vb.with_name("mlp"))?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        batch: &PreparedBatch,
        cache: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let x = self.input_norm.forward(input)?;
        let x = self.attn.forward(&x, batch, cache)?;
        let x = F::add(&x, input)?;

        let residual = x;
        let x = self.post_attn_norm.forward(&residual)?;
        let x = self.mlp.forward(&x)?;
        Ok(F::add(&x, &residual)?)
    }
}

/// The transformer: everything from token ids to the hidden state, plus the projection that turns
/// that hidden state into logits.
#[derive(Debug)]
pub struct LlamaModel {
    config: LlamaConfig,
    embedding: Embedding,
    layers: Vec<DecodeLayer>,
    norm: RmsNorm,
    out_proj: Linear,
}

impl LlamaModel {
    /// Build the model from the parameters under `vb`, which points at the model's own namespace.
    pub fn build(config: LlamaConfig, vb: &VarBuilder) -> Result<LlamaModel> {
        let d = config.hidden_size;
        let head_dim = config.head_dim()?;

        // Stored as a cosine half and a sine half of every position; the kernel wants the two
        // interleaved per position, which is what the transpose does.
        let rotary_cache = vb.get("rope", &[2, 1, config.max_context_length, head_dim])?;
        let rotary_cache = rotary_cache
            .view(&[2, config.max_context_length, head_dim])?
            .transpose(0, 1)?
            .contiguous()?
            .view(&[config.max_context_length, 2 * head_dim])?;

        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            layers.push(DecodeLayer::build(
                &config,
                &vb.with_name(&format!("block{i}")),
                rotary_cache.clone(),
                i,
            )?);
        }

        Ok(LlamaModel {
            embedding: Embedding::build(d, config.vocab_size, &vb.with_name("embd"))?,
            layers,
            norm: RmsNorm::build(d, config.norm_eps, &vb.with_name("norm"))?,
            out_proj: Linear::build(d, config.vocab_size, false, &vb.with_name("out_proj"))?,
            config,
        })
    }

    /// The hidden state of every packed token, `<float>(totalQLen, hidden)`.
    pub fn forward(
        &self,
        input: &Tensor,
        batch: &PreparedBatch,
        cache: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut x = self.embedding.forward(input)?;
        for layer in &self.layers {
            x = layer.forward(&x, batch, cache)?;
        }
        self.norm.forward(&x)
    }

    /// The logits of a hidden state, `<float>(rows, vocabSize)`.
    pub fn forward_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        self.out_proj.forward(hidden)
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    pub fn output_dim(&self) -> i32 {
        self.config.vocab_size
    }
}

impl crate::model::ModelForGeneration for LlamaForGeneration {
    fn forward(&self, batch: &PreparedBatch, cache: &mut KVCacheManager) -> Result<Tensor> {
        LlamaForGeneration::forward(self, batch, cache)
    }

    fn is_stop_token(&self, token_id: i32) -> bool {
        LlamaForGeneration::is_stop_token(self, token_id)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn device(&self) -> Device {
        self.device
    }

    fn output_dim(&self) -> i32 {
        self.model.output_dim()
    }

    fn kv_cache_spec(&self) -> Result<KVCacheSpec> {
        LlamaForGeneration::kv_cache_spec(self)
    }

    fn build_prompt(&self, history: &[Message]) -> Result<Prompt> {
        LlamaForGeneration::build_prompt(self, history)
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

/// A Llama model read from a package, ready to generate.
#[derive(Debug)]
pub struct LlamaForGeneration {
    model: LlamaModel,
    tokenizer: Tokenizer,
    name: String,
    device: Device,
    float_type: DType,
    eot_id: i32,
}

impl LlamaForGeneration {
    /// The section of `model.ini` that says what the package holds.
    pub const MODEL_SECTION: &'static str = "model";

    /// Read the model out of `package`, onto `device`.
    pub fn from_package(device: Device, package: &ZipFile) -> Result<LlamaForGeneration> {
        let config = crate::ini::IniConfig::parse(&package.read_to_string(crate::MODEL_CONFIG)?)?;
        let model_section = config.section(Self::MODEL_SECTION)?;
        let model_type = model_section.get_str("type")?.to_string();
        let model_file = model_section.get_str("model_file")?.to_string();

        let section = config.section(&model_type)?;
        let llama_config = LlamaConfig::from_section(section)?;
        let eot_id = section.get("eot_token_id")?;

        // The weights are cast to whatever the device works in as they are read.
        let float_type = F::default_float_type(device)?;
        let vb =
            VarBuilder::from_reader(&mut package.open_entry(&model_file)?, device, float_type)?;

        Ok(LlamaForGeneration {
            model: LlamaModel::build(llama_config, &vb.with_name(&model_type))?,
            tokenizer: Tokenizer::from_package(package)?,
            name: model_type,
            device,
            float_type,
            eot_id,
        })
    }

    /// The logits of the next token of every sequence in `batch`, `<float>(numSequences, V)`.
    ///
    /// The batch carries the tokens; the cache is written as the pass goes, which is what lets the
    /// next call carry only the tokens after these.
    pub fn forward(&self, batch: &PreparedBatch, cache: &mut KVCacheManager) -> Result<Tensor> {
        let tokens = batch
            .token_ids()
            .ok_or_else(|| Error::model("batch carries no tokens to forward"))?;

        let hidden = self.model.forward(tokens, batch, cache)?;
        // Only the last query of each sequence says anything about what comes next.
        let last = F::lookup(&hidden, batch.last_query_indices())?;
        self.model.forward_lm_head(&last)
    }

    /// Lay a conversation out the way Llama 3 was trained to read it.
    ///
    /// Every turn is wrapped in header and end-of-turn control tokens. A history ending with a
    /// user turn is left open at an assistant header, which is the model's cue to answer; one
    /// ending with an assistant turn is left mid-turn, so the model carries that turn on.
    pub fn build_prompt(&self, history: &[Message]) -> Result<Prompt> {
        Self::build_prompt_from(history)
    }

    /// [`LlamaForGeneration::build_prompt`] without a model, since the layout depends on the
    /// conversation rather than on any of the weights.
    pub fn build_prompt_from(history: &[Message]) -> Result<Prompt> {
        let (last, earlier) = history
            .split_last()
            .ok_or_else(|| Error::model("a prompt needs at least one message"))?;

        let mut prompt = Prompt::new();
        prompt.append_control_token("<|begin_of_text|>");

        for message in earlier {
            Self::append_turn(&mut prompt, &message.role, &message.content);
        }

        match last.role.as_str() {
            "user" => {
                Self::append_turn(&mut prompt, &last.role, &last.content);
                prompt
                    .append_control_token("<|start_header_id|>")
                    .append_text("assistant")
                    .append_control_token("<|end_header_id|>")
                    .append_text("\n\n");
            }
            // The model continues the turn instead of starting a new one, so it is left open.
            "assistant" => {
                prompt
                    .append_control_token("<|start_header_id|>")
                    .append_text(&last.role)
                    .append_control_token("<|end_header_id|>")
                    .append_text(format!("\n\n{}", last.content));
            }
            other => {
                return Err(Error::model(format!(
                    "the last message is from {other:?}; it has to be the user or the assistant"
                )))
            }
        }

        Ok(prompt)
    }

    /// One complete turn, header and all.
    fn append_turn(prompt: &mut Prompt, role: &str, content: &str) {
        prompt
            .append_control_token("<|start_header_id|>")
            .append_text(role)
            .append_control_token("<|end_header_id|>")
            .append_text(format!("\n\n{content}"))
            .append_control_token("<|eot_id|>");
    }

    /// The tokens of a prompt.
    pub fn encode_prompt(&self, prompt: &Prompt) -> Result<Vec<i64>> {
        prompt.encode(&self.tokenizer)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Whether the model is done once it has produced `token_id`.
    pub fn is_stop_token(&self, token_id: i32) -> bool {
        token_id == self.eot_id
    }

    /// The cache layout this model needs.
    pub fn kv_cache_spec(&self) -> Result<KVCacheSpec> {
        Ok(KVCacheSpec {
            num_layers: self.model.config.num_layers,
            num_key_value_heads: self.model.config.num_key_value_heads,
            head_dim: self.model.config.head_dim()?,
            max_context_length: self.model.config.max_context_length,
            dtype: self.float_type,
        })
    }

    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.model.config
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn output_dim(&self) -> i32 {
        self.model.output_dim()
    }
}
