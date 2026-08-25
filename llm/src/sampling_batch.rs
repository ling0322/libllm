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

//! Choosing the next token for every sequence in a batch at once.
//!
//! Sampling parameters belong to a request rather than to the model, so a batch that mixes
//! requests samples each row of the logits under its own temperature, top-k and top-p. The rows
//! are named by index because a batch samples only the sequences that are ready for it, which is
//! not always all of them.

use crate::flint::{functional as F, Device, Tensor};

use crate::error::{Error, Result};

/// The sampling parameters of every sequence in one forward pass.
pub struct SamplingBatch {
    sequence_indices: Vec<i64>,
    temperatures: Vec<f32>,
    top_ks: Vec<i32>,
    top_ps: Vec<f32>,
}

impl SamplingBatch {
    /// `sequence_indices` names the rows of the logits to sample, and the other three give each
    /// of those rows its parameters.
    pub fn new(
        sequence_indices: Vec<i64>,
        temperatures: Vec<f32>,
        top_ks: Vec<i32>,
        top_ps: Vec<f32>,
    ) -> Result<SamplingBatch> {
        let size = sequence_indices.len();
        if temperatures.len() != size || top_ks.len() != size || top_ps.len() != size {
            return Err(Error::model(
                "every sequence being sampled needs its own temperature, top-k and top-p",
            ));
        }

        Ok(SamplingBatch {
            sequence_indices,
            temperatures,
            top_ks,
            top_ps,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.sequence_indices.is_empty()
    }

    pub fn len(&self) -> usize {
        self.sequence_indices.len()
    }

    /// The rows this batch samples, in the order it samples them.
    pub fn sequence_indices(&self) -> &[i64] {
        &self.sequence_indices
    }

    /// Move the parameters onto `device`, ready to sample there.
    pub fn prepare(&self, device: Device) -> Result<PreparedSampling> {
        if self.is_empty() {
            return Err(Error::model(
                "a sampling batch with no sequences samples nothing",
            ));
        }

        let size = self.len() as i32;
        let to_device = |tensor: Tensor| -> Result<Tensor> { Ok(tensor.to_device(device)?) };

        Ok(PreparedSampling {
            sequence_indices: to_device(Tensor::from_i64(&[size], &self.sequence_indices)?)?,
            temperatures: to_device(Tensor::from_f32(&[size], &self.temperatures)?)?,
            top_ks: to_device(Tensor::from_i32(&[size], &self.top_ks)?)?,
            top_ps: to_device(Tensor::from_f32(&[size], &self.top_ps)?)?,
        })
    }
}

/// A sampling batch with its parameters on the device, ready to run.
#[derive(Debug)]
pub struct PreparedSampling {
    sequence_indices: Tensor,
    temperatures: Tensor,
    top_ks: Tensor,
    top_ps: Tensor,
}

impl PreparedSampling {
    /// The token drawn for each of this batch's rows of `logits` `<float>(rows, vocabSize)`,
    /// as `<long>(sequences)`.
    pub fn sample(&self, logits: &Tensor) -> Result<Tensor> {
        let dim = logits.dim()?;
        if dim != 2 {
            return Err(Error::model(format!(
                "sampling takes <float>(rows, vocabSize), got a {dim}-D tensor"
            )));
        }

        let rows = F::lookup(logits, &self.sequence_indices)?;
        Ok(F::sample_with_params(
            &rows,
            &self.temperatures,
            &self.top_ks,
            &self.top_ps,
        )?)
    }
}
