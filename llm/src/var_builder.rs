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

//! Reading the parameters of a model.
//!
//! A package holds its tensors in one `tdicv2` stream: a flat table of name to tensor, written
//! with the shape and the element type of each. A [`VarBuilder`] reads that table once and then
//! hands out views of it, one per module, so that a layer asks for `"weight"` and gets the tensor
//! the whole name of which is `"model.layer0.attn.weight"`.

use std::collections::HashMap;
use std::fmt;
use std::io::Read;
use std::rc::Rc;

use crate::flint::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::reader::BinaryRead;

/// The header every parameter file starts with.
const MAGIC: &str = "llyn::tdicv2    ";

/// The magic number that closes a tensor's data, as a check that it was read to the right place.
const DATA_MAGIC: i16 = 0x55aa;

/// The largest rank and dimension a stored tensor may have, which keep a corrupt file from asking
/// for an unreasonable allocation.
const MAX_RANK: i16 = 16;
const MAX_DIM: i32 = 1048576;

/// The parameters of a model, and where in them this builder points.
///
/// Cloning a builder or narrowing it with [`VarBuilder::with_name`] shares the parameters rather
/// than copying them.
#[derive(Clone)]
pub struct VarBuilder {
    params: Rc<HashMap<String, Tensor>>,
    namespace: String,
    device: Device,
    float_type: DType,
}

impl VarBuilder {
    /// Read a parameter file, moving what it holds onto `device` as each tensor is asked for.
    ///
    /// Float tensors are cast to `float_type` on the way out, so that a file written in one
    /// precision can drive a device that works in another.
    pub fn from_reader(
        reader: &mut impl Read,
        device: Device,
        float_type: DType,
    ) -> Result<VarBuilder> {
        let mut params = HashMap::new();

        reader.expect_tag(MAGIC)?;
        reader.expect_tag("<d> ")?;

        let mut tag = reader.read_string(4)?;
        while tag != "</d>" {
            if tag != "<r> " {
                return Err(Error::format(format!(
                    "expected a record or the end of the parameters, got {tag:?}"
                )));
            }

            let (name, tensor) = read_named_tensor(reader)?;
            params.insert(name, tensor);

            reader.expect_tag("</r>")?;
            tag = reader.read_string(4)?;
        }

        Ok(VarBuilder {
            params: Rc::new(params),
            namespace: String::new(),
            device,
            float_type,
        })
    }

    /// A builder pointing at the `name` sub-namespace of this one.
    pub fn with_name(&self, name: &str) -> VarBuilder {
        let mut child = self.clone();
        child.namespace = self.name_of(name);
        child
    }

    /// The tensor called `name` here, checked against the shape the caller expects.
    pub fn get(&self, name: &str, shape: &[i32]) -> Result<Tensor> {
        let tensor = self.get_unchecked(name)?;
        if tensor.shape() != shape {
            return Err(Error::model(format!(
                "tensor {:?} has shape {:?}, expected {:?}",
                self.name_of(name),
                tensor.shape(),
                shape
            )));
        }
        Ok(tensor)
    }

    /// The tensor called `name` here, whatever shape it turns out to have.
    pub fn get_unchecked(&self, name: &str) -> Result<Tensor> {
        let full_name = self.name_of(name);
        let tensor = self
            .params
            .get(&full_name)
            .ok_or_else(|| Error::model(format!("tensor {full_name:?} not found in model")))?;

        let tensor = tensor.to_device(self.device)?;
        if tensor.dtype() == DType::Float || tensor.dtype() == DType::Float16 {
            Ok(tensor.cast(self.float_type)?)
        } else {
            Ok(tensor)
        }
    }

    /// Whether a tensor called `name` is here.
    pub fn has(&self, name: &str) -> bool {
        self.params.contains_key(&self.name_of(name))
    }

    /// The whole name of `name` in this namespace.
    pub fn name_of(&self, name: &str) -> String {
        if self.namespace.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.namespace, name)
        }
    }

    /// The name of this namespace itself, for a message about what a module was missing.
    pub fn name(&self) -> &str {
        &self.namespace
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn float_type(&self) -> DType {
        self.float_type
    }

    /// The whole names of every tensor the file held, in order. For finding out what a package
    /// actually calls things when a model fails to find what it expected.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.params.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// How many tensors the file held.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

impl fmt::Debug for VarBuilder {
    /// Names the namespace and how much is in the file, rather than every tensor in it: a model
    /// holds hundreds, and this is read in the middle of an error message.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VarBuilder")
            .field("namespace", &self.namespace)
            .field("tensors", &self.params.len())
            .field("device", &self.device)
            .field("float_type", &self.float_type)
            .finish()
    }
}

/// Reads one `name -> tensor` record.
fn read_named_tensor(reader: &mut impl Read) -> Result<(String, Tensor)> {
    let name_length = reader.read_i16()?;
    if name_length <= 0 {
        return Err(Error::format("tensor name is empty"));
    }

    let name = reader.read_string(name_length as usize)?;
    let tensor = read_tensor(reader)?;
    Ok((name, tensor))
}

/// Reads one tensor: its shape, then the single slot of data under it.
fn read_tensor(reader: &mut impl Read) -> Result<Tensor> {
    reader.expect_tag("tnsr")?;

    let rank = reader.read_i16()?;
    if !(0..=MAX_RANK).contains(&rank) {
        return Err(Error::format(format!("tensor rank {rank} is out of range")));
    }

    let mut shape = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        let size = reader.read_i32()?;
        if size <= 0 || size >= MAX_DIM {
            return Err(Error::format(format!(
                "tensor dimension {size} is out of range"
            )));
        }
        shape.push(size);
    }

    reader.expect_tag("tdat")?;
    let slots = reader.read_i32()?;
    if slots != 1 {
        return Err(Error::format(format!(
            "tensor holds {slots} data slots, expected 1"
        )));
    }

    let dtype = DType::from_code(reader.read_i16()? as i32)
        .map_err(|error| Error::format(format!("tensor has an unknown element type: {error}")))?;
    let numel = reader.read_i64()?;
    let expected: i64 = shape.iter().map(|&size| size as i64).product();
    if numel != expected {
        return Err(Error::format(format!(
            "tensor holds {numel} elements but its shape calls for {expected}"
        )));
    }

    let data = reader.read_exact_bytes(dtype.total_size(numel) as usize)?;
    let tensor = Tensor::from_bytes(&shape, dtype, &data)?;

    if reader.read_i16()? != DATA_MAGIC {
        return Err(Error::format(
            "tensor data did not end where it should have",
        ));
    }

    Ok(tensor)
}
