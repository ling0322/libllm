use crate::{operator::F, Error, Result, Tensor};
use std::{collections::HashMap, rc::Rc};

#[derive(Clone)]
pub struct Builder {
    tensor_map: Rc<HashMap<String, Tensor>>,
    name: String,
}

impl Builder {
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let tensor_name = self.name.clone() + "." + name;
        match self.tensor_map.get(&tensor_name) {
            Some(tensor) => Ok(tensor.clone()),
            None => Err(Error::TensorNotExistError),
        }
    }
}

pub struct LayerNorm {
    w: Tensor,
    b: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn from_builder(eps: f32, builder: &Builder) -> Result<Self> {
        let w = builder.get_tensor("weight")?;
        let b = builder.get_tensor("bias")?;

        return Ok(Self { w, b, eps });
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        F::layer_norm(x, &self.w, &self.b, self.eps)
    }
}

pub struct Embedding {
    w: Tensor,
}

impl Embedding {
    pub fn from_builder(builder: &Builder) -> Result<Self> {
        let w = builder.get_tensor("weight")?;
        return Ok(Self { w });
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        F::lookup(&self.w, x)
    }
}

pub struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    pub fn from_builder(has_bias: bool, builder: &Builder) -> Result<Self> {
        let w = builder.get_tensor("weight")?;
        let b = if has_bias {
            Some(builder.get_tensor("bias")?)
        } else {
            None
        };

        return Ok(Linear { w, b });
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        F::matmul(x, &self.w.transpose(0, 1)?)
    }
}

pub struct RmsNorm {
    w: Tensor,
    eps: f32,
}

impl RmsNorm {
    pub fn from_builder(eps: f32, builder: &Builder) -> Result<Self> {
        let w = builder.get_tensor("weight")?;
        return Ok(Self { w, eps });
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        F::rms_norm(x, &self.w, self.eps)
    }
}
