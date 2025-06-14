use crate::{lten, DType, Device, Result, Tensor};
use std::ptr;

pub struct F {}

impl F {
    pub fn copy(dest: &mut Tensor, src: &Tensor) -> Result<()> {
        let retcode = unsafe { lten::lten_copy(dest.tensorp, src.tensorp) };
        if retcode != 0 {
            Err(lten::last_error())
        } else {
            Ok(())
        }
    }

    pub fn fill(tensor: &mut Tensor, value: f32) -> Result<()> {
        let retcode = unsafe { lten::lten_fill_float(tensor.tensorp, value) };
        if retcode != 0 {
            Err(lten::last_error())
        } else {
            Ok(())
        }
    }

    pub fn print(tensor: &Tensor) {
        let retcode = unsafe { lten::lten_print(tensor.tensorp) };
        if retcode != 0 {
            eprintln!(
                "an error occured when dropping a tensor: {}",
                lten::last_error_string()
            );
        }
    }

    pub fn to(device: Device, tensor: &Tensor) -> Result<Tensor> {
        let devicel = device.to_lten();
        let tensorp = unsafe { lten::lten_to_device(tensor.tensorp, devicel) };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
        }
    }

    pub fn cast(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
        let dtypel = dtype.to_lten();
        let tensorp = unsafe { lten::lten_to_dtype(tensor.tensorp, dtypel) };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
        }
    }

    pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        Self::apply_op(
            lhs,
            Some(rhs),
            None,
            None,
            0,
            0,
            0.0,
            0.0,
            lten::OPERATOR_ADD,
        )
    }

    pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        Self::apply_op(
            lhs,
            Some(rhs),
            None,
            None,
            0,
            0,
            0.0,
            0.0,
            lten::OPERATOR_MUL,
        )
    }

    pub fn apply_rope(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        Self::apply_op(
            lhs,
            Some(rhs),
            None,
            None,
            0,
            0,
            0.0,
            0.0,
            lten::OPERATOR_ROPE,
        )
    }

    pub fn softmax(tensor: &Tensor) -> Result<Tensor> {
        Self::apply_op(
            tensor,
            None,
            None,
            None,
            0,
            0,
            0.0,
            0.0,
            lten::OPERATOR_SOFTMAX,
        )
    }

    pub fn gelu(tensor: &Tensor) -> Result<Tensor> {
        Self::apply_op(
            tensor,
            None,
            None,
            None,
            0,
            0,
            0.0,
            0.0,
            lten::OPERATOR_GELU,
        )
    }

    pub fn swiglu(tensor: &Tensor) -> Result<Tensor> {
        Self::apply_op(
            tensor,
            None,
            None,
            None,
            0,
            0,
            0.0,
            0.0,
            lten::OPERATOR_SWIGLU,
        )
    }

    pub fn contiguous(tensor: &Tensor) -> Result<Tensor> {
        Self::apply_op(
            tensor,
            None,
            None,
            None,
            0,
            0,
            0.0,
            0.0,
            lten::OPERATOR_CONTIGUOUS,
        )
    }

    pub fn sum(tensor: &Tensor, dim: i64) -> Result<Tensor> {
        Self::apply_op(
            tensor,
            None,
            None,
            None,
            dim,
            0,
            0.0,
            0.0,
            lten::OPERATOR_SUM,
        )
    }

    pub fn max(tensor: &Tensor, dim: i64) -> Result<Tensor> {
        Self::apply_op(
            tensor,
            None,
            None,
            None,
            dim,
            0,
            0.0,
            0.0,
            lten::OPERATOR_MAX,
        )
    }

    pub fn matmul(tensor: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        Self::apply_op(
            tensor,
            Some(rhs),
            None,
            None,
            0,
            0,
            0.0,
            0.0,
            lten::OPERATOR_MATMUL,
        )
    }

    pub fn lookup(tensor: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        Self::apply_op(
            tensor,
            Some(rhs),
            None,
            None,
            0,
            0,
            0.0,
            0.0,
            lten::OPERATOR_LOOKUP,
        )
    }

    pub fn scalar_mul(tensor: &Tensor, rhs: f32) -> Result<Tensor> {
        Self::apply_op(
            tensor,
            None,
            None,
            None,
            0,
            0,
            rhs,
            0.0,
            lten::OPERATOR_SCALAR_MUL,
        )
    }

    pub fn layer_norm(tensor: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        Self::apply_op(
            tensor,
            Some(weight),
            Some(bias),
            None,
            0,
            0,
            eps,
            0.0,
            lten::OPERATOR_LAYER_NORM,
        )
    }

    pub fn rms_norm(tensor: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        Self::apply_op(
            tensor,
            Some(weight),
            None,
            None,
            0,
            0,
            eps,
            0.0,
            lten::OPERATOR_RMS_NORM,
        )
    }

    fn apply_op(
        targ0: &Tensor,
        targ1: Option<&Tensor>,
        targ2: Option<&Tensor>,
        targ3: Option<&Tensor>,
        iarg0: i64,
        iarg1: i64,
        farg0: f32,
        farg1: f32,
        op: i32,
    ) -> Result<Tensor> {
        let tensorp = unsafe {
            lten::lten_apply_operator(
                targ0.tensorp,
                match targ1 {
                    None => ptr::null_mut(),
                    Some(t) => t.tensorp,
                },
                match targ2 {
                    None => ptr::null_mut(),
                    Some(t) => t.tensorp,
                },
                match targ3 {
                    None => ptr::null_mut(),
                    Some(t) => t.tensorp,
                },
                iarg0,
                iarg1,
                farg0,
                farg1,
                op,
            )
        };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
        }
    }
}
