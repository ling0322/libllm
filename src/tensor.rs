use crate::{lten, Error, Result};
use std::fmt;
use std::{ffi::c_void, ptr, rc::Rc};

pub struct Shape {
    size: Vec<i64>,
}

impl From<(i64, i64)> for Shape {
    fn from(t: (i64, i64)) -> Self {
        Shape {
            size: vec![t.0, t.1],
        }
    }
}

impl From<(i64, i64, i64)> for Shape {
    fn from(t: (i64, i64, i64)) -> Self {
        Shape {
            size: vec![t.0, t.1, t.2],
        }
    }
}

impl From<(i64, i64, i64, i64)> for Shape {
    fn from(t: (i64, i64, i64, i64)) -> Self {
        Shape {
            size: vec![t.0, t.1, t.2, t.3],
        }
    }
}

impl Shape {
    pub(crate) fn as_ptr(&self) -> *const i64 {
        self.size.as_ptr()
    }

    pub(crate) fn dim(&self) -> i32 {
        self.size.len() as i32
    }
}

pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    pub fn to_lynn(&self) -> i32 {
        match self {
            Self::Cpu => lten::Device::Cpu as i32,
            Self::Cuda => lten::Device::Cuda as i32,
            _ => panic!("unknown device"),
        }
    }
}

pub enum DType {
    Float,
    Int64,
    UInt8,
    Float16,
    QInt4,
    Int8,
    Unknown,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DType::Float => f.write_str("float32"),
            DType::Int64 => f.write_str("int64"),
            DType::UInt8 => f.write_str("uint8"),
            DType::Float16 => f.write_str("float16"),
            DType::QInt4 => f.write_str("qint4x32"),
            DType::Int8 => f.write_str("int8"),
            DType::Unknown => f.write_str("unknown"),
        }
    }
}

impl DType {
    pub fn is_float(&self) -> bool {
        return match self {
            DType::Float => true,
            DType::Float16 => true,
            _ => false,
        };
    }

    pub fn to_lynn(&self) -> i32 {
        match self {
            Self::Float => lten::Dtype::Float as i32,
            Self::Int64 => lten::Dtype::Int64 as i32,
            Self::UInt8 => lten::Dtype::Uint8 as i32,
            Self::Float16 => lten::Dtype::Float16 as i32,
            Self::QInt4 => lten::Dtype::QInt4 as i32,
            Self::Int8 => lten::Dtype::Int8 as i32,
            _ => panic!("unknown device"),
        }
    }
}

pub struct Tensor {
    tensorp: lten::LTensorPtr,
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.tensorp.is_null() {
            return;
        }

        let retcode = unsafe { lten::lten_destroy_tensor(self.tensorp) };
        if retcode != 0 {
            eprintln!(
                "an error occured when dropping a tensor: {}",
                lten::last_error_string()
            );
        }

        self.tensorp = ptr::null_mut();
    }
}

impl Tensor {
    pub fn new<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Result<Tensor> {
        let dtypel = dtype.to_lynn();
        let devicel = device.to_lynn();
        let s = shape.into();

        let tensorp = unsafe { lten::lten_new_tensor(s.dim(), s.as_ptr(), dtypel, devicel) };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
        }
    }

    pub fn print(&self) {
        let retcode = unsafe { lten::lten_print(self.tensorp) };
        if retcode != 0 {
            eprintln!(
                "an error occured when dropping a tensor: {}",
                lten::last_error_string()
            );
        }
    }
}
