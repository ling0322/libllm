use crate::{lten, Error, Result};
use std::any::TypeId;
use std::fmt;
use std::ops::{Bound, RangeBounds};
use std::ptr;

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
    pub(crate) fn from_lten(devicel: i32) -> Device {
        match devicel {
            lten::DEVICE_CPU => Self::Cpu,
            lten::DEVICE_CUDA => Self::Cuda,
            _ => panic!("unknown device"),
        }
    }

    pub(crate) fn to_lten(&self) -> i32 {
        match self {
            Self::Cpu => lten::DEVICE_CPU,
            Self::Cuda => lten::DEVICE_CUDA,
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
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

    pub fn from_type<T: 'static>() -> Result<Self> {
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            Ok(DType::Float)
        } else if TypeId::of::<T>() == TypeId::of::<i64>() {
            Ok(DType::Int64)
        } else {
            Err(Error::InvalidDTypeError)
        }
    }

    pub(crate) fn from_lten(dtype: i32) -> Self {
        match dtype {
            lten::DTYPE_FLOAT => Self::Float,
            lten::DTYPE_INT64 => Self::Int64,
            lten::DTYPE_UINT8 => Self::UInt8,
            lten::DTYPE_FLOAT16 => Self::Float16,
            lten::DTYPE_QINT4 => Self::QInt4,
            lten::DTYPE_INT8 => Self::Int8,
            _ => panic!("unknown lten dtype: {}", dtype),
        }
    }

    pub(crate) fn to_lten(&self) -> i32 {
        match self {
            Self::Float => lten::DTYPE_FLOAT,
            Self::Int64 => lten::DTYPE_INT64,
            Self::UInt8 => lten::DTYPE_UINT8,
            Self::Float16 => lten::DTYPE_FLOAT16,
            Self::QInt4 => lten::DTYPE_QINT4,
            Self::Int8 => lten::DTYPE_UINT8,
            _ => panic!("unknown dtype"),
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
        let dtypel = dtype.to_lten();
        let devicel = device.to_lten();
        let s = shape.into();

        let tensorp = unsafe { lten::lten_new_tensor(s.dim(), s.as_ptr(), dtypel, devicel) };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
        }
    }

    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Result<Tensor> {
        let mut tensor = Self::new(shape, dtype, device)?;
        if dtype.is_float() {
            tensor.fill_float(0.0)?;
            Ok(tensor)
        } else {
            Err(Error::LtenError(
                "unsupported dtype for Tensor::zeros".to_string(),
            ))
        }
    }

    pub fn from_slice<S: Into<Shape>, T: 'static + Copy>(shape: S, data: &[T]) -> Result<Tensor> {
        let dtype = DType::from_type::<T>()?;
        let mut tensor = Self::new(shape, dtype, Device::Cpu)?;
        let numel = tensor.numel()?;

        if numel != data.len() {
            return Err(Error::LtenError(
                "shape and slice length mismatch".to_string(),
            ));
        }

        let tensor_data = tensor.data_mut::<T>()?;
        tensor_data.copy_from_slice(data);

        Ok(tensor)
    }

    pub fn dim(&self) -> Result<i64> {
        let mut d: i32 = 0;
        let retcode = unsafe { lten::lten_get_dim(self.tensorp, &mut d) };
        if retcode != 0 {
            Err(lten::last_error())
        } else {
            Ok(d as i64)
        }
    }

    pub fn shape(&self, dim: i64) -> Result<i64> {
        let mut size: i64 = 0;
        let retcode = unsafe { lten::lten_get_shape(self.tensorp, dim as i32, &mut size) };
        if retcode != 0 {
            Err(lten::last_error())
        } else {
            Ok(size)
        }
    }

    pub fn dtype(&self) -> Result<DType> {
        let mut dtypel: i32 = 0;
        let retcode = unsafe { lten::lten_get_dtype(self.tensorp, &mut dtypel) };
        if retcode != 0 {
            Err(lten::last_error())
        } else {
            Ok(DType::from_lten(dtypel))
        }
    }

    pub fn device(&self) -> Result<Device> {
        let mut devicel: i32 = 0;
        let retcode = unsafe { lten::lten_get_device(self.tensorp, &mut devicel) };
        if retcode != 0 {
            Err(lten::last_error())
        } else {
            Ok(Device::from_lten(devicel))
        }
    }

    pub fn numel(&self) -> Result<usize> {
        let mut numel: i64 = 0;
        let retcode = unsafe { lten::lten_get_numel(self.tensorp, &mut numel) };
        if retcode != 0 {
            Err(lten::last_error())
        } else {
            Ok(numel as usize)
        }
    }

    pub fn view<S: Into<Shape>>(&self, s: S) -> Result<Tensor> {
        let s: Shape = s.into();
        let tensorp = unsafe { lten::lten_view(self.tensorp, s.dim(), s.as_ptr()) };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
        }
    }

    pub fn expand<S: Into<Shape>>(&self, s: S) -> Result<Tensor> {
        let s: Shape = s.into();
        let tensorp = unsafe { lten::lten_expand(self.tensorp, s.dim(), s.as_ptr()) };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
        }
    }

    pub fn data_mut<'a, T: 'static>(&mut self) -> Result<&'a mut [T]>
    where
        Self: 'a,
    {
        let dtype = self.dtype()?;
        let t_dtype = DType::from_type::<T>()?;

        if dtype != t_dtype {
            Err(Error::InvalidDTypeError)
        } else {
            let p = unsafe { lten::lten_get_data_ptr(self.tensorp) as *mut T };
            Ok(unsafe { std::slice::from_raw_parts_mut::<'a, T>(p, self.numel()?) })
        }
    }

    pub fn data<'a, T: 'static>(&self) -> Result<&'a [T]>
    where
        Self: 'a,
    {
        let dtype = self.dtype()?;
        let t_dtype = DType::from_type::<T>()?;

        if dtype != t_dtype {
            Err(Error::InvalidDTypeError)
        } else {
            let p = unsafe { lten::lten_get_data_ptr(self.tensorp) as *mut T };
            Ok(unsafe { std::slice::from_raw_parts::<'a, T>(p, self.numel()?) })
        }
    }

    pub fn slice<R: RangeBounds<i64>>(&self, dim: i64, r: R) -> Result<Tensor> {
        let begin = r.start_bound();
        let begin: Result<i64> = match begin {
            Bound::Included(i) => Ok(*i),
            Bound::Excluded(_) => Err(Error::UnsupportedRangeError),
            Bound::Unbounded => Ok(0),
        };
        let begin = begin?;

        let end = r.end_bound();
        let end: Result<i64> = match end {
            Bound::Included(_) => Err(Error::UnsupportedRangeError),
            Bound::Excluded(i) => Ok(*i),
            Bound::Unbounded => Ok(lten::RANGE_NONE),
        };
        let end = end?;

        let tensorp = unsafe { lten::lten_slice(self.tensorp, dim as i32, begin, end) };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
        }
    }

    pub fn select(&self, dim: i64, index: i64) -> Result<Tensor> {
        if dim != 0 {
            return Err(Error::LtenError(
                "Tensor::select only supports dim=0".to_string(),
            ));
        }
        let tensorp = unsafe { lten::lten_index(self.tensorp, index) };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
        }
    }

    pub fn copy_from(&mut self, src: &Tensor) -> Result<()> {
        let retcode = unsafe { lten::lten_copy(self.tensorp, src.tensorp) };
        if retcode != 0 {
            Err(lten::last_error())
        } else {
            Ok(())
        }
    }

    pub fn fill_float(&mut self, value: f32) -> Result<()> {
        let retcode = unsafe { lten::lten_fill_float(self.tensorp, value) };
        if retcode != 0 {
            Err(lten::last_error())
        } else {
            Ok(())
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

    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        let devicel = device.to_lten();
        let tensorp = unsafe { lten::lten_to_device(self.tensorp, devicel) };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
        }
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        let dtypel = dtype.to_lten();
        let tensorp = unsafe { lten::lten_to_dtype(self.tensorp, dtypel) };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
        }
    }

    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op(Some(rhs), 0, 0.0, lten::OPERATOR_ADD)
    }

    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op(Some(rhs), 0, 0.0, lten::OPERATOR_MUL)
    }

    pub fn apply_rope(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op(Some(rhs), 0, 0.0, lten::OPERATOR_ROPE)
    }

    pub fn softmax(&self) -> Result<Tensor> {
        self.apply_op(None, 0, 0.0, lten::OPERATOR_SOFTMAX)
    }

    pub fn gelu(&self) -> Result<Tensor> {
        self.apply_op(None, 0, 0.0, lten::OPERATOR_GELU)
    }

    pub fn swiglu(&self) -> Result<Tensor> {
        self.apply_op(None, 0, 0.0, lten::OPERATOR_SWIGLU)
    }

    pub fn contiguous(&self) -> Result<Tensor> {
        self.apply_op(None, 0, 0.0, lten::OPERATOR_CONTIGUOUS)
    }

    pub fn sum(&self, dim: i64) -> Result<Tensor> {
        self.apply_op(None, dim, 0.0, lten::OPERATOR_SUM)
    }

    pub fn max(&self, dim: i64) -> Result<Tensor> {
        self.apply_op(None, dim, 0.0, lten::OPERATOR_MAX)
    }

    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op(Some(rhs), 0, 0.0, lten::OPERATOR_MATMUL)
    }

    pub fn lookup(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op(Some(rhs), 0, 0.0, lten::OPERATOR_LOOKUP)
    }

    pub fn scalar_mul(&self, rhs: f32) -> Result<Tensor> {
        self.apply_op(None, 0, rhs, lten::OPERATOR_SCALAR_MUL)
    }

    fn apply_op(&self, targ1: Option<&Tensor>, iarg0: i64, farg0: f32, op: i32) -> Result<Tensor> {
        let tensorp = unsafe {
            lten::lten_apply_operator(
                self.tensorp,
                match targ1 {
                    None => ptr::null_mut(),
                    Some(t) => t.tensorp,
                },
                iarg0,
                farg0,
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
