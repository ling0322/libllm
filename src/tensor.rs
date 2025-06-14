use crate::{lten, operator::F, Error, Result};
use std::any::TypeId;
use std::fmt;
use std::io::BufReader;
use std::io::Read;
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

impl From<&Vec<i64>> for Shape {
    fn from(t: &Vec<i64>) -> Self {
        Shape { size: t.clone() }
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

    pub(crate) fn from_lten(dtype: i32) -> Result<Self> {
        match dtype {
            lten::DTYPE_FLOAT => Ok(Self::Float),
            lten::DTYPE_INT64 => Ok(Self::Int64),
            lten::DTYPE_UINT8 => Ok(Self::UInt8),
            lten::DTYPE_FLOAT16 => Ok(Self::Float16),
            lten::DTYPE_QINT4 => Ok(Self::QInt4),
            lten::DTYPE_INT8 => Ok(Self::Int8),
            _ => Err(Error::LtenError("invalid lten dtype value".to_string())),
        }
    }

    pub fn num_bytes(&self, numel: usize) -> Result<usize> {
        match self {
            Self::Float => Ok(numel * 4),
            Self::Int64 => Ok(numel * 8),
            Self::UInt8 => Ok(numel),
            Self::Float16 => Ok(numel * 2),
            Self::QInt4 => Ok(numel / 2),
            Self::Int8 => Ok(numel),
            _ => Err(Error::LtenError("unknown dtype".to_string())),
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
    pub(crate) tensorp: lten::LTensorPtr,
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
            tensor.fill(0.0)?;
            Ok(tensor)
        } else {
            Err(Error::LtenError(
                "unsupported dtype for Tensor::zeros".to_string(),
            ))
        }
    }

    pub fn from_reader<R: Read>(r: &mut BufReader<R>) -> Result<Tensor> {
        // opening tag
        let mut s = String::new();
        r.take(16).read_to_string(&mut s)?;
        if s != "[tensor]        " {
            return Err(Error::LtenError("invalid tensor data".to_string()));
        }

        let mut b = [0u8; 8];

        // version
        r.read_exact(&mut b)?;
        let version = i64::from_le_bytes(b) as i32;
        if version != 1 {
            return Err(Error::LtenError("unsupported tensor".to_string()));
        }

        // shape
        r.read_exact(&mut b)?;
        let dim = i64::from_le_bytes(b);
        let mut shape: Vec<i64> = Vec::new();
        let mut shape_numel: i64 = 1;
        for _ in 0..dim {
            r.read_exact(&mut b)?;
            let n = i64::from_le_bytes(b);
            shape.push(n);
            shape_numel *= n;
        }

        // dtype
        r.read_exact(&mut b)?;
        let dtypel = i64::from_le_bytes(b) as i32;
        let dtype = DType::from_lten(dtypel)?;

        // numel
        r.read_exact(&mut b)?;
        let numel = i64::from_le_bytes(b);
        if numel != shape_numel {
            return Err(Error::LtenError("shape and numel mismatch".to_string()));
        }

        // data
        let tensor = Tensor::new(&shape, dtype, Device::Cpu)?;
        let total_bytes = dtype.num_bytes(numel as usize)?;
        let data = unsafe { std::slice::from_raw_parts_mut::<u8>(tensor.raw_data()?, total_bytes) };
        r.read_exact(data)?;

        // closing tag
        r.take(16).read_to_string(&mut s)?;
        if s != "[/tensor]       " {
            return Err(Error::LtenError("invalid tensor data".to_string()));
        }

        Ok(tensor)
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
            Ok(DType::from_lten(dtypel)?)
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

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Result<Tensor> {
        let tensorp = unsafe { lten::lten_transpose(self.tensorp, dim0 as i32, dim1 as i32) };
        if tensorp.is_null() {
            Err(lten::last_error())
        } else {
            Ok(Tensor { tensorp })
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
            let p = self.raw_data::<T>()?;
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
        F::copy(self, src)
    }

    pub fn fill(&mut self, value: f32) -> Result<()> {
        F::fill(self, value)
    }

    pub fn print(&self) {
        F::print(self);
    }

    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        F::to(device, self)
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        F::cast(self, dtype)
    }

    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        F::add(self, rhs)
    }

    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        F::mul(self, rhs)
    }

    pub fn contiguous(&self) -> Result<Tensor> {
        F::contiguous(self)
    }

    pub fn scalar_mul(&self, rhs: f32) -> Result<Tensor> {
        F::scalar_mul(self, rhs)
    }

    fn raw_data<'a, T: 'static>(&self) -> Result<*mut T>
    where
        Self: 'a,
    {
        let p = unsafe { lten::lten_get_data_ptr(self.tensorp) as *mut T };
        if p.is_null() {
            Err(lten::last_error())
        } else {
            Ok(p)
        }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let tensorp = unsafe { lten::lten_clone(self.tensorp) };
        if tensorp.is_null() {
            panic!("failed to call lten_clone()");
        }

        Tensor { tensorp }
    }
}
