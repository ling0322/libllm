//! Safe Rust bindings for the native flint tensor library.
//!
//! A [`Tensor`] owns a handle to shape metadata plus a reference to storage that other tensors may
//! share. Reshaping operations such as [`Tensor::view`] or [`Tensor::transpose`] return a new
//! tensor over the same elements rather than copying them, and `clone` does the same, so cloning a
//! tensor is cheap but does not give you an independent copy of the data.
//!
//! ```no_run
//! use llm::flint::{self, DType, Device, Tensor};
//!
//! flint::init();
//! let x = Tensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//! assert_eq!(x.shape(), vec![2, 3]);
//! assert_eq!(x.dtype(), DType::Float);
//! assert_eq!(x.device(), Device::Cpu);
//! assert_eq!(x.transpose(0, 1)?.to_vec_f32()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
//! # Ok::<(), llm::flint::Error>(())
//! ```
//!
//! The operations that work on tensors rather than describe them live in [`functional`], which
//! mirrors `flint/functional.h`:
//!
//! ```no_run
//! use llm::flint::{functional as F, Tensor};
//!
//! let x = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0])?;
//! let sums = F::sum(&x, F::LAST_DIM)?;
//! assert_eq!(sums.to_vec_f32()?, vec![3.0, 7.0]);
//! # Ok::<(), llm::flint::Error>(())
//! ```
//!
//! # Threading
//!
//! [`Tensor`] is deliberately neither `Send` nor `Sync`. The underlying operators keep per-device
//! state that is not prepared for concurrent use, so a tensor stays on the thread that made it.

mod ffi;
pub mod functional;

use std::ffi::CStr;
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::sync::Once;

/// Element type of a tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum DType {
    Float = 1,
    Long = 2,
    UInt8 = 3,
    Float16 = 4,
    Int8 = 6,
    Fp4E2M0x2 = 7,
    Bool = 8,
    Int32 = 9,
}

impl DType {
    /// The number this type is written as, in a model file or over the C interface.
    pub fn code(self) -> i32 {
        self as i32
    }

    /// The type `code` names, as [`DType::code`] wrote it.
    pub fn from_code(code: i32) -> Result<DType> {
        DType::from_raw(code)
    }

    /// The number of bytes `numel` elements of this type occupy once packed together.
    ///
    /// Not always `numel` times a fixed width: [`DType::Fp4E2M0x2`] counts a packed pair of
    /// quantized values as one element, so a pair takes one byte.
    pub fn total_size(self, numel: i64) -> i64 {
        match self {
            DType::Float | DType::Int32 => 4 * numel,
            DType::Float16 => 2 * numel,
            DType::Long => 8 * numel,
            DType::UInt8 | DType::Int8 | DType::Bool | DType::Fp4E2M0x2 => numel,
        }
    }

    fn from_raw(raw: i32) -> Result<DType> {
        match raw {
            1 => Ok(DType::Float),
            2 => Ok(DType::Long),
            3 => Ok(DType::UInt8),
            4 => Ok(DType::Float16),
            6 => Ok(DType::Int8),
            7 => Ok(DType::Fp4E2M0x2),
            8 => Ok(DType::Bool),
            9 => Ok(DType::Int32),
            other => Err(Error::unsupported(format!("unknown dtype {other}"))),
        }
    }
}

/// Where a tensor's storage lives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum Device {
    Cpu = 0,
    Cuda = 1,
}

impl Device {
    /// Whether this build has operators for the device and the machine can run them.
    ///
    /// Worth asking before running anything: the operators a device is missing end the process
    /// rather than reporting an error, so a caller that can fall back should check first.
    pub fn is_available(self) -> bool {
        init();
        let mut available: i32 = 0;
        match check(unsafe { ffi::fl_is_device_available(self as i32, &mut available) }) {
            Ok(()) => available != 0,
            Err(_) => false,
        }
    }

    fn from_raw(raw: i32) -> Result<Device> {
        match raw {
            0 => Ok(Device::Cpu),
            1 => Ok(Device::Cuda),
            other => Err(Error::unsupported(format!("unknown device {other}"))),
        }
    }
}

/// One end of a slice range. [`Bound::End`] leaves that end of the dimension where it is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bound {
    /// A position, which may be negative to count from the back.
    At(i32),
    /// The start or the end of the dimension, whichever side this bound is on.
    End,
}

impl Bound {
    fn to_raw(self) -> i32 {
        match self {
            Bound::At(index) => index,
            Bound::End => ffi::FL_NONE,
        }
    }
}

impl From<i32> for Bound {
    fn from(index: i32) -> Bound {
        Bound::At(index)
    }
}

/// A failure reported by the library.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    code: i32,
    message: String,
}

impl Error {
    /// The C error code, or zero for a failure the binding itself detected.
    pub fn code(&self) -> i32 {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    /// Whether the call was rejected for an argument it could not accept.
    pub fn is_invalid_arg(&self) -> bool {
        self.code == ffi::FL_ERROR_INVALID_ARG
    }

    /// Whether the call was accepted but could not be carried out, such as running out of memory
    /// or asking for a device the build does not support.
    pub fn is_aborted(&self) -> bool {
        self.code == ffi::FL_ERROR_ABORTED
    }

    fn unsupported(message: String) -> Error {
        Error { code: 0, message }
    }

    /// Reads the error the last C call left on this thread. Only called right after one failed.
    fn last() -> Error {
        // Safety: the pointer is owned by the library and stays valid until this thread makes
        // another call into it, which cannot happen while we are copying it out.
        let message = unsafe {
            let raw = ffi::fl_get_last_error_message();
            if raw.is_null() {
                String::new()
            } else {
                CStr::from_ptr(raw).to_string_lossy().into_owned()
            }
        };
        let code = unsafe { ffi::fl_get_last_error_code() };
        Error {
            code,
            message: if message.is_empty() {
                "unknown error".to_string()
            } else {
                message
            },
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (code 0x{:04x})", self.message, self.code)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Turns a status code into a `Result`, picking up the message the failing call left behind.
fn check(status: i32) -> Result<()> {
    if status == ffi::FL_OK {
        Ok(())
    } else {
        Err(Error::last())
    }
}

static INIT: Once = Once::new();

/// Select the operator backends for this machine.
///
/// Every constructor calls this, so it is only worth calling directly to get the cost out of the
/// way at a point of your choosing. Repeated calls do nothing.
pub fn init() {
    INIT.call_once(|| unsafe { ffi::fl_init() });
}

/// The memory usage of one device.
///
/// A device that does not report its usage, which is what the CPU backend does, reports zero
/// everywhere; a caller that has to size an allocation from this should check [`total`] first.
///
/// [`total`]: MemorySnapshot::total
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemorySnapshot {
    /// The memory the device has.
    pub total: i64,
    /// The memory no process has reserved yet. Memory this process already took from the driver
    /// is not free even once its tensors are gone, since the allocator holds on to it for reuse.
    pub free: i64,
    /// The bytes the tensors of this process hold.
    pub allocated: i64,
    /// The largest [`allocated`](MemorySnapshot::allocated) reached since the last
    /// [`MemorySnapshot::reset_peak_stats`].
    pub peak_allocated: i64,
}

impl MemorySnapshot {
    /// Measure the memory usage of `device`.
    pub fn capture(device: Device) -> Result<MemorySnapshot> {
        init();
        let mut raw = ffi::FlMemorySnapshot::default();
        check(unsafe { ffi::fl_memory_capture(device as i32, &mut raw) })?;
        Ok(MemorySnapshot {
            total: raw.total,
            free: raw.free,
            allocated: raw.allocated,
            peak_allocated: raw.peak_allocated,
        })
    }

    /// Set the peak of `device` back to zero, so that the next measurement covers only what
    /// happens from here. This is how the size of one forward pass is measured.
    pub fn reset_peak_stats(device: Device) -> Result<()> {
        init();
        check(unsafe { ffi::fl_memory_reset_peak_stats(device as i32) })
    }
}

/// A tensor: a shape over storage that other tensors may share.
///
/// Dropping a tensor releases its handle; the storage goes away once the last tensor referring to
/// it is gone.
pub struct Tensor {
    raw: ffi::FlTensor,
    /// Keeps the type off `Send`/`Sync`, since the operators behind it are not ready for either.
    _not_sync: PhantomData<*const ()>,
}

impl Tensor {
    /// Wraps a handle a C call just produced.
    ///
    /// # Safety
    ///
    /// `raw` must be non-null and freshly owned; the tensor takes over destroying it.
    unsafe fn from_raw(raw: ffi::FlTensor) -> Tensor {
        Tensor {
            raw,
            _not_sync: PhantomData,
        }
    }

    /// Runs a C call that produces a tensor, handing back the handle it wrote.
    fn produce(call: impl FnOnce(*mut ffi::FlTensor) -> i32) -> Result<Tensor> {
        let mut raw: ffi::FlTensor = std::ptr::null_mut();
        check(call(&mut raw))?;
        debug_assert!(!raw.is_null(), "a successful call must produce a handle");
        Ok(unsafe { Tensor::from_raw(raw) })
    }

    /// Runs a C call that reports a value about this tensor.
    fn query<T: Default>(&self, call: impl FnOnce(ffi::FlTensor, *mut T) -> i32) -> Result<T> {
        let mut value = T::default();
        check(call(self.raw, &mut value))?;
        Ok(value)
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[i32], dtype: DType, device: Device) -> Result<Tensor> {
        init();
        Tensor::produce(|out| unsafe {
            ffi::fl_tensor_zeros(
                shape.as_ptr(),
                shape.len() as i32,
                dtype as i32,
                device as i32,
                out,
            )
        })
    }

    /// Create a tensor without writing anything into it.
    ///
    /// For storage that is about to be overwritten in full, such as a KV cache pool: zeroing tens
    /// of gigabytes that the first forward pass overwrites anyway is a cost with nothing to show
    /// for it. Reading an element before writing it gives whatever the allocator handed back, so
    /// this is only worth using when every element is written first.
    pub fn empty(shape: &[i32], dtype: DType, device: Device) -> Result<Tensor> {
        init();
        Tensor::produce(|out| unsafe {
            ffi::fl_tensor_empty(
                shape.as_ptr(),
                shape.len() as i32,
                dtype as i32,
                device as i32,
                out,
            )
        })
    }

    /// Create a CPU tensor holding a copy of `data`, laid out row-major.
    ///
    /// Fails if `data` does not hold exactly as many elements as `shape` describes.
    pub fn from_f32(shape: &[i32], data: &[f32]) -> Result<Tensor> {
        Tensor::from_elements(shape, data, DType::Float)
    }

    /// Create a CPU tensor of 64-bit integers holding a copy of `data`.
    pub fn from_i64(shape: &[i32], data: &[i64]) -> Result<Tensor> {
        Tensor::from_elements(shape, data, DType::Long)
    }

    /// Create a CPU tensor of 32-bit integers holding a copy of `data`, the type the paged
    /// attention operations take their block tables and sequence lengths in.
    pub fn from_i32(shape: &[i32], data: &[i32]) -> Result<Tensor> {
        Tensor::from_elements(shape, data, DType::Int32)
    }

    /// Create a CPU tensor of bytes holding a copy of `data`, the type the element-wise
    /// comparisons take their inputs in.
    pub fn from_u8(shape: &[i32], data: &[u8]) -> Result<Tensor> {
        Tensor::from_elements(shape, data, DType::UInt8)
    }

    /// Create a CPU tensor of `dtype` over the raw bytes of `data`, laid out row-major.
    ///
    /// The element types a tensor can hold are not all types Rust has a use for on its own, so
    /// this is how a quantized or half precision tensor gets built: hand over the bytes as they
    /// were stored. `data` must be exactly as long as `shape` and `dtype` describe.
    pub fn from_bytes(shape: &[i32], dtype: DType, data: &[u8]) -> Result<Tensor> {
        Tensor::from_elements(shape, data, dtype)
    }

    fn from_elements<T>(shape: &[i32], data: &[T], dtype: DType) -> Result<Tensor> {
        init();
        let size = std::mem::size_of_val(data) as i64;
        Tensor::produce(|out| unsafe {
            ffi::fl_tensor_from_data(
                shape.as_ptr(),
                shape.len() as i32,
                dtype as i32,
                data.as_ptr() as *const c_void,
                size,
                out,
            )
        })
    }

    /// Number of dimensions.
    pub fn dim(&self) -> Result<i32> {
        self.query(|t, out| unsafe { ffi::fl_tensor_get_dim(t, out) })
    }

    /// Size of dimension `dim`, which may be negative to count from the back.
    pub fn shape_at(&self, dim: i32) -> Result<i32> {
        self.query(|t, out| unsafe { ffi::fl_tensor_get_shape(t, dim, out) })
    }

    /// Sizes of every dimension.
    pub fn shape(&self) -> Vec<i32> {
        let dim = self.dim().unwrap_or(0);
        (0..dim).filter_map(|d| self.shape_at(d).ok()).collect()
    }

    /// Stride of dimension `dim`, in elements.
    pub fn stride(&self, dim: i32) -> Result<i32> {
        self.query(|t, out| unsafe { ffi::fl_tensor_get_stride(t, dim, out) })
    }

    /// Total number of elements.
    pub fn numel(&self) -> i64 {
        self.query(|t, out| unsafe { ffi::fl_tensor_get_numel(t, out) })
            .unwrap_or(0)
    }

    pub fn dtype(&self) -> DType {
        self.try_dtype().expect("a live tensor has a known dtype")
    }

    pub fn try_dtype(&self) -> Result<DType> {
        let raw: i32 = self.query(|t, out| unsafe { ffi::fl_tensor_get_dtype(t, out) })?;
        DType::from_raw(raw)
    }

    pub fn device(&self) -> Device {
        self.try_device().expect("a live tensor has a known device")
    }

    pub fn try_device(&self) -> Result<Device> {
        let raw: i32 = self.query(|t, out| unsafe { ffi::fl_tensor_get_device(t, out) })?;
        Device::from_raw(raw)
    }

    /// Whether the elements sit next to each other in memory, which is what [`Tensor::view`]
    /// requires.
    pub fn is_contiguous(&self) -> bool {
        self.query(|t, out| unsafe { ffi::fl_tensor_is_contiguous(t, out) })
            .map(|flag: i32| flag != 0)
            .unwrap_or(false)
    }

    /// Read the same elements under a new shape, sharing the storage.
    pub fn view(&self, shape: &[i32]) -> Result<Tensor> {
        Tensor::produce(|out| unsafe {
            ffi::fl_tensor_view(self.raw, shape.as_ptr(), shape.len() as i32, out)
        })
    }

    /// Read the same bytes as another element type, sharing the storage.
    ///
    /// The last dimension is the one that changes size: a `<u8>(n, 64)` becomes a `<f16>(n, 32)`.
    /// Nothing is copied and nothing is converted, which is what lets one pool of bytes back the
    /// tensors of layers that keep different types in it.
    ///
    /// The last dimension must be packed, and the bytes it spans, every other stride and the
    /// tensor's own offset must all divide by the size of `dtype`.
    pub fn view_as(&self, dtype: DType) -> Result<Tensor> {
        Tensor::produce(|out| unsafe { ffi::fl_tensor_view_as(self.raw, dtype as i32, out) })
    }

    /// Exchange two dimensions, sharing the storage. The result is usually not contiguous.
    pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Tensor> {
        Tensor::produce(|out| unsafe { ffi::fl_tensor_transpose(self.raw, dim0, dim1, out) })
    }

    /// Take the half-open range `[begin, end)` of dimension `dim`, sharing the storage.
    ///
    /// Both bounds accept a plain `i32`, negative to count from the back, or [`Bound::End`] to
    /// leave that side alone.
    pub fn slice(
        &self,
        dim: i32,
        begin: impl Into<Bound>,
        end: impl Into<Bound>,
    ) -> Result<Tensor> {
        let (begin, end) = (begin.into().to_raw(), end.into().to_raw());
        Tensor::produce(|out| unsafe { ffi::fl_tensor_slice(self.raw, dim, begin, end, out) })
    }

    /// Take one entry of the first dimension, dropping that dimension.
    pub fn subtensor(&self, index: i32) -> Result<Tensor> {
        Tensor::produce(|out| unsafe { ffi::fl_tensor_subtensor(self.raw, index, out) })
    }

    /// Add a dimension of size one at `dim`.
    pub fn unsqueeze(&self, dim: i32) -> Result<Tensor> {
        Tensor::produce(|out| unsafe { ffi::fl_tensor_unsqueeze(self.raw, dim, out) })
    }

    /// Remove the dimension at `dim`, which must have size one.
    pub fn squeeze(&self, dim: i32) -> Result<Tensor> {
        Tensor::produce(|out| unsafe { ffi::fl_tensor_squeeze(self.raw, dim, out) })
    }

    /// Return a contiguous tensor with the same elements, copying only if needed.
    pub fn contiguous(&self) -> Result<Tensor> {
        Tensor::produce(|out| unsafe { ffi::fl_tensor_contiguous(self.raw, out) })
    }

    /// Copy the tensor to another device.
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        Tensor::produce(|out| unsafe { ffi::fl_tensor_to_device(self.raw, device as i32, out) })
    }

    /// Convert the elements to another data type.
    pub fn cast(&self, dtype: DType) -> Result<Tensor> {
        Tensor::produce(|out| unsafe { ffi::fl_tensor_cast(self.raw, dtype as i32, out) })
    }

    /// Number of bytes the elements occupy once packed together.
    pub fn nbytes(&self) -> Result<i64> {
        self.query(|t, out| unsafe { ffi::fl_tensor_get_nbytes(t, out) })
    }

    /// Copy the elements out in row-major order, bringing them back from the device and packing
    /// them first if needed.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.to_vec(DType::Float)
    }

    /// Copy 64-bit integer elements out in row-major order.
    pub fn to_vec_i64(&self) -> Result<Vec<i64>> {
        self.to_vec(DType::Long)
    }

    /// Copy 32-bit integer elements out in row-major order.
    pub fn to_vec_i32(&self) -> Result<Vec<i32>> {
        self.to_vec(DType::Int32)
    }

    /// Copy byte elements out in row-major order.
    pub fn to_vec_u8(&self) -> Result<Vec<u8>> {
        self.to_vec(DType::UInt8)
    }

    /// Copy boolean elements out in row-major order, as the comparisons in
    /// [`functional`] produce them.
    pub fn to_vec_bool(&self) -> Result<Vec<bool>> {
        // Read the bytes rather than `bool` itself: a byte that is neither 0 nor 1 is a valid u8
        // but not a valid bool, and nothing here can promise the library never writes one.
        let bytes: Vec<u8> = self.to_vec(DType::Bool)?;
        Ok(bytes.into_iter().map(|byte| byte != 0).collect())
    }

    fn to_vec<T: Default + Clone>(&self, expected: DType) -> Result<Vec<T>> {
        let dtype = self.try_dtype()?;
        if dtype != expected {
            return Err(Error::unsupported(format!(
                "tensor holds {dtype:?}, not {expected:?}; cast it first"
            )));
        }

        let nbytes = self.nbytes()?;
        let count = nbytes as usize / std::mem::size_of::<T>();
        let mut values = vec![T::default(); count];
        check(unsafe {
            ffi::fl_tensor_copy_to_host(self.raw, values.as_mut_ptr() as *mut c_void, nbytes)
        })?;
        Ok(values)
    }
}

impl Clone for Tensor {
    /// Makes another handle on the same storage rather than copying the elements. Use
    /// [`Tensor::contiguous`] on a tensor you want to own outright.
    fn clone(&self) -> Tensor {
        Tensor::produce(|out| unsafe { ffi::fl_tensor_clone(self.raw, out) })
            .expect("cloning a live tensor cannot fail")
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        // Safety: the handle came from a successful C call and is destroyed exactly once, since
        // Tensor is not Copy and every clone makes its own handle.
        unsafe { ffi::fl_tensor_destroy(self.raw) };
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("dtype", &self.try_dtype().ok())
            .field("device", &self.try_device().ok())
            .field("contiguous", &self.is_contiguous())
            .finish()
    }
}
