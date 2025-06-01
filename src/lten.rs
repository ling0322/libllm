use std::ffi::{c_char, c_void, CStr};

pub(crate) type LTensorPtr = *mut c_void;

extern "C" {
    pub(crate) fn lten_last_error_message() -> *const c_char;
    pub(crate) fn lten_destroy_tensor(tensor: LTensorPtr) -> i32;
    pub(crate) fn lten_new_tensor(
        dim: i32,
        shape: *const i64,
        dtype: i32,
        device: i32,
    ) -> LTensorPtr;
    pub(crate) fn lten_get_dim(tensor: LTensorPtr, dim: *mut i32) -> i32;
    pub(crate) fn lten_get_shape(tensor: LTensorPtr, dim: i32, size: *mut i64) -> i32;
    pub(crate) fn lten_get_dtype(tensor: LTensorPtr, dtype: *mut i32) -> i32;
    pub(crate) fn lten_get_device(tensor: LTensorPtr, device: *mut i32) -> i32;
    pub(crate) fn lten_view(tensor: LTensorPtr, dim: i32, shape: *const i64) -> LTensorPtr;
    pub(crate) fn lten_expand(tensor: LTensorPtr, dim: i32, shape: *const i64) -> LTensorPtr;
    pub(crate) fn lten_slice(tensor: LTensorPtr, dim: i32, begin: i64, end: i64) -> LTensorPtr;
    pub(crate) fn lten_index(tensor: LTensorPtr, index: i64) -> LTensorPtr;
    pub(crate) fn lten_to_device(tensor: LTensorPtr, device: i32) -> LTensorPtr;
    pub(crate) fn lten_to_dtype(tensor: LTensorPtr, dtype: i32) -> LTensorPtr;
    pub(crate) fn lten_copy(dest: LTensorPtr, src: LTensorPtr) -> i32;
    pub(crate) fn lten_copy_memory(tensor: LTensorPtr, buf: *mut c_void, bufsiz: i64) -> i32;
    pub(crate) fn lten_fill_float(tensor: LTensorPtr, value: f32) -> i32;
    pub(crate) fn lten_print(tensor: LTensorPtr) -> i32;
    pub(crate) fn lten_apply_operator(
        targ0: LTensorPtr,
        targ1: LTensorPtr,
        iarg0: i64,
        farg0: f32,
        op: i32,
    ) -> LTensorPtr;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Operator {
    Add = 0,
    Mul = 1,
    Rope = 2,
    Softmax = 3,
    Gelu = 4,
    Swiglu = 5,
    Contiguous = 6,
    Sum = 7,
    Max = 8,
    Matmul = 9,
    Lookup = 10,
    ScalarMul = 11,
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Device {
    Cpu = 0x0000_0000,
    Cuda = 0x0001_0000,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Dtype {
    Float = 1,
    Int64 = 2,
    Uint8 = 3,
    Float16 = 4,
    QInt4 = 5,
    Int8 = 6,
}

pub(crate) fn last_error_string() -> String {
    unsafe {
        let ptr = lten_last_error_message();
        let c_str = CStr::from_ptr(ptr);

        c_str.to_string_lossy().to_string()
    }
}

pub(crate) fn last_error() -> crate::Error {
    crate::Error::LynnError(last_error_string())
}
