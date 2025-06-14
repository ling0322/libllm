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
    pub(crate) fn lten_get_numel(tensor: LTensorPtr, numel: *mut i64) -> i32;
    pub(crate) fn lten_get_data_ptr(tensor: LTensorPtr) -> *mut c_void;
    pub(crate) fn lten_get_dtype(tensor: LTensorPtr, dtype: *mut i32) -> i32;
    pub(crate) fn lten_get_device(tensor: LTensorPtr, device: *mut i32) -> i32;
    pub(crate) fn lten_view(tensor: LTensorPtr, dim: i32, shape: *const i64) -> LTensorPtr;
    pub(crate) fn lten_transpose(tensor: LTensorPtr, dim0: i32, dim1: i32) -> LTensorPtr;
    pub(crate) fn lten_expand(tensor: LTensorPtr, dim: i32, shape: *const i64) -> LTensorPtr;
    pub(crate) fn lten_clone(tensor: LTensorPtr) -> LTensorPtr;
    pub(crate) fn lten_slice(tensor: LTensorPtr, dim: i32, begin: i64, end: i64) -> LTensorPtr;
    pub(crate) fn lten_index(tensor: LTensorPtr, index: i64) -> LTensorPtr;
    pub(crate) fn lten_to_device(tensor: LTensorPtr, device: i32) -> LTensorPtr;
    pub(crate) fn lten_to_dtype(tensor: LTensorPtr, dtype: i32) -> LTensorPtr;
    pub(crate) fn lten_copy(dest: LTensorPtr, src: LTensorPtr) -> i32;
    pub(crate) fn lten_fill_float(tensor: LTensorPtr, value: f32) -> i32;
    pub(crate) fn lten_print(tensor: LTensorPtr) -> i32;
    pub(crate) fn lten_apply_operator(
        targ0: LTensorPtr,
        targ1: LTensorPtr,
        targ2: LTensorPtr,
        targ3: LTensorPtr,
        iarg0: i64,
        iarg1: i64,
        farg0: f32,
        farg1: f32,
        op: i32,
    ) -> LTensorPtr;
}

pub(crate) const OPERATOR_ADD: i32 = 0;
pub(crate) const OPERATOR_MUL: i32 = 1;
pub(crate) const OPERATOR_ROPE: i32 = 2;
pub(crate) const OPERATOR_SOFTMAX: i32 = 3;
pub(crate) const OPERATOR_GELU: i32 = 4;
pub(crate) const OPERATOR_SWIGLU: i32 = 5;
pub(crate) const OPERATOR_CONTIGUOUS: i32 = 6;
pub(crate) const OPERATOR_SUM: i32 = 7;
pub(crate) const OPERATOR_MAX: i32 = 8;
pub(crate) const OPERATOR_MATMUL: i32 = 9;
pub(crate) const OPERATOR_LOOKUP: i32 = 10;
pub(crate) const OPERATOR_SCALAR_MUL: i32 = 11;
pub(crate) const OPERATOR_LAYER_NORM: i32 = 12;
pub(crate) const OPERATOR_RMS_NORM: i32 = 13;

pub(crate) const DEVICE_CPU: i32 = 0x0000_0000;
pub(crate) const DEVICE_CUDA: i32 = 0x0001_0000;

pub(crate) const DTYPE_FLOAT: i32 = 1;
pub(crate) const DTYPE_INT64: i32 = 2;
pub(crate) const DTYPE_UINT8: i32 = 3;
pub(crate) const DTYPE_FLOAT16: i32 = 4;
pub(crate) const DTYPE_QINT4: i32 = 5;
pub(crate) const DTYPE_INT8: i32 = 6;

pub(crate) const RANGE_NONE: i64 = -0x1000000000000000;

pub(crate) fn last_error_string() -> String {
    unsafe {
        let ptr = lten_last_error_message();
        let c_str = CStr::from_ptr(ptr);

        c_str.to_string_lossy().to_string()
    }
}

pub(crate) fn last_error() -> crate::Error {
    crate::Error::LtenError(last_error_string())
}
