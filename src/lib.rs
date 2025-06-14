mod layer;
mod lten;
mod operator;
mod tensor;

pub use operator::F;
pub use tensor::DType;
pub use tensor::Device;
pub use tensor::Tensor;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("tensor operator error")]
    LtenError(String),

    #[error("unsupported range type error")]
    UnsupportedRangeError,

    #[error("tensor not exist")]
    TensorNotExistError,

    #[error("invalid dtype error")]
    InvalidDTypeError,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
