pub mod backend;
pub mod platform;
pub mod profiling;

pub use backend::{Backend, BackendKind, BackendRegistry};
pub use platform::{detect_platform, get_simd_features, Platform};
pub use profiling::{HardwareProfile, Profiler};
