pub mod api_server;
pub mod engine;
pub mod models;
pub mod platform;

pub use api_server::{ApiServer, ServerConfig};
pub use engine::VenusEngine;
pub use models::{ChatMessage, ChatCompletionRequest, ChatCompletionResponse};

// Re-export platform detection
pub use platform::{detect_platform, get_simd_features};