use serde::{Deserialize, Serialize};

/// Supported quantization formats in Modulus.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationFormat {
    None,
    Q8_0,
    Q4_0,
    Q4_K,
    Awq,
}

/// Configuration describing how weights and activations will be quantized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub format: QuantizationFormat,
    pub group_size: Option<usize>,
    pub per_channel: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            format: QuantizationFormat::Q4_0,
            group_size: Some(128),
            per_channel: true,
        }
    }
}

/// Planner interface so external tools (Python Modulus CLI) can plug in.
pub trait Quantizer: Send + Sync {
    fn config(&self) -> &QuantizationConfig;
    fn quantize_weight(&self, tensor_name: &str, data: &[f32]) -> anyhow::Result<Vec<u8>>;
    fn dequantize_weight(&self, tensor_name: &str, data: &[u8]) -> anyhow::Result<Vec<f32>>;
}
