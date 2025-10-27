use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::modulus::quantization::QuantizationFormat;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Llm,
    Vlm,
}

impl ModelKind {
    pub fn supports_vision(&self) -> bool {
        matches!(self, ModelKind::Vlm)
    }
}

impl Default for ModelKind {
    fn default() -> Self {
        ModelKind::Llm
    }
}

/// Vision-specific assets needed for VLM execution.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VisionAssets {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_processor_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projector_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_path: Option<String>,
}

/// Represents artifacts produced by Modulus (weights + metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifacts {
    pub model_path: String,
    pub tokenizer_path: String,
    pub quant_format: QuantizationFormat,
    #[serde(default)]
    pub model_kind: ModelKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<VisionAssets>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactBundle {
    pub name: String,
    pub version: String,
    pub description: String,
    pub artifacts: Vec<ModelArtifacts>,
}

impl ArtifactBundle {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            description: String::new(),
            artifacts: Vec::new(),
        }
    }

    pub fn add_artifact(&mut self, artifact: ModelArtifacts) {
        self.artifacts.push(artifact);
    }
}
