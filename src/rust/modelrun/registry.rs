use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::inferx::EngineManager;
use crate::modulus::pipeline::ModelArtifacts;

/// High-level deployment configuration consumed by ModelRun.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeploymentConfig {
    pub default_model: Option<String>,
    pub models: HashMap<String, ModelArtifacts>,
}

/// Registry tracks available artifacts and exposes lookup helpers for the API.
#[derive(Clone)]
pub struct ModelRegistry {
    config: DeploymentConfig,
    base_dir: PathBuf,
}

impl ModelRegistry {
    pub fn new(config: DeploymentConfig, base_dir: PathBuf) -> Self {
        Self { config, base_dir }
    }

    pub fn load_from_dir(base_dir: &Path) -> Result<Self> {
        let manifest = base_dir.join("deployment.json");
        if manifest.exists() {
            let contents = fs::read_to_string(&manifest)
                .with_context(|| format!("failed to read {}", manifest.display()))?;
            let config: DeploymentConfig = serde_json::from_str(&contents)
                .with_context(|| format!("failed to parse {}", manifest.display()))?;
            Ok(Self::new(config, base_dir.to_path_buf()))
        } else {
            Ok(Self::new(
                DeploymentConfig::default(),
                base_dir.to_path_buf(),
            ))
        }
    }

    pub fn config(&self) -> &DeploymentConfig {
        &self.config
    }

    pub fn default_model_name(&self) -> Option<&str> {
        self.config.default_model.as_deref()
    }

    pub fn model(&self, name: &str) -> Option<&ModelArtifacts> {
        self.config.models.get(name)
    }

    pub fn models(&self) -> impl Iterator<Item = (&String, &ModelArtifacts)> {
        self.config.models.iter()
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.base_dir.join(candidate)
        }
    }

    pub fn model_path(&self, artifact: &ModelArtifacts) -> PathBuf {
        self.resolve_path(&artifact.model_path)
    }

    pub fn tokenizer_path(&self, artifact: &ModelArtifacts) -> PathBuf {
        self.resolve_path(&artifact.tokenizer_path)
    }

    pub fn preload(&self, manager: &EngineManager) -> Result<()> {
        for (name, artifact) in self.models() {
            let model_path = self.model_path(artifact);
            info!("Loading model {} from {}", name, model_path.display());
            let model_path_str = model_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF8 model path for {}", name))?;

            manager
                .load_model(name, model_path_str)
                .with_context(|| format!("failed to load model {}", name))?;
        }

        Ok(())
    }
}
