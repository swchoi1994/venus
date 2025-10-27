use crate::inferx::engine::InferXEngine;
use anyhow::Result;
use std::sync::Arc;

/// Supported backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Metal,
    Cuda,
    Rocm,
    Dsp,
}

/// Trait implemented by each hardware backend. It exposes the minimal surface
/// InferX needs to schedule inference work efficiently.
pub trait Backend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn name(&self) -> &'static str;
    fn initialize(&self) -> Result<()> {
        Ok(())
    }
    fn attach_engine(&self, _engine: &Arc<InferXEngine>) -> Result<()> {
        Ok(())
    }
}

/// Registry keeps backends discoverable for ModelRun.
#[derive(Default)]
pub struct BackendRegistry {
    backends: Vec<Arc<dyn Backend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    pub fn register(&mut self, backend: Arc<dyn Backend>) {
        self.backends.push(backend);
    }

    pub fn list(&self) -> &[Arc<dyn Backend>] {
        &self.backends
    }

    pub fn get(&self, kind: BackendKind) -> Option<Arc<dyn Backend>> {
        self.backends
            .iter()
            .find(|backend| backend.kind() == kind)
            .cloned()
    }
}
