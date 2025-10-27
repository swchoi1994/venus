use crate::inferx::engine::InferXEngine;
use parking_lot::RwLock;
use std::sync::Arc;

/// Configuration for the token scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_tokens: usize,
    pub enable_paged_attention: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1,
            max_tokens: 2048,
            enable_paged_attention: true,
        }
    }
}

/// Lightweight handle that schedules requests across loaded engines.
pub struct SchedulerHandle {
    engines: Arc<RwLock<Vec<Arc<InferXEngine>>>>,
    config: SchedulerConfig,
}

impl SchedulerHandle {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            engines: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }

    pub fn register_engine(&self, engine: Arc<InferXEngine>) {
        self.engines.write().push(engine);
    }

    pub fn engines(&self) -> Vec<Arc<InferXEngine>> {
        self.engines.read().clone()
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}
