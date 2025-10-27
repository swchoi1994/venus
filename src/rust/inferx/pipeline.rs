use crate::inferx::engine::InferXEngine;
use std::sync::Arc;

/// Represents a logical inference pipeline composed of stages.
#[derive(Default)]
pub struct InferencePipeline {
    pub stages: Vec<PipelineStage>,
}

impl InferencePipeline {
    pub fn new(stages: Vec<PipelineStage>) -> Self {
        Self { stages }
    }

    pub fn execute(&self, engine: &InferXEngine, prompt: &str) -> anyhow::Result<String> {
        let mut working = prompt.to_string();
        for stage in &self.stages {
            working = stage.run(engine, &working)?;
        }
        Ok(working)
    }
}

/// A pipeline stage can mutate the prompt or post-process outputs.
#[derive(Clone)]
pub struct PipelineStage {
    pub name: String,
    handler: Arc<dyn Fn(&InferXEngine, &str) -> anyhow::Result<String> + Send + Sync>,
}

impl PipelineStage {
    pub fn new<F>(name: impl Into<String>, handler: F) -> Self
    where
        F: Fn(&InferXEngine, &str) -> anyhow::Result<String> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            handler: Arc::new(handler),
        }
    }

    pub fn run(&self, engine: &InferXEngine, input: &str) -> anyhow::Result<String> {
        (self.handler)(engine, input)
    }
}
