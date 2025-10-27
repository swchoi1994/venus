use serde::{Deserialize, Serialize};

/// Describes hardware capabilities detected at runtime. ModelRun can ingest
/// this structure to decide placement and batching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub backend: String,
    pub total_memory_gb: u64,
    pub available_memory_gb: u64,
    pub peak_tokens_per_second: f32,
}

impl HardwareProfile {
    pub fn cpu_fallback() -> Self {
        Self {
            backend: "cpu".to_string(),
            total_memory_gb: 0,
            available_memory_gb: 0,
            peak_tokens_per_second: 0.0,
        }
    }
}

pub struct Profiler;

impl Profiler {
    pub fn snapshot() -> HardwareProfile {
        if let Ok(mem_info) = sys_info::mem_info() {
            HardwareProfile {
                backend: "cpu".to_string(),
                total_memory_gb: mem_info.total / 1024 / 1024,
                available_memory_gb: mem_info.avail / 1024 / 1024,
                peak_tokens_per_second: 0.0,
            }
        } else {
            HardwareProfile::cpu_fallback()
        }
    }
}
