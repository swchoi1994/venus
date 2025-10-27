use serde::Serialize;
use serde_json::Value;
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize)]
pub struct TelemetryEvent<'a> {
    pub timestamp: u128,
    pub event: &'a str,
    pub payload: Value,
}

pub trait TelemetrySink: Send + Sync {
    fn submit(&self, event: TelemetryEvent<'_>);
}

/// Simple stdout sink for local development.
pub struct StdoutSink;

impl TelemetrySink for StdoutSink {
    fn submit(&self, event: TelemetryEvent<'_>) {
        let millis = event.timestamp;
        println!("[{}] {}: {}", millis, event.event, event.payload);
    }
}

pub fn new_event(event: &str, payload: Value) -> TelemetryEvent<'_> {
    TelemetryEvent {
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default(),
        event,
        payload,
    }
}
