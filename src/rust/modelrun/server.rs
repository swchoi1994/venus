use axum::response::sse::{KeepAlive, Sse};
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use std::{convert::Infallible, path::PathBuf, sync::Arc, time::Duration};
use tower_http::cors::CorsLayer;
use tracing::{error, info};

use crate::inferx::{EngineManager, InferXEngine};
use crate::modelrun::ModelRegistry;
use crate::models::*;

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_dir: PathBuf,
    pub num_workers: usize,
}

pub struct ApiServer {
    pub config: ServerConfig,
    engine_manager: Arc<EngineManager>,
    registry: Arc<ModelRegistry>,
}

#[derive(Clone)]
struct AppState {
    engine_manager: Arc<EngineManager>,
    registry: Arc<ModelRegistry>,
}

impl ApiServer {
    pub async fn new(config: ServerConfig) -> anyhow::Result<Self> {
        let engine_manager = Arc::new(EngineManager::new());
        let registry = Arc::new(ModelRegistry::load_from_dir(&config.model_dir)?);

        registry.preload(engine_manager.as_ref())?;

        Ok(Self {
            config,
            engine_manager,
            registry,
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let state = AppState {
            engine_manager: self.engine_manager.clone(),
            registry: self.registry.clone(),
        };

        let app = Router::new()
            .route("/", get(root))
            .route("/health", get(health))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .layer(CorsLayer::permissive())
            .with_state(state);

        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;

        info!("API server listening on {}", addr);

        axum::serve(listener, app).await?;

        Ok(())
    }
}

// Handlers

async fn root() -> &'static str {
    "Venus Inference Engine API Server"
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let data = state
        .registry
        .models()
        .map(|(name, artifact)| {
            let metadata = serde_json::json!({
                "kind": artifact.model_kind,
                "quantization": artifact.quant_format,
                "vision": artifact.vision,
                "prompt_template": artifact.prompt_template,
            });

            ModelInfo {
                id: name.clone(),
                object: "model".to_string(),
                created: artifact.created.unwrap_or_default(),
                owned_by: "edgeflow".to_string(),
                metadata: Some(metadata),
            }
        })
        .collect();

    Json(ModelList {
        object: "list".to_string(),
        data,
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, StatusCode> {
    let mut request = request;

    let (selected_model, engine) = match state.engine_manager.get_engine(&request.model) {
        Some(engine) => (request.model.clone(), engine),
        None => {
            let fallback = state
                .registry
                .default_model_name()
                .ok_or(StatusCode::NOT_FOUND)?
                .to_string();
            let engine = state
                .engine_manager
                .get_engine(&fallback)
                .ok_or(StatusCode::NOT_FOUND)?;
            (fallback, engine)
        }
    };

    request.model = selected_model.clone();

    let has_attachments = request
        .messages
        .iter()
        .any(|message| !message.attachments.is_empty());

    if has_attachments {
        let supports_vision = state
            .registry
            .model(&selected_model)
            .map(|artifact| artifact.model_kind.supports_vision())
            .unwrap_or(false);

        if !supports_vision {
            return Err(StatusCode::BAD_REQUEST);
        }
    }

    // Format messages into a prompt
    let prompt = format_chat_prompt(&request.messages);

    if request.stream {
        // Streaming response
        let stream = generate_stream(engine, prompt, request);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming response
        match engine.generate(
            &prompt,
            request.temperature,
            request.top_p,
            request.top_k,
            request.max_tokens as i32,
        ) {
            Ok(response_text) => {
                let prompt_tokens = engine.count_tokens(&prompt).unwrap_or(0);
                let completion_tokens = engine.count_tokens(&response_text).unwrap_or(0);

                let response = ChatCompletionResponse::new(
                    selected_model,
                    ChatMessage {
                        role: "assistant".to_string(),
                        content: response_text,
                        name: None,
                    },
                    prompt_tokens,
                    completion_tokens,
                );

                Ok(Json(response).into_response())
            }
            Err(e) => {
                error!("Generation error: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
}

async fn completions(
    State(state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> Result<impl IntoResponse, StatusCode> {
    // TODO: Implement completions endpoint
    Ok(Json(serde_json::json!({
        "error": "Completions endpoint not yet implemented"
    })))
}

// Helper functions

fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|msg| {
            let attachment_text = if msg.attachments.is_empty() {
                String::new()
            } else {
                let details = msg
                    .attachments
                    .iter()
                    .map(|att| att.describe())
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("\n{}", details)
            };

            format!("{}: {}{}", msg.role, msg.content, attachment_text)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn generate_stream(
    engine: Arc<InferXEngine>,
    prompt: String,
    request: ChatCompletionRequest,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        // TODO: Implement actual streaming generation
        // For now, generate the full response and chunk it

        match engine.generate(
            &prompt,
            request.temperature,
            request.top_p,
            request.top_k,
            request.max_tokens as i32,
        ) {
            Ok(response_text) => {
                // Send initial chunk with role
                let initial_chunk = ChatCompletionChunk {
                    id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    object: "chat.completion.chunk".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: request.model.clone(),
                    choices: vec![ChatCompletionDelta {
                        index: 0,
                        delta: DeltaContent {
                            role: Some("assistant".to_string()),
                            content: None,
                        },
                        finish_reason: None,
                    }],
                };

                yield Ok(Event::default()
                    .data(serde_json::to_string(&initial_chunk).unwrap()));

                // Send content in chunks
                for chunk in response_text.chars().collect::<Vec<_>>().chunks(10) {
                    let content_chunk = ChatCompletionChunk {
                        id: initial_chunk.id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: initial_chunk.created,
                        model: request.model.clone(),
                        choices: vec![ChatCompletionDelta {
                            index: 0,
                            delta: DeltaContent {
                                role: None,
                                content: Some(chunk.iter().collect()),
                            },
                            finish_reason: None,
                        }],
                    };

                    yield Ok(Event::default()
                        .data(serde_json::to_string(&content_chunk).unwrap()));

                    tokio::time::sleep(Duration::from_millis(20)).await;
                }

                // Send final chunk
                let final_chunk = ChatCompletionChunk {
                    id: initial_chunk.id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: initial_chunk.created,
                    model: request.model.clone(),
                    choices: vec![ChatCompletionDelta {
                        index: 0,
                        delta: DeltaContent {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                };

                yield Ok(Event::default()
                    .data(serde_json::to_string(&final_chunk).unwrap()));

                // Send [DONE] marker
                yield Ok(Event::default().data("[DONE]"));
            }
            Err(e) => {
                error!("Stream generation error: {}", e);
                yield Ok(Event::default().data("[DONE]"));
            }
        }
    }
}
