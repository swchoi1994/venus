use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response, sse::Event},
    routing::{get, post},
    Json, Router,
};
use axum::response::sse::{KeepAlive, Sse};
use futures::stream::Stream;
use std::{path::PathBuf, sync::Arc, time::Duration, convert::Infallible};
use tower_http::cors::CorsLayer;
use tracing::{info, error};

use crate::engine::EngineManager;
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
}

#[derive(Clone)]
struct AppState {
    engine_manager: Arc<EngineManager>,
}

impl ApiServer {
    pub async fn new(config: ServerConfig) -> anyhow::Result<Self> {
        let engine_manager = Arc::new(EngineManager::new());
        
        // Load models from directory
        // TODO: Scan model directory and load models
        
        Ok(Self {
            config,
            engine_manager,
        })
    }
    
    pub async fn run(self) -> anyhow::Result<()> {
        let state = AppState {
            engine_manager: self.engine_manager.clone(),
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
    let models = state.engine_manager.list_models();
    
    let model_list = ModelList {
        object: "list".to_string(),
        data: models.into_iter().map(|id| ModelInfo {
            id: id.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "venus".to_string(),
        }).collect(),
    };
    
    Json(model_list)
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, StatusCode> {
    // Get the engine for the requested model
    let engine = state.engine_manager
        .get_engine(&request.model)
        .ok_or(StatusCode::NOT_FOUND)?;
    
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
                    request.model,
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
    messages.iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn generate_stream(
    engine: Arc<crate::engine::VenusEngine>,
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