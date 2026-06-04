use std::sync::Arc;

// HTTP boundary for decoding requests, queuing inference work, and shaping API errors.

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};

use crate::{
    audio::{decode_16k_mono_audio, normalize},
    batch::BatchRequest,
    classifier::GenderResponse,
};

#[derive(Clone)]
struct AppState {
    batcher: mpsc::Sender<BatchRequest>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error: self.message,
            }),
        )
            .into_response()
    }
}

pub(crate) fn router(batcher: mpsc::Sender<BatchRequest>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/gender", post(classify_gender))
        .with_state(Arc::new(AppState { batcher }))
}

async fn healthz() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn classify_gender(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Json<GenderResponse>, ApiError> {
    let samples =
        decode_16k_mono_audio(&body).map_err(|err| ApiError::bad_request(err.to_string()))?;
    let samples = normalize(&samples);
    let (tx, rx) = oneshot::channel();
    state
        .batcher
        .send(BatchRequest {
            samples,
            response: tx,
        })
        .await
        .map_err(|_| ApiError::unavailable("batch worker is not running"))?;

    let response = rx
        .await
        .map_err(|_| ApiError::unavailable("batch worker dropped the request"))?
        .map_err(ApiError::unavailable)?;
    Ok(Json(response))
}
