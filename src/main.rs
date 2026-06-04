mod api;
mod audio;
mod batch;
mod classifier;
mod model;

// CLI entrypoint: selects a device, loads the model, starts batching, and serves HTTP.

use std::{net::SocketAddr, time::Duration};

use anyhow::{Context, Result};
use candle_core::Device;
use clap::{Parser, ValueEnum};

use crate::{batch::batch_worker, classifier::GenderClassifier};

const MODEL_ID: &str = "norwoodsystems/norwood-maleVSfemale";

#[derive(Parser, Debug)]
struct Args {
    /// Address the REST API listens on.
    #[arg(long, default_value = "127.0.0.1:3000")]
    listen: SocketAddr,

    /// Hugging Face model id.
    #[arg(long, default_value = MODEL_ID)]
    model: String,

    /// Maximum number of queued requests per inference batch.
    #[arg(long, default_value_t = 32)]
    max_batch_size: usize,

    /// Milliseconds to wait for additional requests before running a batch.
    #[arg(long, default_value_t = 80)]
    batch_delay_ms: u64,

    /// Inference device.
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceArg {
    Auto,
    Cuda,
    Cpu,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let device = select_device(args.device)?;
    tracing::info!("using inference device: {device:?}");

    let classifier = GenderClassifier::load(&args.model, device)?;
    let (tx, rx) = tokio::sync::mpsc::channel(args.max_batch_size * 16);
    let batch_delay = Duration::from_millis(args.batch_delay_ms);
    std::thread::spawn(move || batch_worker(classifier, rx, args.max_batch_size, batch_delay));

    let app = api::router(tx);

    tracing::info!("listening on http://{}", args.listen);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn select_device(device: DeviceArg) -> Result<Device> {
    match device {
        DeviceArg::Cpu => Ok(Device::Cpu),
        DeviceArg::Cuda => Device::new_cuda(0).context("initialize CUDA device 0"),
        DeviceArg::Auto => match Device::new_cuda(0) {
            Ok(device) => Ok(device),
            Err(err) => {
                tracing::warn!("CUDA device unavailable, falling back to CPU: {err}");
                Ok(Device::Cpu)
            }
        },
    }
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::error!("failed to listen for shutdown signal: {err}");
    }
}
