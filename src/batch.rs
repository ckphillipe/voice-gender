use std::time::Duration;

// Small batching loop that keeps HTTP handlers async while running inference on one worker.

use tokio::sync::{mpsc, oneshot};

use crate::classifier::{GenderClassifier, GenderResponse};

pub(crate) struct BatchRequest {
    pub(crate) samples: Vec<f32>,
    pub(crate) response: oneshot::Sender<Result<GenderResponse, String>>,
}

pub(crate) fn batch_worker(
    classifier: GenderClassifier,
    mut rx: mpsc::Receiver<BatchRequest>,
    max_batch_size: usize,
    batch_delay: Duration,
) {
    while let Some(first) = rx.blocking_recv() {
        let mut batch = vec![first];
        // Trade a small delay for better GPU/CPU utilization under bursty load.
        std::thread::sleep(batch_delay);

        while batch.len() < max_batch_size {
            match rx.try_recv() {
                Ok(request) => batch.push(request),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }

        let samples = batch
            .iter()
            .map(|request| request.samples.clone())
            .collect::<Vec<_>>();
        match classifier.predict_batch(samples) {
            Ok(responses) => {
                for (request, response) in batch.into_iter().zip(responses) {
                    let _ = request.response.send(Ok(response));
                }
            }
            Err(err) => {
                let message = err.to_string();
                for request in batch {
                    let _ = request.response.send(Err(message.clone()));
                }
            }
        }
    }
}
