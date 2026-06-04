use anyhow::{bail, Context, Result};

// Hugging Face model loading and batch prediction formatting.

use candle_core::{DType, Device, Tensor, D};
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::Serialize;

use crate::model::{Config, Wav2Vec2Classifier};

#[derive(Debug, Serialize)]
pub(crate) struct GenderResponse {
    pub(crate) label: String,
    pub(crate) scores: Vec<GenderScore>,
}

#[derive(Debug, Serialize)]
pub(crate) struct GenderScore {
    pub(crate) label: String,
    pub(crate) score: f32,
}

pub(crate) struct GenderClassifier {
    config: Config,
    model: Wav2Vec2Classifier,
    device: Device,
}

impl GenderClassifier {
    pub(crate) fn load(model_id: &str, device: Device) -> Result<Self> {
        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());
        let api = Api::new()?.repo(repo);
        let config_path = api.get("config.json")?;
        let weights_path = api.get("model.safetensors")?;

        let config: Config =
            serde_json::from_slice(&std::fs::read(&config_path).context("read config.json")?)?;
        let vb = unsafe {
            // Candle memory-maps trusted safetensors fetched from the configured model repo.
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };
        let model = Wav2Vec2Classifier::load(&config, vb)?;
        Ok(Self {
            config,
            model,
            device,
        })
    }

    pub(crate) fn predict_batch(&self, samples: Vec<Vec<f32>>) -> Result<Vec<GenderResponse>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let sample_lengths: Vec<_> = samples.iter().map(Vec::len).collect();
        if sample_lengths
            .iter()
            .any(|&len| self.model.feature_output_len(len) == 0)
        {
            bail!("audio is too short for Wav2Vec2 feature extraction");
        }

        let max_len = sample_lengths.iter().copied().max().unwrap_or(0);
        let mut padded = Vec::with_capacity(samples.len() * max_len);
        for sample in samples {
            padded.extend(sample.iter().copied());
            padded.resize(padded.len() + max_len - sample.len(), 0.0);
        }

        let input = Tensor::from_vec(padded, (sample_lengths.len(), max_len), &self.device)?;
        let logits = self.model.forward(&input, &sample_lengths)?;
        let probabilities = candle_nn::ops::softmax(&logits, D::Minus1)?.to_vec2::<f32>()?;

        Ok(probabilities
            .into_iter()
            .map(|scores| self.format_scores(scores))
            .collect())
    }

    fn format_scores(&self, scores: Vec<f32>) -> GenderResponse {
        let mut scores: Vec<_> = scores
            .into_iter()
            .enumerate()
            .map(|(idx, score)| GenderScore {
                label: self
                    .config
                    .id2label
                    .get(&idx.to_string())
                    .cloned()
                    .unwrap_or_else(|| idx.to_string()),
                score,
            })
            .collect();
        scores.sort_by(|a, b| b.score.total_cmp(&a.score));
        let label = scores
            .first()
            .map(|score| score.label.clone())
            .unwrap_or_default();
        GenderResponse { label, scores }
    }
}
