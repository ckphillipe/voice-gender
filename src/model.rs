use std::collections::BTreeMap;

// Minimal Wav2Vec2 sequence-classification implementation for Candle safetensors.

use anyhow::Result;
use candle_core::{Device, Module, Tensor, D};
use candle_nn::{linear, Conv1d, Conv1dConfig, Dropout, GroupNorm, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct Config {
    activation_dropout: f64,
    attention_dropout: f64,
    classifier_proj_size: usize,
    conv_bias: bool,
    conv_dim: Vec<usize>,
    conv_kernel: Vec<usize>,
    conv_stride: Vec<usize>,
    #[serde(rename = "feat_extract_dropout")]
    _feat_extract_dropout: f64,
    feat_extract_norm: String,
    feat_proj_dropout: f64,
    final_dropout: f64,
    hidden_dropout: f64,
    hidden_size: usize,
    intermediate_size: usize,
    layer_norm_eps: f64,
    num_attention_heads: usize,
    num_conv_pos_embedding_groups: usize,
    num_conv_pos_embeddings: usize,
    num_hidden_layers: usize,
    pub(crate) id2label: BTreeMap<String, String>,
}

pub(crate) struct Wav2Vec2Classifier {
    feature_extractor: FeatureExtractor,
    feature_projection: FeatureProjection,
    encoder: Encoder,
    projector: Linear,
    classifier: Linear,
    final_dropout: Dropout,
}

impl Wav2Vec2Classifier {
    pub(crate) fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let wav2vec2 = vb.pp("wav2vec2");
        Ok(Self {
            feature_extractor: FeatureExtractor::load(cfg, wav2vec2.clone())?,
            feature_projection: FeatureProjection::load(cfg, wav2vec2.clone())?,
            encoder: Encoder::load(cfg, wav2vec2.pp("encoder"))?,
            projector: linear(
                cfg.hidden_size,
                cfg.classifier_proj_size,
                vb.pp("projector"),
            )?,
            classifier: linear(
                cfg.classifier_proj_size,
                cfg.id2label.len(),
                vb.pp("classifier"),
            )?,
            final_dropout: Dropout::new(cfg.final_dropout as f32),
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor, sample_lengths: &[usize]) -> Result<Tensor> {
        let xs = self.feature_extractor.forward(xs)?;
        let xs = self.feature_projection.forward(&xs)?;
        let frame_lengths = self.feature_lengths(sample_lengths);
        let frame_mask = frame_mask(&frame_lengths, xs.dim(1)?, xs.device())?;
        let attention_mask = attention_mask(&frame_mask)?;
        let xs = self.encoder.forward(&xs, Some(&attention_mask))?;
        let xs = masked_mean(&xs, &frame_mask)?;
        let xs = self.projector.forward(&xs)?.tanh()?;
        let xs = self.final_dropout.forward(&xs, false)?;
        Ok(self.classifier.forward(&xs)?)
    }

    pub(crate) fn feature_output_len(&self, sample_len: usize) -> usize {
        self.feature_extractor.output_len(sample_len)
    }

    fn feature_lengths(&self, sample_lengths: &[usize]) -> Vec<usize> {
        sample_lengths
            .iter()
            .map(|&len| self.feature_output_len(len))
            .collect()
    }
}

struct FeatureExtractor {
    layers: Vec<ConvLayer>,
}

struct ConvLayer {
    conv: Conv1d,
    norm: Option<GroupNorm>,
}

impl FeatureExtractor {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.conv_dim.len());
        let mut in_channels = 1;
        for (idx, ((&out_channels, &kernel_size), &stride)) in cfg
            .conv_dim
            .iter()
            .zip(&cfg.conv_kernel)
            .zip(&cfg.conv_stride)
            .enumerate()
        {
            let conv_cfg = Conv1dConfig {
                stride,
                padding: 0,
                dilation: 1,
                groups: 1,
                ..Default::default()
            };
            let conv_vb = vb.pp(format!("feature_extractor.conv_layers.{idx}.conv"));
            let conv = if cfg.conv_bias {
                candle_nn::conv1d(in_channels, out_channels, kernel_size, conv_cfg, conv_vb)?
            } else {
                candle_nn::conv1d_no_bias(
                    in_channels,
                    out_channels,
                    kernel_size,
                    conv_cfg,
                    conv_vb,
                )?
            };
            let norm = if idx == 0 && cfg.feat_extract_norm == "group" {
                Some(candle_nn::group_norm(
                    out_channels,
                    out_channels,
                    cfg.layer_norm_eps,
                    vb.pp(format!("feature_extractor.conv_layers.{idx}.layer_norm")),
                )?)
            } else {
                None
            };
            layers.push(ConvLayer { conv, norm });
            in_channels = out_channels;
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.unsqueeze(1)?;
        for layer in &self.layers {
            xs = layer.conv.forward(&xs)?;
            if let Some(norm) = &layer.norm {
                xs = norm.forward(&xs)?;
            }
            xs = xs.gelu()?;
        }
        Ok(xs)
    }

    fn output_len(&self, mut len: usize) -> usize {
        for layer in &self.layers {
            let cfg = layer.conv.config();
            let kernel = layer.conv.weight().dim(2).unwrap_or(1);
            if len < kernel {
                return 0;
            }
            len = (len + 2 * cfg.padding - cfg.dilation * (kernel - 1) - 1) / cfg.stride + 1;
        }
        len
    }
}

struct FeatureProjection {
    layer_norm: LayerNorm,
    projection: Linear,
    dropout: Dropout,
}

impl FeatureProjection {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            layer_norm: candle_nn::layer_norm(
                cfg.conv_dim[cfg.conv_dim.len() - 1],
                cfg.layer_norm_eps,
                vb.pp("feature_projection.layer_norm"),
            )?,
            projection: linear(
                cfg.conv_dim[cfg.conv_dim.len() - 1],
                cfg.hidden_size,
                vb.pp("feature_projection.projection"),
            )?,
            dropout: Dropout::new(cfg.feat_proj_dropout as f32),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.transpose(1, 2)?;
        let xs = self.layer_norm.forward(&xs)?;
        let xs = self.projection.forward(&xs)?;
        Ok(self.dropout.forward(&xs, false)?)
    }
}

struct PosConvEmbed {
    conv: Conv1d,
    padding: usize,
}

impl PosConvEmbed {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv1dConfig {
            stride: 1,
            padding: cfg.num_conv_pos_embeddings / 2,
            dilation: 1,
            groups: cfg.num_conv_pos_embedding_groups,
            ..Default::default()
        };
        let conv_vb = vb.pp("pos_conv_embed.conv");
        let weight_g = conv_vb.get(
            (1, 1, cfg.num_conv_pos_embeddings),
            "parametrizations.weight.original0",
        )?;
        let weight_v = conv_vb.get(
            (
                cfg.hidden_size,
                cfg.hidden_size / cfg.num_conv_pos_embedding_groups,
                cfg.num_conv_pos_embeddings,
            ),
            "parametrizations.weight.original1",
        )?;
        let weight_norm = weight_v
            .broadcast_mul(&weight_v)?
            .sum_keepdim((0, 1))?
            .sqrt()?;
        let weight = weight_v
            .broadcast_div(&weight_norm)?
            .broadcast_mul(&weight_g)?;
        let bias = conv_vb.get(cfg.hidden_size, "bias")?;
        Ok(Self {
            conv: Conv1d::new(weight, Some(bias), conv_cfg),
            padding: cfg.num_conv_pos_embeddings / 2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut ys = self.conv.forward(&xs.transpose(1, 2)?)?;
        if self.padding > 0 {
            let seq_len = xs.dim(1)?;
            ys = ys.narrow(2, 0, seq_len)?;
        }
        Ok(ys.gelu()?.transpose(1, 2)?)
    }
}

struct Attention {
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    dropout: Dropout,
}

impl Attention {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            k_proj: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))?,
            v_proj: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))?,
            q_proj: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?,
            out_proj: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("out_proj"))?,
            num_heads: cfg.num_attention_heads,
            head_dim,
            dropout: Dropout::new(cfg.attention_dropout as f32),
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq_len, hidden) = xs.dims3()?;
        let shape = (batch, seq_len, self.num_heads, self.head_dim);
        let q = self
            .q_proj
            .forward(xs)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_scores = (q.matmul(&k.t()?)? * scale)?;
        if let Some(mask) = attention_mask {
            attn_scores = attn_scores.broadcast_add(mask)?;
        }
        let attn = candle_nn::ops::softmax(&attn_scores, D::Minus1)?;
        let attn = self.dropout.forward(&attn, false)?;
        let ys = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, hidden))?;
        Ok(self.out_proj.forward(&ys)?)
    }
}

struct EncoderLayer {
    attention: Attention,
    dropout: Dropout,
    layer_norm: LayerNorm,
    feed_forward_intermediate: Linear,
    feed_forward_output: Linear,
    final_layer_norm: LayerNorm,
    activation_dropout: Dropout,
    hidden_dropout: Dropout,
}

impl EncoderLayer {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(cfg, vb.pp("attention"))?,
            dropout: Dropout::new(cfg.hidden_dropout as f32),
            layer_norm: candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("layer_norm"),
            )?,
            feed_forward_intermediate: linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("feed_forward.intermediate_dense"),
            )?,
            feed_forward_output: linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("feed_forward.output_dense"),
            )?,
            final_layer_norm: candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("final_layer_norm"),
            )?,
            activation_dropout: Dropout::new(cfg.activation_dropout as f32),
            hidden_dropout: Dropout::new(cfg.hidden_dropout as f32),
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = self.attention.forward(xs, attention_mask)?;
        let xs = self.dropout.forward(&xs, false)?;
        let xs = (xs + residual)?;
        let xs = self.layer_norm.forward(&xs)?;

        let residual = &xs;
        let xs = self.feed_forward_intermediate.forward(&xs)?.gelu()?;
        let xs = self.activation_dropout.forward(&xs, false)?;
        let xs = self.feed_forward_output.forward(&xs)?;
        let xs = self.hidden_dropout.forward(&xs, false)?;
        let xs = (xs + residual)?;
        Ok(self.final_layer_norm.forward(&xs)?)
    }
}

struct Encoder {
    pos_conv_embed: PosConvEmbed,
    layer_norm: LayerNorm,
    dropout: Dropout,
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::load(cfg, vb.pp(format!("layers.{idx}")))?);
        }
        Ok(Self {
            pos_conv_embed: PosConvEmbed::load(cfg, vb.clone())?,
            layer_norm: candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("layer_norm"),
            )?,
            dropout: Dropout::new(cfg.hidden_dropout as f32),
            layers,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let pos = self.pos_conv_embed.forward(xs)?;
        let mut xs = (xs + pos)?;
        xs = self.layer_norm.forward(&xs)?;
        xs = self.dropout.forward(&xs, false)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, attention_mask)?;
        }
        Ok(xs)
    }
}

fn frame_mask(frame_lengths: &[usize], max_len: usize, device: &Device) -> Result<Tensor> {
    let mut mask = Vec::with_capacity(frame_lengths.len() * max_len);
    for &len in frame_lengths {
        mask.extend((0..max_len).map(|idx| if idx < len { 1.0_f32 } else { 0.0_f32 }));
    }
    Ok(Tensor::from_vec(
        mask,
        (frame_lengths.len(), max_len),
        device,
    )?)
}

fn attention_mask(frame_mask: &Tensor) -> Result<Tensor> {
    let (batch, seq_len) = frame_mask.dims2()?;
    let mut values = Vec::with_capacity(batch * seq_len);
    for row in frame_mask.to_vec2::<f32>()? {
        values.extend(
            row.into_iter()
                .map(|value| if value > 0.5 { 0.0 } else { -1e9_f32 }),
        );
    }
    Ok(
        Tensor::from_vec(values, (batch, seq_len), frame_mask.device())?
            .unsqueeze(1)?
            .unsqueeze(1)?,
    )
}

fn masked_mean(xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let mask = mask.unsqueeze(2)?;
    let summed = xs.broadcast_mul(&mask)?.sum(1)?;
    let counts = mask.sum(1)?.clamp(1e-6, f32::MAX as f64)?;
    Ok(summed.broadcast_div(&counts)?)
}
