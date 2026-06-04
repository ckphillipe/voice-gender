use std::io::Cursor;

// WAV decoding and sample normalization for the classifier input contract.

use anyhow::{bail, Context, Result};

const TARGET_SAMPLE_RATE: usize = 16_000;
const WAV_MAGIC: &[u8; 4] = b"RIFF";

pub(crate) fn normalize(samples: &[f32]) -> Vec<f32> {
    // Match the model's expected zero-mean, unit-variance waveform input.
    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    let variance = samples
        .iter()
        .map(|sample| {
            let centered = sample - mean;
            centered * centered
        })
        .sum::<f32>()
        / samples.len() as f32;
    let scale = (variance + 1e-7).sqrt();
    samples
        .iter()
        .map(|sample| (sample - mean) / scale)
        .collect()
}

pub(crate) fn decode_16k_mono_audio(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.is_empty() {
        bail!("request body is empty");
    }

    if bytes.starts_with(WAV_MAGIC) {
        decode_16k_mono_wav(bytes)
    } else {
        decode_16k_mono_pcm_s16le(bytes)
    }
}

fn decode_16k_mono_wav(bytes: &[u8]) -> Result<Vec<f32>> {
    let cursor = Cursor::new(bytes.to_vec());
    let mut reader =
        hound::WavReader::new(cursor).context("request body is not a valid WAV file")?;
    let spec = reader.spec();
    if spec.channels != 1 {
        bail!("expected mono WAV audio, got {} channels", spec.channels);
    }
    if spec.sample_rate as usize != TARGET_SAMPLE_RATE {
        bail!(
            "expected {TARGET_SAMPLE_RATE} Hz WAV audio, got {} Hz",
            spec.sample_rate
        );
    }

    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => match spec.bits_per_sample {
            8 => reader
                .samples::<i8>()
                .map(|s| s.map(|s| s as f32 / i8::MAX as f32))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            16 => reader
                .samples::<i16>()
                .map(|s| s.map(|s| s as f32 / i16::MAX as f32))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            24 | 32 => reader
                .samples::<i32>()
                .map(|s| s.map(|s| s as f32 / ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            bits => bail!("unsupported integer WAV bit depth: {bits}"),
        },
    };
    if samples.is_empty() {
        bail!("WAV file contains no samples");
    }
    Ok(samples)
}

fn decode_16k_mono_pcm_s16le(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        bail!("raw PCM body must contain an even number of bytes");
    }

    let samples = bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / i16::MAX as f32)
        .collect::<Vec<_>>();
    if samples.is_empty() {
        bail!("raw PCM body contains no samples");
    }
    Ok(samples)
}
