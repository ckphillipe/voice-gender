# voice-gender

Small Rust HTTP service for voice gender classification from 16 kHz mono WAV
audio using a Wav2Vec2/Candle model.

The default model is
[`norwoodsystems/norwood-maleVSfemale`](https://huggingface.co/norwoodsystems/norwood-maleVSfemale).
Model files are downloaded through the Hugging Face cache on first run.

## Status

This project is early-stage. Treat predictions as model outputs, not verified
identity, demographic, medical, or forensic facts.

## Requirements

- Rust stable
- Ubuntu 24.04 or a comparable Linux environment
- NVIDIA driver and CUDA toolkit for GPU inference

## Install On Ubuntu 24.04

Install the base build tools and Rust:

```bash
sudo apt update
sudo apt install -y build-essential ca-certificates curl pkg-config wget libssl-dev

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup default stable
```

Build the CPU binary:

```bash
cargo build --release
```

Run on CPU:

```bash
./target/release/voice-gender --device cpu --listen 127.0.0.1:3000
```

## CUDA On Ubuntu 24.04

Use NVIDIA's Ubuntu 24.04 apt repository rather than the Ubuntu archive
`nvidia-cuda-toolkit` package.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

Install a CUDA toolkit version supported by your installed NVIDIA driver. The
example below uses CUDA 12.8; change both variables if you choose another minor.

```bash
export CUDA_APT_VERSION=12-8
export CUDA_HOME=/usr/local/cuda-${CUDA_APT_VERSION/-/.}

sudo apt install -y cuda-toolkit-${CUDA_APT_VERSION}
```

Make the toolkit available to Cargo and the runtime:

```bash
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

Check the driver and compiler are visible:

```bash
nvidia-smi
nvcc --version
```

Build with Candle CUDA support:

```bash
cargo build --release --features cuda
```

If the CUDA build fails with `NvccNotFound` or ``nvcc --version` failed`, the
toolkit is not visible to Cargo. Re-check `CUDA_HOME`, `PATH`, and
`nvcc --version` in the same shell used for the build.

Run on CUDA:

```bash
./target/release/voice-gender --device cuda --listen 127.0.0.1:3000
```

`--device auto` also tries CUDA first and falls back to CPU if CUDA cannot be
initialized.

## API

Health check:

```bash
curl -i http://127.0.0.1:3000/health
```

Classify a 16 kHz mono WAV file:

```bash
curl -X POST \
  --data-binary @voice.wav \
  http://127.0.0.1:3000/v1/gender
```

Classify raw 16 kHz mono signed 16-bit little-endian PCM:

```bash
curl -X POST \
  --data-binary @voice.s16le \
  http://127.0.0.1:3000/v1/gender
```

Example response:

```json
{
  "label": "female",
  "scores": [
    { "label": "female", "score": 0.98 },
    { "label": "male", "score": 0.02 }
  ]
}
```

The service auto-detects WAV bodies by their RIFF header. Bodies without a WAV
header are treated as raw 16 kHz mono signed 16-bit little-endian PCM. WAV input
is rejected if it is not mono or does not use a 16 kHz sample rate.

## Request Batching

Each HTTP request contains one audio sample to classify. Internally, the service
waits for a short configurable delay after the first queued request, gathers
additional pending requests, pads them to the longest sample in the group, and
runs one batched model forward pass.

`--max-batch-size` controls the maximum number of requests included in one
internal inference batch. `--batch-delay-ms` controls how long the worker waits
for more requests after receiving the first request in a batch:

```bash
./target/release/voice-gender --max-batch-size 32 --batch-delay-ms 80
```

Higher batch size and delay values can improve throughput when many clients send
requests at the same time, especially on CUDA. They may also add queueing
latency under load. Set `--batch-delay-ms 0` to avoid the extra batching wait.
This does not change the API shape: clients still send one WAV or raw PCM body
per request.

## Configuration

```text
--listen <ADDR>          REST API listen address (default: 127.0.0.1:3000)
--model <MODEL_ID>       Hugging Face model id
--max-batch-size <N>     Maximum queued requests per batch
--batch-delay-ms <MS>    Milliseconds to wait for more requests per batch
--device <auto|cuda|cpu> Inference device
```

## Development

```bash
cargo fmt
cargo test
```

## License

MIT

## References

- NVIDIA CUDA Linux installation guide: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- Rust toolchain installer: https://rustup.rs/
