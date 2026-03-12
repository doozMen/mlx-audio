# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is mlx-audio?

A Python audio inference library for Apple Silicon, built on Apple's MLX framework. Provides local TTS, STT, Speech-to-Speech, VAD, audio codecs, and language identification — 30+ models with no cloud dependencies.

## Build & Install

```bash
pip install -e .                # Editable install (core only)
pip install -e ".[tts]"         # Text-to-Speech deps
pip install -e ".[stt]"         # Speech-to-Text deps
pip install -e ".[sts]"         # Speech-to-Speech deps
pip install -e ".[server]"      # FastAPI server deps
pip install -e ".[all,dev]"     # Everything + dev tools
```

Requires Python 3.10+ on Apple Silicon. ffmpeg needed for MP3/FLAC (not WAV).

## Testing

```bash
pytest mlx_audio/tests/         # Core (DSP, utils, lazy imports)
pytest mlx_audio/tts/tests/     # TTS models
pytest mlx_audio/stt/tests/     # STT models
pytest mlx_audio/sts/tests/     # STS models
pytest mlx_audio/codec/tests/   # Audio codecs
pytest mlx_audio/vad/tests/     # Voice Activity Detection
```

CI runs a matrix: style → core → modular (stt/tts/sts separately) → full suite. All on macOS-14.

## Code Quality

```bash
pre-commit run --all            # Black + isort (preferred)
black mlx_audio/                # Format (88-char lines)
isort mlx_audio/                # Import sort (Black profile)
```

Config lives in `pyproject.toml` under `[tool.black]` and `[tool.isort]`.

## CLI Entry Points

```bash
mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 --text 'Hello' --voice af_heart
mlx_audio.stt.generate --model mlx-community/whisper-large-v3-turbo-asr-fp16 --audio file.wav
mlx_audio.convert --hf-path <source> --mlx-path <dest> --dtype bfloat16
mlx_audio.server --host 0.0.0.0 --port 8000
```

## Architecture

### Module Map

| Module | Purpose |
|--------|---------|
| `tts/` | Text-to-Speech — 23 model implementations (Kokoro, Qwen3-TTS, CSM, OuteTTS, Soprano, etc.) |
| `stt/` | Speech-to-Text — Whisper, Qwen3-ASR, Parakeet, Voxtral, VibeVoice, MedASR |
| `sts/` | Speech-to-Speech — SAM-Audio (source separation), MossFormer2, LFM2.5 |
| `vad/` | Voice Activity Detection & Speaker Diarization — Sortformer |
| `codec/` | Audio codecs — 12 implementations (EnCodec, SNAC, Vocos, BigVGAN, ECAPA-TDNN, etc.) |
| `lid/` | Language Identification — MMS-LID |
| `dsp.py` | Pure MLX audio DSP — STFT/iSTFT, mel-filterbanks, window functions (no external deps) |

### Critical Patterns

**Lazy Imports** — The most important architectural constraint. Core (`dsp.py`, `utils.py`) imports only mlx/numpy/huggingface_hub. TTS/STT/STS deps are deferred so users install only what they need. `mlx_audio/tests/test_lazy_imports.py` enforces this via subprocess isolation (pytest collection pollutes `sys.modules`). Breaking lazy imports breaks the modular install story.

**Model Loading Flow:**
1. `load()` / `load_model()` from category utils (e.g., `tts.utils`, `stt.utils`)
2. Model identified from HuggingFace repo name
3. Downloaded via `huggingface_hub` if not cached
4. Config from `config.json` → model class resolved via `MODEL_REMAPPING` dict
5. Weights from `.safetensors` → `nn.Module` instantiated and returned

**Generation Pattern:**
```python
for result in model.generate(...):
    audio = result.audio  # mx.array
```
Models return generators yielding dataclass results with `.audio` and optional metadata. Supports streaming.

**MLX Specifics:**
- All tensors are `mx.array` (not numpy/torch)
- Models extend `mlx.nn.Module`
- Lazy evaluation by default — computation deferred until values needed
- Weight loading via `.load_weights()` from flattened dict (`tree_flatten()`)

**Model Config Pattern:**
Each model has a `ModelConfig` dataclass with `from_dict()` classmethod. Enables flexible config serialization and supports Optional types, nested dataclasses.

### Server

`server.py` provides OpenAI-compatible REST API endpoints for TTS and STT via FastAPI. Run with `mlx_audio.server`.

### Web UI

Next.js frontend in `mlx_audio/ui/` with 3D audio visualization. Run with `cd mlx_audio/ui && npm install && npm run dev`.

## Key Dependencies

- **mlx** ≥0.25.2 — Apple's ML framework (Apple Silicon only)
- **transformers** ==5.0.0rc3 — pinned to specific RC
- **mlx-lm** ==0.30.5 — MLX language model support
- **librosa** ==0.11.0 — audio feature extraction
- **miniaudio** ≥1.61 — audio I/O (WAV/MP3/FLAC)

## Adding a New Model

Each model lives in its own subdirectory under the appropriate category (e.g., `tts/models/mymodel/`). Must include:
- `__init__.py` with model class
- Config dataclass with `from_dict()`
- Entry in the category's `MODEL_REMAPPING` dict (in category `utils.py`)
- Weights loaded from `.safetensors` via standard `load_weights()` pattern

Ensure new imports don't leak into core — respect the lazy import boundary.
