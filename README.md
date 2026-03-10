# demucs-mlx-swift

Swift package for Demucs-style music source separation, designed for **macOS + iOS** apps and structured for **MLX backend integration**.

## What This Repo Contains

- `DemucsMLX` library target (importable from iOS/macOS apps)
- `demucs-mlx-swift` CLI executable target (demo app)
- chunked overlap-add inference pipeline (split/overlap/batch/shifts)
- WAV audio loading and stem writing utilities
- model registry and stable public separation API

## Current Status

This repo implements the separation pipeline and public API with an MLX Demucs runtime backend.

Model files are resolved from local paths first and then downloaded from Hugging Face if missing.

## Requirements

- Swift 6.2+
- macOS 14+ or iOS 17+
- Xcode 15+
- Apple Silicon recommended

## Installation (SPM)

```swift
dependencies: [
    .package(url: "https://github.com/kylehowells/demucs-mlx-swift", branch: "master")
]
```

Then add product dependency and import:

```swift
import DemucsMLX
```

## Library Usage

```swift
import DemucsMLX

let separator = try DemucsSeparator(
    modelName: "htdemucs",
    parameters: DemucsSeparationParameters(
        shifts: 1,
        overlap: 0.25,
        split: true,
        segmentSeconds: 8.0,
        batchSize: 8,
        seed: nil
    )
)

let input = try AudioIO.loadAudio(from: URL(fileURLWithPath: "song.wav"))
let result = try separator.separate(audio: input)

for (source, audio) in result.stems {
    try AudioIO.writeWAV(audio, to: URL(fileURLWithPath: "\(source).wav"))
}
```

## CLI Demo

Build:

```bash
swift build -c release
```

Run:

```bash
.build/release/demucs-mlx-swift track.wav -o separated
```

Options:

- `--list-models`
- `-n, --name`
- `--segment`
- `--overlap`
- `--shifts`
- `--seed`
- `-b, --batch-size`
- `--no-split`
- `-o, --out`
- `--model-dir`

Model resolution order for `htdemucs`:

1. Explicit `--model-dir` (or library `modelDirectory`)
2. `DEMUCS_MLX_SWIFT_MODEL_DIR`
3. Local defaults: `.scratch/models/<model>`, `Models/<model>`, `./<model>`
4. Hugging Face snapshot download (default repo: `iky1e/demucs-mlx`)

Environment overrides:

- `DEMUCS_MLX_SWIFT_MODEL_REPO` can be set to a Hub repo ID (`org/repo`) or URL (`https://huggingface.co/org/repo`).
- Default repo currently includes `htdemucs`; other registry names require corresponding files in your local dir or selected Hub repo.

## Metal Shader Library (Required for MLX Inference)

MLX inference requires `mlx.metallib`.

After `swift build`, generate it with:

```bash
./scripts/build_mlx_metallib.sh release
```

If you run an `xcodebuild`/DerivedData binary, place `mlx.metallib` next to that executable.

If you see `missing Metal Toolchain`, run:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Make Targets

```bash
make build   # release + mlx.metallib
make debug   # debug + mlx.metallib
make test
make clean
```

## Roadmap

1. Add additional converted checkpoints and configs (`htdemucs_ft`, `htdemucs_6s`, etc.) to the default Hub repo.
2. Add regression tests against reference stems.
3. Add performance tuning (kernel fusion/custom kernels) for faster inference.
