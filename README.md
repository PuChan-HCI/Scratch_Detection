This directory contains cleaned visualization scripts for sharing.

What I changed
- Removed user-specific absolute paths.
- Scripts now accept CLI arguments and read `SCRATCH_MODEL` env var.
- No private Roboflow keys or other secrets are present.

Quick usage

Run with defaults (expects `model.onnx` and `video.mp4` in the current folder):

```
python visualize_scratches.py
```

Provide explicit paths:

```
python visualize_scratches.py --model path/to/model.onnx --video path/to/input.mp4 --output out.mp4 --results results.json
```

Or set the environment variable for the default model:

Windows PowerShell
```
$env:SCRATCH_MODEL = "path\to\model.onnx"
python visualize_scratches.py
```

Unix / macOS
```
export SCRATCH_MODEL=path/to/model.onnx
python visualize_scratches.py
```

Notes
- If you want different defaults for the output filenames, pass `--output` and `--results`.
- The scripts do not contain any API keys or external credentials. If your workflow requires Roboflow or similar, provide keys outside the script (env vars or a secrets manager).