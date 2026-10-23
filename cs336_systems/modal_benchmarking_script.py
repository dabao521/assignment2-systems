import subprocess

import modal

app = modal.App("cs336-benchmarking")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(
        "/Users/jintaoshen/assignment2-systems",
        remote_path="/app",
        ignore=[
            ".venv/**",
            ".git/**",
            "data/**",
            "*.npy",
            "*.pt",
            "*.csv",
            "*.png",
            "__pycache__/**",
        ],
        copy=True
    )
    .run_commands(
        "pip install uv",
        "cd /app && uv sync --frozen",
    )
    .env({"PATH": "/app/.venv/bin:$PATH"})
    )

volume = modal.Volume.from_name("cs336-data")

@app.function(
    image=image,
    gpu="H200",
    volumes={"/data": volume},
    timeout=7200,
)
def run(
    mode: str,
    warmup: int,
    iterations: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
) -> None:
    import os

    os.chdir("/app")

    command: list[str] = [
        "python",
        "-u",
        "cs336_systems/benchmarking_script.py",
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--d-model", str(d_model),
        "--d-ff", str(d_ff),
        "--num-layers", str(num_layers),
        "--num-heads", str(num_heads),
        "--mode", mode,
        "--device", "cuda",
    ]

    try:
        subprocess.run(command, check=True)
    finally:
        volume.commit()

@app.local_entrypoint()
def main(
    mode: str,
    warmup: int,
    iterations: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    )-> None:
    run.remote(mode, warmup, iterations, d_model, d_ff, num_layers, num_heads)

