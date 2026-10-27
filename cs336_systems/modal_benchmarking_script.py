import subprocess

import modal

app = modal.App("cs336-benchmarking")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "ca-certificates")
    .run_commands(
        "wget -O /tmp/nsys.deb "
        "'https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_5/nsightsystems-linux-cli-public-2026.5.1.161-3889610.deb'",
        "apt-get update",
        "apt-get install -y /tmp/nsys.deb",
        "rm /tmp/nsys.deb",
        "nsys --version",
    )
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
    context_length: int,
    use_mixed_precision: bool,
    profile_memory: bool,
) -> None:
    import os

    os.chdir("/app")

    command: list[str] = [
        "nsys",
        "profile",
        "--trace=cuda,cudnn,cublas,osrt,nvtx",
        "--pytorch=functions-trace,autograd-shapes-nvtx",
        #"--cudabacktrace=all",
        "--python-backtrace=cuda",
        "--gpu-metrics-devices=0",
        #"--capture-range=nvtx",
        #"--nvtx-capture=benchmark@*",
        "--sample=none",
        "--cpuctxsw=none",
        "--output=/data/transformer_profile",
        "--force-overwrite=true",
        "--capture-range-end=stop",
        "--capture-range=cudaProfilerApi",
        "--cuda-memory-usage", "true",
        "--",
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
        "--context-length", str(context_length),
        "--use-mixed-precision", str(use_mixed_precision),
        "--profile-memory", str(profile_memory),
    ]

    try:
        subprocess.run(command, check=True)
    finally:
        volume.commit()

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
)
def generate_reports() -> None:
    import subprocess

    reports: dict[str, str] = {
        "cuda_gpu_kern_sum": "kernel_stats.txt",
        "nvtx_sum": "nvtx_stats.txt",
        "cuda_api_sum": "cuda_api_stats.txt",
        "cuda_gpu_mem_size_sum": "cuda_gpu_mem_size_sum.txt"
    }

    for report_name, filename in reports.items():
        with open(f"/data/{filename}", "w") as output:
            subprocess.run(
                [
                    "nsys",
                    "stats",
                    "--report",
                    report_name,
                    "--force-export=true",
                    "/data/transformer_profile.nsys-rep",
                ],
                stdout=output,
                check=True,
            )

    volume.commit()

@app.function(
    image=image,
    gpu="H200",
    volumes={"/data": volume},
    timeout=7200,
)
def mixed_precision() -> None:
    import os

    os.chdir("/app")

    command: list[str] = [
        "python",
        "cs336_systems/benchmarking_mixed_precision.py",
        "--dtype", "float16",
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
    context_length: int,
    use_mixed_precision: bool,
    profile_memory: bool,
    )-> None:
    run.remote(
        mode,
        warmup,
        iterations,
        d_model,
        d_ff,
        num_layers,
        num_heads,
        context_length,
        use_mixed_precision,
        profile_memory,
    )

