import argparse
import torch
from torch import nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print(f"first feed-forward layer output has dtype {x.dtype}")
        x = self.relu(x)
        x = self.ln(x)
        print(f"layer norm output has dtype {x.dtype}")
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main() -> None:
    parser = argparse.ArgumentParser(description="benchmarking mixed precision")

    parser.add_argument("--dtype", type=str, required=True)
    args = parser.parse_args()


    dtype: torch.dtype | None = None
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        raise ValueError("Unexpected dtype")

    in_features: int = 10
    out_features: int = 10
    device = torch.device("cuda")
    # we have to explictly move it to cuda to benefit from autocast
    model: nn.Module = ToyModel(in_features, out_features).to(device)
    inputs = torch.rand(10, 10).to(device)

    print(f"auto cast to {dtype}")
    with torch.autocast(device_type="cuda", dtype=dtype):
        print("Autocast enabled ", torch.is_autocast_enabled("cuda"))
        print("Autocast dtype ", torch.get_autocast_dtype("cuda"))
        print("param device ", next(model.parameters()).device)

        for name, param in model.named_parameters():
            print(f"parameter {name} has data type {param.dtype}")

        print(f"inputs has dtype {inputs.dtype}")
        logits = model(inputs)
        print(f"logts has dtype {logits.dtype}")
        loss = logits.mean()
        print(f"loss has type {loss.dtype}")

        loss.backward()

        for name, param in model.named_parameters():
            print(f"parameter {name} 's gradient has dtype {param.grad.dtype if param.grad is not None else None}")

if __name__ == "__main__":
    main()
