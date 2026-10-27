import torch

def main() -> None:
    print("float32 += float32")
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
        print(s)

    print("float16 += float16")
    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
        print(s)

    print("float32 += float16")
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
        print(s)

    print("float32 + float16")
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
        print(s)

if __name__ == "__main__":
    main()
