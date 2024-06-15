import sys

import torch

M = int(sys.argv[1])
N = int(sys.argv[2])
iters = int(sys.argv[3])
dtype = torch.float16

A = torch.randn(M, N, dtype=dtype, device="cuda")
cache = torch.empty(3 * 1024 * 1024, dtype=torch.uint8)

@torch.compile
def fn(x):
    return torch.nn.functional.relu(x)
fn(A)

events = [
    (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    for _ in range(iters)
] 
for start, end in events:
    cache.zero_()
    start.record()
    _ = fn(A)
    end.record()

torch.cuda.synchronize()

total_duration = 0
for start, end in events:
    total_duration += start.elapsed_time(end)

flops = M * N * iters / (total_duration / 1000)
bandwidth = 2 * M * N * iters * torch.finfo(dtype).bits / 8 / (total_duration / 1000)

print(f"Time elapsed: {total_duration}ms")
print(f"TFLOPS: {flops / 1e12}")
print(f"Bandwidth: {bandwidth / 1e9}GB/s")