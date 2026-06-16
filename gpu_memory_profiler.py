import torch

class GPUMemoryProfiler:
    def __init__(self, device=0):
        self.device = device
        self.snapshots = []
        self.labels = []

    def snapshot(self, label=""):
        torch.cuda.synchronize(self.device)
        alloc = torch.cuda.memory_allocated(self.device) / 1024**2
        self.snapshots.append(alloc)
        self.labels.append(label)
        return alloc

    def report(self):
        print("\nGPU Memory Report")
        print("=" * 50)
        for i, (s, l) in enumerate(zip(self.snapshots, self.labels)):
            d = ""
            if i > 0:
                diff = s - self.snapshots[i-1]
                d = f" ({chr(43) if diff >= 0 else chr(32)}{diff:.1f} MB)"
            print(f"  [{i:3d}] {l:25s} | {s:8.1f} MB{d}")
        print(f"\nPeak: {max(self.snapshots):.1f} MB")

def demo():
    p = GPUMemoryProfiler()
    p.snapshot("Start")
    m = torch.nn.Linear(1000, 1000).cuda()
    p.snapshot("After load")
    for i in range(5):
        x = torch.randn(32, 1000).cuda()
        m(x)
        p.snapshot(f"Forward {i}")
    p.report()

if __name__ == "__main__":
    demo()
