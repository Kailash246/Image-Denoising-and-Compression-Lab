"""Train denoising VAEs (one per latent size) on CIFAR-10.
Input = noisy image, target = clean image, so the model learns to denoise.
Usage: python train.py --dims 16 64 128 256 --epochs 15
"""
import argparse, os
import torch
from torchvision import datasets
from vae import ConvVAE, vae_loss

p = argparse.ArgumentParser()
p.add_argument("--dims", type=int, nargs="+", default=[16, 64, 128, 256])
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--max_sigma", type=float, default=0.3)

if __name__ == "__main__":
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and hasattr(os, "cpu_count"):
        torch.set_num_threads(os.cpu_count() or 4)

    data = datasets.CIFAR10("data", train=True, download=True)
    # Pre-load into in-memory float tensor for high-throughput batching on CPU
    x_all = torch.from_numpy(data.data).permute(0, 3, 1, 2).float().div_(255.0).to(dev)
    n_samples = len(x_all)
    os.makedirs("models", exist_ok=True)

    for d in args.dims:
        model = ConvVAE(d).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1.5e-3)
        for ep in range(args.epochs):
            perm = torch.randperm(n_samples, device=dev)
            tot = 0.0
            for i in range(0, n_samples, args.batch):
                x = x_all[perm[i:i + args.batch]]
                if torch.rand(1).item() > 0.5:
                    x = torch.flip(x, [-1])
                sigma = torch.rand(x.size(0), 1, 1, 1, device=dev) * args.max_sigma
                noisy = (x + torch.randn_like(x) * sigma).clamp(0, 1)
                recon, mu, lv = model(noisy)
                loss, rec, kl = vae_loss(recon, x, mu, lv)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * x.size(0)
            print(f"[latent={d}] epoch {ep+1}/{args.epochs} loss={tot/n_samples:.2f}")
        model_path = os.path.join("models", f"vae_{d}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"saved {model_path}")
