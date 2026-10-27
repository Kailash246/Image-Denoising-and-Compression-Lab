"""Backend: preprocessing, model loading, noise injection, inference, metrics."""
import glob, os, re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from vae import ConvVAE, PATCH

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_cache = {}


# ---------- preprocessing ----------
def preprocess(img: Image.Image, size: int = 128) -> torch.Tensor:
    """PIL -> (1,3,size,size) float tensor in [0,1]. size must be a multiple of 32."""
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_patches(x):  # (1,3,H,W) -> (N,3,32,32)
    p = F.unfold(x, PATCH, stride=PATCH)               # (1, 3*32*32, N)
    return p.permute(0, 2, 1).reshape(-1, 3, PATCH, PATCH)


def from_patches(p, size):  # (N,3,32,32) -> (1,3,size,size)
    cols = p.reshape(1, -1, 3 * PATCH * PATCH).permute(0, 2, 1)
    return F.fold(cols, (size, size), PATCH, stride=PATCH)


def to_pil(x) -> Image.Image:
    arr = (x.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------- model loading ----------
def available_dims():
    pattern = os.path.join(MODEL_DIR, "vae_*.pt")
    dims = []
    for f in glob.glob(pattern):
        match = re.search(r"vae_(\d+)\.pt", f)
        if match:
            dims.append(int(match.group(1)))
    return sorted(dims)


def load_model(latent_dim: int) -> ConvVAE:
    if latent_dim not in _cache:
        m = ConvVAE(latent_dim).to(DEV)
        model_path = os.path.join(MODEL_DIR, f"vae_{latent_dim}.pt")
        m.load_state_dict(torch.load(model_path, map_location=DEV, weights_only=True))
        _cache[latent_dim] = m.eval()
    return _cache[latent_dim]


# ---------- noise ----------
def add_noise(x, sigma: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (x + torch.randn(x.shape, generator=g) * sigma).clamp(0, 1)


# ---------- inference ----------
@torch.no_grad()
def encode_decode(x, latent_dim: int):
    """Real pass through encoder -> latent (mean) -> decoder, patch by patch."""
    model = load_model(latent_dim)
    size = x.shape[-1]
    patches = to_patches(x).to(DEV)
    mu, _ = model.encode(patches)
    recon = model.decode(mu)
    return from_patches(recon.cpu(), size), mu.cpu()


# ---------- metrics ----------
def mse(a, b) -> float:
    return F.mse_loss(a, b).item()


def psnr(m: float) -> float:
    return float("inf") if m == 0 else 10 * np.log10(1.0 / m)


def run_pipeline(img, latent_dim, sigma, size=128, seed=0):
    x = preprocess(img, size)
    noisy = add_noise(x, sigma, seed)
    recon, latents = encode_decode(noisy, latent_dim)
    m_noisy, m_rec = mse(noisy, x), mse(recon, x)
    # Information-loss baseline: same image, no noise
    recon_clean, _ = encode_decode(x, latent_dim)
    m_clean = mse(recon_clean, x)
    raw = PATCH * PATCH * 3
    return dict(
        original=x, noisy=noisy, recon=recon, latents=latents,
        error_map=(recon - x).abs().mean(1).squeeze(0).numpy(),
        mse_noisy=m_noisy, mse_recon=m_rec, mse_clean_recon=m_clean,
        psnr_noisy=psnr(m_noisy), psnr_recon=psnr(m_rec),
        compression_ratio=raw / latent_dim,
        latent_floats=latents.numel(), pixel_values=x.numel(),
    )


def sweep(img, sigma, size=128, seed=0):
    """Reconstruction quality across every trained latent size (information-loss test)."""
    x = preprocess(img, size)
    noisy = add_noise(x, sigma, seed)
    rows = []
    for d in available_dims():
        r_noisy, _ = encode_decode(noisy, d)
        r_clean, _ = encode_decode(x, d)
        rows.append(dict(latent_dim=d, compression=PATCH * PATCH * 3 / d,
                         mse_noisy_input=mse(r_noisy, x), mse_clean_input=mse(r_clean, x)))
    return rows
