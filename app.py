import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

import backend as bk

st.set_page_config(page_title="Image Denoising & Compression Lab", layout="wide")
st.title("Image Denoising and Compression Lab")
st.caption("Image → (+noise) → VAE encoder → latent vector → VAE decoder → reconstruction")

dims = bk.available_dims()
if not dims:
    st.error("No trained models found. Run `python train.py` first (see README).")
    st.stop()

with st.sidebar:
    st.header("Controls")
    file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    use_sample = False
    if file is None and os.path.exists("sample.png"):
        use_sample = st.checkbox("Use sample photo", value=True)
    sigma = st.slider("Noise level (σ)", 0.0, 0.5, 0.15, 0.01)
    latent_dim = st.select_slider(
        "Compression (latent size)", options=dims, value=dims[len(dims) // 2],
        format_func=lambda d: f"{d} dims ({3072 / d:.0f}:1)")
    size = st.selectbox("Working resolution", [64, 128, 192, 256], index=1)
    seed = st.number_input("Noise seed", 0, 9999, 0)

if file is not None:
    img = Image.open(file)
elif use_sample and os.path.exists("sample.png"):
    img = Image.open("sample.png")
else:
    st.info("Upload an image in the sidebar to begin.")
    st.stop()
res = bk.run_pipeline(img, latent_dim, sigma, size, int(seed))

c1, c2, c3 = st.columns(3)
c1.image(bk.to_pil(res["original"]), caption="Original", width="stretch")
c2.image(bk.to_pil(res["noisy"]), caption=f"Noisy (σ={sigma})", width="stretch")
c3.image(bk.to_pil(res["recon"]), caption=f"Reconstructed (latent={latent_dim})", width="stretch")

st.subheader("Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("MSE noisy vs original", f"{res['mse_noisy']:.5f}")
m2.metric("MSE reconstruction vs original", f"{res['mse_recon']:.5f}",
          delta=f"{res['mse_recon'] - res['mse_noisy']:.5f}", delta_color="inverse")
m3.metric("PSNR (noisy → recon)", f"{res['psnr_recon']:.2f} dB",
          delta=f"{res['psnr_recon'] - res['psnr_noisy']:.2f} dB")
m4.metric("Compression ratio", f"{res['compression_ratio']:.0f}:1",
          help=f"{res['pixel_values']} pixel values → {res['latent_floats']} latent values")

st.caption(f"Information loss (no noise added): MSE = {res['mse_clean_recon']:.5f}. "
           "Reconstruction MSE below the noisy MSE means the VAE removed noise.")

e1, e2 = st.columns(2)
with e1:
    st.subheader("Reconstruction error map")
    fig, ax = plt.subplots()
    im = ax.imshow(res["error_map"], cmap="inferno"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig)
with e2:
    st.subheader("Latent code statistics")
    fig, ax = plt.subplots()
    ax.hist(res["latents"].flatten().numpy(), bins=40)
    ax.set_xlabel("latent value"); ax.set_ylabel("count")
    st.pyplot(fig)

st.subheader("Compression sweep (all trained models)")
df = pd.DataFrame(bk.sweep(img, sigma, size, int(seed)))
df["compression"] = df["compression"].round(1)
st.line_chart(df.set_index("latent_dim")[["mse_noisy_input", "mse_clean_input"]])
st.dataframe(df, width="stretch")
st.caption("Smaller latent → higher compression → higher MSE (more information lost).")
