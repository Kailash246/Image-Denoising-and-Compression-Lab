# Task for the coding agent: finish, run and verify this project

Project: Image Denoising and Compression Lab (Streamlit + PyTorch VAE). Code is already written. Your job is to get it fully working locally and verified.

## Steps
1. Create a virtual environment and run `pip install -r requirements.txt`.
2. Run `python train.py --dims 16 64 128 256 --epochs 15`. If there is no GPU and it is too slow, use `--epochs 8`. Confirm that `models/vae_16.pt`, `vae_64.pt`, `vae_128.pt` and `vae_256.pt` exist.
3. Fix any errors you hit in the code (imports, shapes, deprecated Streamlit args such as `use_container_width`) without changing the architecture. Every image must still pass through the VAE encoder and decoder.
4. Run `streamlit run app.py`, upload a sample photo, and check that:
   - original, noisy and reconstructed images all display
   - the noise slider and compression slider both change the output
   - MSE, PSNR, compression ratio, error map, latent histogram and the sweep chart/table all render
   - reconstruction MSE is lower than noisy MSE at σ around 0.15 to 0.3
   - MSE rises as latent size falls (256 -> 16)
5. Save a screenshot to `screenshot.png`.
6. Update README.md with any changes and the measured results table (latent size vs MSE).

## Constraints
- Keep Streamlit as the frontend and PyTorch as the model framework.
- Do not replace the VAE with resizing or a standard filter.
- Keep `requirements.txt` accurate.
