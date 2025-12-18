# %%
# We use --prerelease=allow to get the nightly builds
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --prerelease=allow

# %%
# Install the vLLM nightly wheel directly
uv pip install vllm --torch-backend=auto

# %%
uv pip install unsloth unsloth_zoo bitsandbytes

# %%
uv pip install -U transformers


