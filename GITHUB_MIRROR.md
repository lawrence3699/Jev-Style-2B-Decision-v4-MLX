# GitHub mirror of `Jev-Style-2B-Decision-v4-MLX`

This repository mirrors the public Hugging Face model at:

<https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX>

Snapshot of Hugging Face revision `a14bbe4dafb154f0b05884314c8e227b1c6a5cd6`.

Files smaller than 100 MiB are stored on the `main` branch. Larger files are attached to this GitHub Release:

<https://github.com/lawrence3699/Jev-Style-2B-Decision-v4-MLX/releases/tag/huggingface-snapshot-2026-10-28>

See `RELEASE_ASSETS.tsv` for asset names, original paths, sizes, and SHA-256 checksums. Asset names use `__` in place of `/` for files stored in subfolders.

## Reassembling split files

Assets ending in `.part-aa`, `.part-ab`, and so on are consecutive pieces of one original file. Download every piece and concatenate them in lexical order. For example:

```bash
cat model.safetensors.part-* > model.safetensors
```

Verify the reconstructed file against the `original_sha256` value in `RELEASE_ASSETS.tsv`.
