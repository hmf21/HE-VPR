# Model weights

Place the following trained checkpoints in this directory before using `run.py extract`:

| File | Task | SHA-256 |
| --- | --- | --- |
| `vpr.ckpt` | VPR, epoch 44 | `C2F31D321CDF6FA54FADE5D5BA1719163662F817E7A5FB0A995B72AE5A5C3471` |
| `he.ckpt` | Height embedding, epoch 25 | `BA1E3D01AA4243147471980BDFDDFB14CB16CDF98C6F30B4AEE456A19CEEB067` |

Both checkpoints include the backbone. Upload them as GitHub Release assets, then replace these placeholders:

- VPR download: TODO: add Release asset URL
- HE download: TODO: add Release asset URL

Do not commit the checkpoint files as ordinary Git objects.
