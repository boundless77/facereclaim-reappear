# checkpoints/

Place the pretrained weights here (they are **not bundled** with the repository):

| File | What | Required by |
|------|------|-------------|
| `last.ckpt` | The FaceReclaim diffusion model trained end-to-end (FRD + FAD + FIP, ~1.3B parameters). **Provided by the authors.** | `inference.py` (Stage-2 restore) / `main.py` resume |
| `vae.ckpt` | The image autoencoder (VAE, downsampling factor f=4, AutoencoderKL) **pretrained by the authors** on CelebA-HQ, FFHQ and VGGFace2 (see the paper, "Face Embedding"). **This is NOT the official Stable Diffusion VAE.** Provided by the authors. | `configs/FaceReclaim_s1.yaml` -> `first_stage_config.ckpt_path` |
| `arcface_checkpoint.tar` | The authors' ArcFace identity model (architecture in `models/arcface_models.py`), serialized so that `torch.load` returns the model directly (a *pickled model*; the `models` package must be importable, which is why runs use `PYTHONPATH=.../stage3_facereclaim`). **Provided by the authors.** Note: official insightface/arcface_torch checkpoints are *state_dicts* and are not directly compatible; if you do not use the released checkpoint, retrain / convert it to match `models/arcface_models.py`. | `ldm/modules/encoders/modules.py` (IDExtractor, FIP) |

> The **CLIP ViT-L/14** text encoder is the official OpenAI weight (not fine-tuned);
> download it yourself - see `ldm/modules/encoders/openai/clip-vit-large-patch14/README.md`.
>
> Paths in code/config are relative to this stage folder (e.g. `./checkpoints/vae.ckpt`,
> `checkpoints/arcface_checkpoint.tar`); run scripts from the stage folder.
