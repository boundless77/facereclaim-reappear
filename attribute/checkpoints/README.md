# checkpoints/

Place the **ArcFace** identity-extraction model here (required by `attribute/model.py`):

| File | What | Required by |
|------|------|-------------|
| `arcface_checkpoint.tar` | The authors' ArcFace identity model (architecture in `models/arcface_models.py`), serialized so that `torch.load` returns the model directly (a *pickled model*; the `models` package must be importable, which is why runs use `PYTHONPATH=<repo-root>`). **Provided by the authors.** Note: official insightface/arcface_torch checkpoints are *state_dicts* and are not directly compatible; if you do not use the released checkpoint, retrain / convert it to match `models/arcface_models.py`. | `attribute/model.py` (IDExtractor) |

> The code loads `checkpoints/arcface_checkpoint.tar` relative to this stage folder
> (run scripts from the stage folder, or adjust the path).
