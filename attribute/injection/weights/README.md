# injection/weights/

The generic extraction script (`extract_attribute_generic.py`) loads the FAD **attribute encoder**
checkpoint via `--gnet_weights` (default `weights/G_net_epoch_1000.pth`).

Place the trained FAD `att_encoder` checkpoint from **Stage 1** here (or point `--gnet_weights`
to wherever it is), e.g.:

```bash
cp ../../../stage1_fad/attribute/models/att_encoder_epoch_1000.pth weights/G_net_epoch_1000.pth
```

(Paths are relative to `attribute/injection/`; run the scripts from that folder.)
