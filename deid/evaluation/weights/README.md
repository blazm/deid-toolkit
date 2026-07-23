# Pre-trained model weights

Place the following pre-trained weights in this directory.

| File | Eval script |
|---|---|
| `affecnet8_epoch5_acc0.6209.pth` | `deid/evaluation/dan.py` |
| `face_gender_classification_transfer_learning_with_ResNet18.pth` | `deid/evaluation/restnet18_GD.py` |
| `VGG_FACE.t7` | `deid/evaluation/vggface.py`, `vggface_optimized.py` |
| `checkpoint_step_79999_gpu_0.pt` | `deid/evaluation/swinface.py` |
| `adaface_ir50_ms1mv2.ckpt` | `deid/evaluation/adaface_iv.py`, `adaface_optimized.py` |

## Notes

- **VGG-Face**: [DeepFace model-zoo](https://github.com/deepinsight/insightface/tree/master/model-zoo)
- **AdaFace**: [AdaFace pretrained](https://github.com/prcwg/adaface)
- **SWINface**: [SWINface releases](https://github.com/swin-face/SWINface)
- **AffectNet / ResNet18**: Source models from their respective projects
