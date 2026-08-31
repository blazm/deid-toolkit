# De-identification baselines — toolkit batch scripts

This folder ships **our part** of the de-identification toolchain: the toolkit's own
prototype batch runner (`deidentify_batch*.py`) for each of the 20 approaches evaluated
in the study behind this toolkit, plus a few local utility scripts we wrote (e.g. NullFace's
local face-embedding extractor, RP's pipeline loader).

What is **deliberately NOT** here:

- the full official method repositories (they are large research codebases), and
- all model weights.

Per method you get:

1. our batch script(s) — the interface you run in the paper pipeline;
2. `weights_manifest.txt` — the exact weight files expected, with relative paths and
   sizes (generated from the working collection); obtain them from the method's official
   release (Hugging Face / Google Drive / OneDrive links are in the official repo's
   README). Keep the working directory layout shown in the manifest.

## Common conventions

All batch scripts share the same interface and behavior:

- **Input:** a folder of pre-aligned single-face images (any of the toolkit's aligned
  datasets, e.g. CelebA-test, RaFD, MUG-Still, KDEF);
- **Output:** PNGs in `--output` with the **same basenames** as the inputs (the
  aligned/de-identified pairing convention used by the evaluators);
- **Safe to re-run:** existing outputs are skipped;
- **Failures** (e.g. undetectable faces) are not written and are appended to a
  failure log next to the output folder;
- **Resolution:** 256×256 output unless the method's protocol dictates otherwise
  (noted below).

Typical call:

```bash
python deidentify_batch_X.py --input <aligned_dir> --output <deid_dir> [--batch-size N] [--seed 0] [--out-size 256]
```

Run inside the method's own conda environment (baselines have mutually incompatible
dependencies; one env per method).

## The 20 approaches

Status column ("verified") = the script produced complete, validated output for the
study on the research machine (2026-08). "Official code" links point to the
official implementation of each paper; place it alongside the runner script (our
scripts insert the parent directory into `sys.path`) and follow the official
README for its dependencies. Weight download links are listed in the official
repos (and summarized per method in `weights_manifest.txt`).

| Method | Paper | Official code / weights | Files here | Env (working config) | Weights |
|---|---|---|---|---|---|
| **DeepPrivacy** | ISVC 2019 | [hukkelas/DeepPrivacy](https://github.com/hukkelas/DeepPrivacy) | `deidentify_batch.py` | dedicated conda env (torch) | 1.3 GB |
| **CLEANIR** | Applied Sciences 2020 | [chodurkhyun/cleanir](https://github.com/chodurkhyun/cleanir) | `deidentify_batch.py` | TensorFlow (per official repo) | 0.1 GB |
| **IPFA** | ACM MM 2021 | [RuoyuChen10/Facial_Attributes_Obfuscation](https://github.com/RuoyuChen10/Facial_Attributes_Obfuscation) | `deidentify_batch.py`, `deidentify_batch_256.py` (deployed protocol, 256²), `deidentify_batch_optiona.py` | torch | 3.1 GB |
| **RiDDLE** | CVPRW 2022 | [DongzeLi-CASIA/RiDDLE](https://github.com/DongzeLi-CASIA/RiDDLE) | `deidentify_batch.py` | torch + official deps | 4.2 GB |
| **AMT-GAN** | CVPR 2022 | [cgcl-codes/amt-gan](https://github.com/cgcl-codes/amt-gan) | `deidentify_batch.py` | torch | 0.7 GB |
| **FALCO** | CVPR 2023 | [chi0tzp/FALCO](https://github.com/chi0tzp/FALCO) (branch `cvpr23`) | `deidentify_batch.py` | torch | 3.9 GB |
| **CPP-DeID** | Image & Vision Computing 2023 | [CPP-DeID (StyleCLIP fork used)](https://github.com/orpatashnik/StyleCLIP) | `deidentify_batch.py` | torch | 5.2 GB (StyleGAN2-FFHQ + e4e inv.) |
| **LDFA** | CVPRW 2023 | [KIT-MRT/latent_diffusion_face_anonymization](https://github.com/KIT-MRT/latent_diffusion_face_anonymization) | `deidentify_batch.py` | torch + diffusers | 2.1 GB (SD2-inpaint) |
| **DeepPrivacy2** | WACV 2023 | [hukkelas/deep_privacy2](https://github.com/hukkelas/deep_privacy2) | `deidentify_batch.py` | torch | 0.5 GB |
| **G2Face** | IEEE TIFS 2024 | [harxis/g2face](https://github.com/harxis/g2face) | `deidentify_batch.py` | torch | 2.9 GB |
| **GANonymization** | ACM TOMM 2024 | [hcmlab/GANonymization](https://github.com/hcmlab/GANonymization) | `deidentify_batch.py` | torch + mediapipe | 0.7 GB |
| **FADM** | ACMR@ECCV 2024 | [fzi-forschungszentrum-informatik/fadm](https://github.com/fzi-forschungszentrum-informatik/fadm) (we drive the official `anonymize.py` with 256-crop flags) | (no standalone script) | torch + diffusers | 6.0 GB |
| **DiffPrivate** | PoPETs 2025 | [minha12/DiffPrivate](https://github.com/minha12/DiffPrivate) | `deidentify_batch.py` | torch + official deps | 27.0 GB (SD + DiffAE) |
| **FAMS** | WACV 2025 | [hanweikung/face_anon_simple](https://github.com/hanweikung/face_anon_simple) | `deidentify_batch.py` | torch + diffusers/transformers | 1.0 GB |
| **AIDPro** | IEEE TIFS 2025 | [daizigege/AIDPro](https://github.com/daizigege/AIDPro) | `deidentify_batch.py` | torch | 0.7 GB |
| **NullFace** | ICFAI 2026 | [hanweikung/nullface](https://github.com/hanweikung/nullface) | `deidentify_batch.py`, `anonymize_face.py`, `utils/face_embedding.py`, `utils/sample_vector.py`, `fix_nullface_failures.py` (gap-filler for undetected faces, e.g. profiles) | py-torch 2.11 + cu128, diffusers | SD 1.5 (official release) + InsightFace models (~0.6 GB) |
| **RP** (Reverse Personalization) | WACV 2026 | [hanweikung/reverse-personalization](https://github.com/hanweikung/reverse-personalization) | `deidentify_batch_rp.py`, `anonymize_local.py` (local SDXL pipeline loader), `resize_to_256.py` | py3.12, torch 2.11.0+cu128, diffusers 0.30.0 | 13.4 GB local (pruned fp16 SDXL, canonical filenames) |
| **AnonNET** | ICCVW 2025 (CV4BIOM) | [anilegin/AnonNET](https://github.com/anilegin/AnonNET) | `deidentify_batch_anonnet.py` (local weight redirection + BiSeNet hub patch + DeepFace prompt protocol); **apply `patches/0001-strength-retry-list-arithmetic.patch` to the official checkout** (fixes a crash in the self-verification retry whenever a list `--strength` is used) | py3.9, torch 2.7.1+cu128, **TensorFlow 2.15** (DeepFace stack), deepface 0.0.93 | 28.8 GB (SD1.5 Realistic-Vision, 3 ControlNets, SV2 VAE, annotators, DeepFace weights) |
| **iFADIT** | Pattern Recognition 2025 | [lixionga/ProFace](https://github.com/lixionga/ProFace) (`FacePrivacy/iFADIT`) + [official checkpoint folder (Google Drive)](https://drive.google.com/drive/folders/1XIE9_3LXKiIJNdtroyZvwCKaKnu-x12O) | `deidentify_batch_ifadit.py` (reconstructs the released test.py protocol; mmcv ops stubbed — never used at inference) | py3.10, torch 2.7.1+cu128, `freia`, scipy | 2.6 GB (official Google-Drive weight folder) |
| **PRO-Face** | ACM MM 2022 | [lixionga/ProFace](https://github.com/lixionga/ProFace) (`FacePrivacy/PRO-Face`) | `deidentify_batch_proface.py` + `embedder.py` (restoration net, vendored) with **SimSwap face-swap obfuscation** (`--obf simswap`) + IResNet50-trained restoration | py3.10, torch 2.11.0+cu128 | restore checkpoint in-repo; SimSwap generator + ArcFace weights (official SimSwap release, ~465 MB) must be placed under `SimSwap/`; output 112/224-native, upscaled to 256 per our protocol |

Notes:

- **Verification state:** every script above produced the de-identified image sets used
  in the paper's tables and figures (16 × 4 datasets at setup time; RP / AnonNET /
  iFADIT / PRO-Face added 2026-08). Failure logs from the paper runs accompanied the
  result CSVs where applicable.
- **Weights are never committed** to this repository (see the `.gitignore` in
  `deid-toolkit` and per-method `weights_manifest.txt`). Total footprint of the 20
  approaches ≈ 95 GB locally on the research machine.
- The working collection with full official repos, bats, and build records lives
  separately (`deid-toolkit_baselines`); this folder is the publishable subset.
- PRO-Face *S* (Yuan et al., IEEE TCSVT 2024, doi:10.1109/TCSVT.2023.3344809) was evaluated
  during the setup and rejected: its official weights are Baidu-only and the alternate mirrors
  are dead. PRO-Face *C* (classification-side only) is out of scope. The MM'22 framework above,
  with SimSwap obfuscation, is the one reported in the paper.
