"""RP (Reverse Personalization) anonymization with ALL models loaded from local
pretrained_models/ + insightface_root/ folders (no HF/OS cache references).

Code identical to the upstream `anonymize_faces_in_image.py` except:
- sd_model_path, IP-Adapter image encoder, IP-Adapter-FaceID weights and the
  InsightFace root all resolve to local directories under this folder.
"""

import os
import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from sdxl.leditspp.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline as StableDiffusionPipelineXL_LEDITS,
)
from sdxl.leditspp.scheduling_dpmsolver_multistep_inject import (
    DPMSolverMultistepSchedulerInject,
)

_HERE = os.path.dirname(os.path.abspath(__file__))

LOCAL_SDXL = os.path.join(_HERE, "pretrained_models", "stable-diffusion-xl-base-1.0")
LOCAL_IP_CLIP = os.path.join(_HERE, "pretrained_models", "IP-Adapter")  # subfolder: models/image_encoder
LOCAL_IP_FACEID = os.path.join(_HERE, "pretrained_models", "IP-Adapter-FaceID")  # ip-adapter-faceid_sdxl.bin
LOCAL_INSIGHTFACE = os.path.join(_HERE, "insightface_root")

# NOTE: imported lazily to mirror upstream; used only when enable_face_detection=True
# (our batch runs always use False — inputs are pre-aligned faces)
from utils.extractor import extract_faces
from utils.face_embedding import FaceEmbeddingExtractor
from utils.merger import paste_foreground_onto_background


class FaceDetectionError(ValueError):
    pass


def anonymize_faces_in_image(
    input_image,
    attribute_prompt=None,
    device_num=0,
    skip=0.7,
    id_emb_scale=1.0,
    guidance_scale=-10.0,
    num_inversion_steps=100,
    face_image_size=1024,
    det_thresh=0.1,
    ip_adapter_scale=1.0,
    det_size=640,
    seed=0,
    enable_face_detection=False,
):
    dtype = torch.float16
    device = f"cuda:{device_num}"

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        LOCAL_IP_CLIP, subfolder="models/image_encoder", torch_dtype=dtype
    )

    anon_image = image = Image.open(input_image)

    if enable_face_detection:
        import face_alignment
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, face_detector="sfd"
        )
        face_images, image_to_face_matrices = extract_faces(fa, image, face_image_size)
    else:
        face_images = [image]
        image_to_face_matrices = [None]

    pipe = StableDiffusionPipelineXL_LEDITS.from_pretrained(
        LOCAL_SDXL,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
        LOCAL_SDXL,
        subfolder="scheduler",
        algorithm_type="sde-dpmsolver++",
        solver_order=2,
    )
    pipe = pipe.to(device)

    pipe.load_ip_adapter(
        LOCAL_IP_FACEID,
        subfolder=None,
        weight_name="ip-adapter-faceid_sdxl.bin",
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    extractor = FaceEmbeddingExtractor(
        ctx_id=0,
        det_thresh=det_thresh,
        det_size=(det_size, det_size),
        model_path=LOCAL_INSIGHTFACE,
    )

    for face_image, image_to_face_mat in zip(face_images, image_to_face_matrices):
        try:
            id_embs_inv, id_embs = extractor.get_face_embeddings(
                image_path=face_image,
                seed=seed,
                scale_factor=id_emb_scale,
                dtype=dtype,
                device=device,
            )
        except ValueError as e:
            raise FaceDetectionError(str(e))
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

            _ = pipe.invert(
                image=face_image,
                num_inversion_steps=num_inversion_steps,
                skip=skip,
                source_guidance_scale=guidance_scale,
                ip_adapter_image_embeds=[id_embs_inv],
                generator=generator,
            )

            anon_face_image = pipe(
                prompt="",
                negative_prompt=attribute_prompt,
                ip_adapter_image_embeds=[id_embs],
                num_images_per_prompt=1,
                generator=generator,
                guidance_scale=guidance_scale,
                timesteps=pipe.scheduler.timesteps,
                latents=pipe.init_latents,
                num_inference_steps=num_inversion_steps,
            ).images[0]

            if enable_face_detection:
                anon_image = paste_foreground_onto_background(
                    anon_face_image, anon_image, image_to_face_mat
                )
            else:
                anon_image = anon_face_image

    return anon_image


# ─────────────────────────── module-level pipeline state ───────────────────────────
# The upstream API reloads SDXL + IP-Adapter + InsightFace on every call, which is
# impractical at dataset scale (6.9 GB re-read + weight re-fusion per image). We load
# the heavy parts once and only build the per-call state below.

_PIPE = None
_EXTRACTOR = None


def init_pipeline(device_num=0, det_thresh=0.1, det_size=640):
    global _PIPE, _EXTRACTOR
    dtype = torch.float16
    device = f"cuda:{device_num}"
    if _PIPE is not None:
        _PIPE.scheduler.timesteps = None  # reset stale timesteps between datasets
        return

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        LOCAL_IP_CLIP, subfolder="models/image_encoder", torch_dtype=dtype
    )
    pipe = StableDiffusionPipelineXL_LEDITS.from_pretrained(
        LOCAL_SDXL, image_encoder=image_encoder, torch_dtype=dtype
    )
    pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
        LOCAL_SDXL,
        subfolder="scheduler",
        algorithm_type="sde-dpmsolver++",
        solver_order=2,
    )
    pipe = pipe.to(device)
    pipe.load_ip_adapter(
        LOCAL_IP_FACEID,
        subfolder=None,
        weight_name="ip-adapter-faceid_sdxl.bin",
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(1.0)
    _PIPE = pipe
    _EXTRACTOR = FaceEmbeddingExtractor(
        ctx_id=0,
        det_thresh=det_thresh,
        det_size=(det_size, det_size),
        model_path=LOCAL_INSIGHTFACE,
    )
    print("RP pipeline initialized (SDXL + IP-Adapter-FaceID + InsightFace, all local).")


def anonymize_one(
    image_path,
    skip=0.7,
    id_emb_scale=1.0,
    guidance_scale=-10.0,
    num_inversion_steps=100,
    ip_adapter_scale=1.0,
    seed=0,
    attribute_prompt=None,
):
    """Fast-path anonymization using the module-level pipeline.

    Input must be a pre-aligned face crop (enable_face_detection=False protocol).
    Raises FaceDetectionError when InsightFace finds no face (caller decides policy).
    """
    assert _PIPE is not None, "call init_pipeline() first"
    dtype = _PIPE.dtype
    image = Image.open(image_path)

    id_embs_inv, id_embs = _EXTRACTOR.get_face_embeddings(
        image_path=image,
        seed=seed,
        scale_factor=id_emb_scale,
        dtype=dtype,
        device="cuda:0",
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    _PIPE.invert(
        image=image,
        num_inversion_steps=num_inversion_steps,
        skip=skip,
        source_guidance_scale=guidance_scale,
        ip_adapter_image_embeds=[id_embs_inv],
        generator=generator,
    )
    anon = _PIPE(
        prompt="",
        negative_prompt=attribute_prompt,
        ip_adapter_image_embeds=[id_embs],
        num_images_per_prompt=1,
        generator=generator,
        guidance_scale=guidance_scale,
        timesteps=_PIPE.scheduler.timesteps,
        latents=_PIPE.init_latents,
        num_inference_steps=num_inversion_steps,
    ).images[0]
    return anon
