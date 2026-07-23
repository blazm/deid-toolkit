"""Toolkit documentation — datasets, techniques, CLI commands.

Accessible only after login.
"""
from __future__ import annotations

import streamlit as st


CUSTOM_CSS = """
<style>
.doc-section {
    font-size: 1.3rem;
    font-weight: 600;
    color: #3b5998;
    border-bottom: 2px solid #e8e8e8;
    padding-bottom: 0.4rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.doc-card {
    background: white;
    border-left: 4px solid #3b5998;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.doc-subsection {
    font-size: 1.1rem;
    font-weight: 500;
    color: #3b5998;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
}
.doc-tag {
    display: inline-block;
    background: #f0f4f8;
    border: 1px solid #d0d5dd;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #3b5998;
    margin: 2px;
}
</style>
"""


def render() -> None:
    st.set_page_config(page_title="Face De-Identification Toolkit — Docs", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    col_title, col_back = st.columns([6, 1])
    with col_title:
        st.title("Face De-Identification Toolkit — Documentation")
    with col_back:
        if st.button("← Home", key="docs_back_home"):
            st.session_state.current_page = "home"
            st.rerun()
    st.markdown("---")

    # ================================================================== #
    # OVERVIEW
    # ================================================================== #
    st.markdown('<div class="doc-section">Overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="doc-card">
            This toolkit provides a unified framework for running and evaluating
            privacy-preserving techniques in facial biometrics. It supports a growing
            collection of deep learning–based de-identification methods, evaluation
            metrics (verification, identification, data utility), and an interactive
            web interface for exploring before-and-after results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================================================================== #
    # RESEARCH CONTRIBUTIONS
    # ================================================================== #
    st.markdown('<div class="doc-section">Research Contributions</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="doc-card"><strong>1. Unified DEID Pipeline</strong><br><br>'
            "A reproducible pipeline that applies privacy-preserving techniques "
            "across multiple benchmark datasets, producing standardized results "
            "for fair comparison.</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="doc-card"><strong>2. Comprehensive Evaluation Suite</strong><br><br>'
            "Identity verification (AdaFace, SWINface, VGG-Face), image quality metrics "
            "(FID, LPIPS, SSIM, MSE), and data utility evaluation (gender, emotion, "
            "expression classification) under de-identification.</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="doc-card"><strong>3. Interactive Analysis Tool</strong><br><br>'
            "A Streamlit-based browser for visualizing de-identification effects "
            "including ROC/PR curves, score distributions, confusion matrices, "
            "embedding clustering, and before/after image galleries.</div>",
            unsafe_allow_html=True,
        )

    # ================================================================== #
    # DEID TECHNIQUES
    # ================================================================== #
    st.markdown('<div class="doc-section">DEID Techniques</div>', unsafe_allow_html=True)

    techniques = [
        ("DeepPrivacy2", "GAN-based face anonymization with deep generative models."),
        ("AMT-GAN", "Adversarial Mutilation and Transformation GAN for face de-identification."),
        ("GDDPG", "Generative Deep De-identification via deterministic policy gradient."),
        ("MaskDDPG", "Face masking technique using deep policy gradients."),
        ("DeepFake", "Face swapping for identity preservation."),
        ("AdaFace", "Adaptive margin face de-identification."),
        ("Blur", "Gaussian/median blurring as a baseline."),
        ("Pixelize", "Patch-based pixelation."),
        ("CIAGAN", "Conditional Invertible GAN for controllable de-identification."),
        ("CleanIR", "Clean identity removal with retention of non-face attributes."),
        ("CPP-DEID", "Content-preserving personalization de-identification."),
        ("KSamNet", "Knowledge-aware similarity masking network."),
    ]

    st.markdown('<div class="doc-subsection">Techniques</div>', unsafe_allow_html=True)
    tech_html = ""
    for name, desc in techniques:
        tech_html += f"""
        <div style="background: #f8f9fa; padding: 0.6rem 0.8rem; border-radius: 4px; border-left: 3px solid #3b5998;">
            <div class="doc-tag">{name}</div>
            <div style="font-size: 0.85rem; color: #666; margin-top: 0.3rem;">{desc}</div>
        </div>"""
    st.components.v1.html(
        f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.8rem; width: 100%; margin: 0; padding: 0;">
        {tech_html}
        </div>
        <style>body {{ margin: 0; padding: 0; }}</style>
        """,
        height=280,
    )

    # ================================================================== #
    # SUPPORTED DATASETS
    # ================================================================== #
    st.markdown('<div class="doc-section">Supported Datasets</div>', unsafe_allow_html=True)

    datasets = [
        ("AR Face", "Advanced AR Face Dataset"),
        ("CelebA", "Large-scale Celebrity Attributes"),
        ("CK+", "Colored Extended Cohn-Kanade"),
        ("ColorFERET", "Colored FERET faces"),
        ("FRI", "Face Recognition Interior"),
        ("KDEF", "Swedish emotional faces"),
        ("LFW", "Labeled Faces in the Wild"),
        ("MORPH II", "Aging face database"),
        ("MUCT", "Multi-face, Multi-view, Multi-expression"),
        ("RAF-DB", "Real-world Affective Facial expressions"),
        ("XM2VTS", "Multi-modal face database"),
        ("UtkFace", "Age and gender attributes"),
    ]

    st.markdown('<div class="doc-subsection">Dataset Names</div>', unsafe_allow_html=True)
    ds_html = ""
    for name, desc in datasets:
        ds_html += f"""
        <div style="background: #f8f9fa; padding: 0.6rem 0.8rem; border-radius: 4px; border-left: 3px solid #3b5998;">
            <div class="doc-tag">{name}</div>
            <div style="font-size: 0.85rem; color: #666; margin-top: 0.3rem;">{desc}</div>
        </div>"""
    st.components.v1.html(
        f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.8rem; width: 100%; margin: 0; padding: 0;">
        {ds_html}
        </div>
        <style>body {{ margin: 0; padding: 0; }}</style>
        """,
        height=300,
    )

    # ================================================================== #
    # QUICK START CLI
    # ================================================================== #
    st.markdown('<div class="doc-section">Quick Start — CLI</div>', unsafe_allow_html=True)

    with st.expander("View CLI commands", expanded=False):
        st.markdown("""
```bash
# List available datasets, techniques, and evaluation metrics
deid list datasets
deid list techniques
deid list evaluation

# Select items to evaluate
deid select datasets arface ck+_fix
deid select techniques deepprivacy2 blur
deid select evaluation ssim lpips

# Run the full pipeline
deid run all

# Explore results interactively
deid explore
```
        """,)

    # ================================================================== #
    # EVALUATION METRICS
    # ================================================================== #
    st.markdown('<div class="doc-section">Evaluation Metrics</div>', unsafe_allow_html=True)

    st.markdown('<div class="doc-subsection">Identity Verification</div>', unsafe_allow_html=True)
    st.markdown("AdaFace, SWINface, VGG-Face")

    st.markdown('<div class="doc-subsection">Image Quality</div>', unsafe_allow_html=True)
    st.markdown("FID, LPIPS, SSIM, MSE")

    st.markdown('<div class="doc-subsection">Data Utility</div>', unsafe_allow_html=True)
    st.markdown("Gender classification, Emotion classification, Expression classification")

    # ── TODO items ────────────────────────────────────────────────────────
    st.markdown('<div class="doc-subsection">TODO / Upcoming</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: #fff3cd; border-left: 3px solid #ffc107; padding: 0.6rem 0.8rem; border-radius: 4px; margin-bottom: 0.5rem;">
            <strong>TODO — Pose estimation before vs. after de-identification</strong><br>
            <span style="font-size: 0.85rem; color: #666;">
            Estimate head pose (yaw, pitch, roll) on original and de-identified images
            to verify that pose data utility is preserved. This is critical for downstream
            tasks like pose-invariant recognition and surveillance analytics.
            </span>
        </div>
        <div style="background: #fff3cd; border-left: 3px solid #ffc107; padding: 0.6rem 0.8rem; border-radius: 4px; margin-bottom: 0.5rem;">
            <strong>TODO — Gaze estimation before vs. after de-identification</strong><br>
            <span style="font-size: 0.85rem; color: #666;">
            Estimate gaze direction on original and de-identified images to verify that
            gaze data utility is preserved. Gaze is a sensitive attribute that can
            leak intent; verifying its preservation helps assess whether de-identification
            inadvertently compromises or leaks behavioral signals.
            </span>
        </div>
        <div style="background: #fff3cd; border-left: 3px solid #ffc107; padding: 0.6rem 0.8rem; border-radius: 4px; margin-bottom: 0.5rem;">
            <strong>TODO — FIQ (Face Image Quality) before vs. after de-identification</strong><br>
            <span style="font-size: 0.85rem; color: #666;">
            Compute FIQ scores on original and de-identified images to verify that
            image quality remains above usable thresholds. FIQ evaluates multiple quality
            dimensions (illumination, contrast, resolution, facial region quality) and
            provides an overall score. This helps determine whether de-identified faces
            still meet quality requirements for downstream face recognition systems.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================================================================== #
    # CROWDSOURCED VERIFICATION
    # ================================================================== #
    st.markdown('<div class="doc-section">Crowdsourced Verification</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="doc-card">
            <p>
            The DeID Toolkit includes a <strong>Crowdsourced Verification Survey</strong> — a public-facing
            questionnaire with two phases where volunteers evaluate de-identification effectiveness.
            </p>
            <p>
            <strong>Phase 1 — Validation:</strong> Aligned (original) image pairs test whether
            respondents can reliably distinguish same-person from different-person pairs.
            Accuracy below 80% flags unreliable responses.<br><br>
            <strong>Phase 2 — De-identification:</strong> De-identified image pairs are evaluated
            for identity protection while measuring data utility preservation.
            </p>
            <p><strong>Sharable Link:</strong></p>
            <code style="background: #f0f4f8; padding: 0.3rem 0.5rem; border-radius: 3px;">
            http://193.2.76.178:8501/?page=survey
            </code>
            <p>
            Results contribute to a new evaluation metric: <strong>Human Verification Accuracy</strong>,
            which measures how well humans can identify identities in de-identified images.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
