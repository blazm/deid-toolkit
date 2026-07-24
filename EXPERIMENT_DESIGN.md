# Experiment Design: Embedding Space Analysis for Face De-identification

## Positioning

This work sits at the intersection of **model interpretability**, **failure mode analysis**, and **post-hoc explanation** in facial biometrics. Rather than treating identity verification models as black boxes that produce a match/no-match decision, we interrogate the internal representation space to understand *how* de-identification techniques perturb those representations.

| Framing dimension | What it means for this work |
|-------------------|---------------------------|
| **Model interpretability** | Projecting 512-d SWINFace embeddings to 2D (UMAP/PCA/t-SNE) makes high-dimensional representation manipulation human-understandable. Displacement vectors are an interpretable summary of "what changed" in the model's internal state. |
| **Failure mode analysis** | We characterize *how* a technique fails: does it scatter embeddings uniformly (controlled obfuscation), or collapse them toward a degenerate region (representation collapse)? These are distinct failure modes with different implications for privacy and utility. |
| **Post-hoc explanation** | Without modifying any model or technique, we retroactively explain their behavior by measuring embedding displacements, identity dispersion changes, and perturbation patterns. This is post-hoc because it requires no access to the technique's internals — only its input/output image pairs and a frozen verification model. |

This is distinct from traditional XAI (Grad-CAM, saliency maps) which explains *individual decisions*. We explain *systematic behavioral patterns* across thousands of images — an audit-level analysis rather than a per-sample explanation.

## 1. Motivation

Current face de-identification literature evaluates techniques along two orthogonal axes: **privacy** (how well identity verification fails, measured by AUC/EER) and **utility** (how well demographic attributes are preserved, measured by classification accuracy). These aggregate metrics answer "does it work?" but not **"how does it work?"**

Two techniques with identical AUC=0.52 can exhibit radically different behavior in the underlying representation space:
- **Technique A** scatters embeddings uniformly in all directions (isotropic perturbation) — preserves diversity, makes identification unreliable without destroying all semantic structure.
- **Technique B** collapses all embeddings toward a single region (representation collapse) — everyone looks alike, destroying both identity and demographic information indiscriminately.

Neither behavior is captured by AUC alone. Both achieve chance-level verification, but Technique B offers zero utility while Technique A may preserve attributes. Understanding *how* a technique manipulates representations is essential for selecting the right tool for a given application.

## 2. Research Questions

| ID | Question | Answered by |
|----|----------|-------------|
| RQ1 | How does each de-identification technique perturb individual face embeddings in identity representation space? | Displacement Field Analysis (§3) |
| RQ2 | Do any techniques cause identity collapse (merging distinct identities into similar representations)? | Identity Collapse Detection (§4) |
| RQ3 | Can we classify techniques by their *manipulation pattern* (scatter, collapse, rotation), independent of dataset? | Multi-Technique Comparison Overlay (§5) |
| RQ4 | Does the behavior of de-identification techniques generalize across dataset types (studio vs. in-the-wild)? | Cross-Dataset Consistency Analysis (§6) |

## 3. Displacement Field Analysis (Post-Hoc Explanation)

### What It Shows
For every image in a dataset, we compute the vector from the original face embedding to its de-identified counterpart in a shared 2D projection. The direction and magnitude of each arrow reveals how the technique *moved* that specific identity representation.

### Why Post-Hoc?
No modifications to the de-identification technique or verification model are required. We take as input: (1) original images, (2) de-identified images, and (3) a frozen, pre-trained verification model (SWINFace). The displacement field is computed entirely after the fact — hence "post-hoc."

### Methodology
1. Extract 512-d embeddings (SWINFace) for original and de-identified images
2. Jointly project all embeddings to 2D via UMAP (n_neighbors=15, min_dist=0.1, seed=42) — identical input vectors produce identical projections via deduplication
3. Compute per-image displacement: `v_i = proj(deid_i) - proj(orig_i)`
4. Color arrows by **embedding-space Euclidean distance** (true perturbation magnitude; UMAP is only used for direction visualization)
5. Overlay cosine similarity between `emb(orig_i)` and `emb(deid_i)` as a per-image statistic

### Interpretation
- **Isotropic scattering**: Arrows point in all directions with similar magnitudes. Indicates uniform perturbation — no systematic bias. Typical of blur, pixelization.
- **Directed rotation**: Arrows cluster around a common direction. Suggests the technique applies a consistent transformation (e.g., GAN generator pushes all faces toward a learned "latent center").
- **Center collapse**: Short arrows from peripheral identities converging on a dense central region. Indicates representation collapse — undesirable for utility preservation.
- **Selective perturbation**: Some images show large displacement while others are barely moved. Reveals technique bias (e.g., works better on frontal faces than profile views).

### Manuscript Figure Template
> "Figure 1: Per-image displacement field showing how [technique] perturbs identity embeddings in SWINFace representation space. Arrows originate from original embedding positions (projected via UMAP), colored by true embedding-space Euclidean distance between original and de-identified representations. Darker red indicates stronger perturbation. Dashed rings group images by identity; within-ring consistency reveals per-subject technique behavior."

## 4. Identity Collapse Detection (Failure Mode Analysis)

### What It Shows
For each person (identity) in the dataset, we measure how much their intra-identity diversity is preserved after de-identification. A collapse ratio < 1 means the technique made that person's images more similar to each other in embedding space; ratio > 1 means they were pushed further apart.

### Why Failure Mode Analysis?
Representation collapse is a known failure mode in generative models (GANs, VAEs) but has never been systematically measured in face de-identification. Our collapse ratio metric provides the first quantitative characterization of this failure mode: it distinguishes techniques that *gracefully obscure* identity from those that *degenerately collapse* all representations into a single point.

### Methodology
1. Group images by identity label (from dataset annotation CSVs)
2. Compute mean pairwise cosine distance within each identity group for original embeddings: `d_orig(id) = mean(cos_dist(orig_i, orig_j))` for all i,j belonging to id
3. Compute the same for de-identified embeddings: `d_deid(id)`
4. Collapse ratio: `CR(id) = d_deid(id) / d_orig(id)`
5. Aggregate: mean/median collapse ratio across all identities; histogram of collapse ratios

### Interpretation
- **CR << 1 (collapsed)**: The technique eliminated intra-identity variation. All images of the same person become indistinguishable even before verification — both privacy and utility are lost.
- **CR ~ 1 (neutral)**: Intra-identity diversity preserved, but absolute identity position changed. Ideal behavior for utility-preserving de-identification.
- **CR >> 1 (amplified)**: The technique increased variation within an identity group. May indicate unstable perturbation or overfitting on certain faces.

### Manuscript Figure Template
> "Figure 2: Per-identity collapse ratio distribution for [technique] on [dataset]. Each bar represents one identity, colored by severity (red < 0.5: collapsed, yellow 0.5–1.0: partial, green > 1.0: preserved). The dashed line at CR=1.0 marks the preservation threshold. Mean collapse ratio of [X.XX] indicates that [technique] [preserves/reduces/increases] intra-identity diversity overall."

## 5. Multi-Technique Comparison Overlay (Model Interpretability)

### What It Shows
All techniques' displacement vectors overlaid on a single joint projection, enabling direct visual comparison: "blur scatters uniformly, while pixelization shows moderate directed movement toward [region]."

### Why Model Interpretability?
By projecting all embeddings (originals + N techniques × de-identified) into a single shared space, we make the internal representation transformations of multiple techniques directly comparable. This is interpretable because: (a) the 2D projection preserves local/global structure (UMAP), and (b) displacement vectors from shared origin points reveal each technique's unique "fingerprint" in how it manipulates representations.

### Methodology
1. Extract original embeddings + de-identified embeddings for each technique
2. Joint UMAP projection of ALL embedding sets (originals once + N techniques) in a single call — deduplication ensures shared origin points
3. Each technique color-coded; arrows drawn from identical gray origin points
4. Subsampled to 80 random images for readability at large N

### Interpretation
- Techniques whose arrows fan out equally in all directions are **isotropic** (mechanically simple, e.g., blur).
- Techniques whose arrows consistently point toward the same region are **anisotropic** (learned transformation with a bias direction, e.g., GAN-based methods).
- Overlapping arrow patterns indicate techniques with similar underlying mechanisms.

### Manuscript Figure Template
> "Figure 3: Multi-technique comparison in a joint UMAP projection. Gray dots mark original embedding positions; colored arrows show per-image displacement vectors for each technique. Each technique exhibits a distinct manipulation pattern, revealing that [key finding about behavioral classes]."

## 6. Cross-Dataset Consistency Analysis (Studio vs. In-the-Wild)

### What It Shows
Whether a technique's embedding manipulation pattern is consistent across different data acquisition conditions, or whether it behaves differently on controlled studio portraits vs. unconstrained in-the-wild photographs.

### Dataset Classification

| Type | Datasets | Characteristics | Embedding space properties |
|------|----------|-----------------|----------------------------|
| **Studio-captured** | mug-still | Controlled lighting, uniform background, constrained pose range, professional capture | Narrow embedding distribution; tight inter-identity clusters; limited representation of occlusion/variation |
| **In-the-wild** | celeba-test_aligned | Uncontrolled lighting, diverse backgrounds, natural poses, real-world conditions (glasses, hats, expressions) | Broad embedding distribution; well-separated identity clusters; rich attribute diversity embedded in representations |

### Why It Matters for De-identification
A technique that works uniformly on studio data may fail under in-the-wild conditions:
- **Pose-dependent perturbation**: Blur is isotropic in pixel space but may preserve identity more effectively at extreme poses where facial features are already partially occluded.
- **Lighting artifacts**: Pixelization interacts with shadows and highlights, potentially creating exploitable patterns visible only under certain lighting conditions.
- **Attribute correlation**: In-the-wild images embed demographic attributes (glasses, age markers) alongside identity; techniques that perturb identity may disproportionately affect correlated attributes.

### Methodology
1. Run displacement analysis independently on each dataset
2. Compare mean displacement distributions (Kolmogorov-Smirnov test)
3. Measure correlation between per-image displacements in studio vs. in-the-wild for subjects appearing in both datasets (if overlap exists)
4. For techniques exhibiting directed patterns, compare the mean displacement vector direction across datasets

### Interpretation
- **Consistent behavior**: Same manipulation pattern across datasets → technique is robust to acquisition conditions.
- **Dataset-dependent behavior**: Different patterns on studio vs. in-the-wild → technique effectiveness depends on image characteristics. Requires careful deployment guidance.

### Manuscript Figure Template
> "Figure 4: Cross-dataset displacement comparison showing that [technique] [maintains/diverges in its] embedding manipulation pattern between studio-captured (mug-still) and in-the-wild (CelebA-test) datasets. Mean displacement vectors [overlap/diverge] by [X.X]°, indicating [robustness/condition-dependence] of the technique's perturbation mechanism."

## 7. Experimental Protocol

### Datasets Under Analysis

| Dataset | Type | Images | Identities | Avg Images/ID |
|---------|------|--------|------------|---------------|
| mug-still | Studio-captured | ~671 (with labels) | 40 | ~17 |
| celeba-test_aligned | In-the-wild | ~2824 | 579 | ~3.3 |

### Techniques Under Analysis

| Technique | Category | Mechanism | Expected pattern |
|-----------|----------|-----------|-----------------|
| blur | Pixel-space | Gaussian filter (σ=15) | Isotropic scattering |
| pixelize | Pixel-space | Block averaging (32×32) | Moderate isotropic scattering |

### Embedding Model

| Model | Dimension | Choice rationale |
|-------|-----------|-----------------|
| SWINFace | 512-d | Fast inference with built-in caching; intermediate capacity avoids over/under-fitting |

### Projection Methods

| Method | Parameters | Use case |
|--------|------------|----------|
| UMAP | n_neighbors=15, min_dist=0.1 | Default — preserves local and global structure |
| PCA | — | Interpretable axes (variance-explained %); linear baseline |
| t-SNE | perplexity=30 | Local neighborhood emphasis; validation of UMAP findings |

### Computed Metrics Per Image

| Metric | Space | Description |
|--------|-------|-------------|
| Euclidean displacement | 512-d embedding | True perturbation magnitude: `||emb_deid - emb_orig||` |
| Cosine similarity | 512-d embedding | Directional preservation: `(emb_deid · emb_orig) / (||emb_deid|| × ||emb_orig||)` |
| UMAP arrow direction | 2D projection | Visual representation of perturbation pattern |

### Computed Metrics Per Identity

| Metric | Description |
|--------|-------------|
| Collapse ratio | Intra-identity dispersion after / before de-identification |
| Mean displacement | Average perturbation magnitude across all images of this identity |

## 8. Hypotheses

| ID | Hypothesis | Validation |
|---------------|------------|
| H1 | Pixel-space techniques (blur, pixelize) produce isotropic displacement patterns with no systematic direction bias | RQ1 + RQ3: Displacement vectors uniformly distributed across all directions |
| H2 | De-identification techniques preserve intra-identity diversity (CR ~ 1.0), meaning they perturb identity position without collapsing individual variation | RQ2: Collapse ratio distribution centered near 1.0 |
| H3 | In-the-wild datasets exhibit greater mean displacement magnitudes due to richer feature content available for perturbation | RQ4: celeba-test_aligned shows higher mean displacement than mug-still |
| H4 | Techniques with lower verification AUC do not necessarily cause more identity collapse | RQ1 + RQ2 cross-reference: No strong correlation between AUC and collapse ratio |
