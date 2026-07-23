"""Landing page — research portfolio for the DeID toolkit project."""
from __future__ import annotations

import streamlit as st

from deid.explore.assets import logo_path

# --- Custom CSS matching FRI personal page aesthetic ---
CUSTOM_CSS = """
<style>
/* Compact sections */
.deid-section {
    font-size: 1.2rem;
    font-weight: 600;
    color: #3b5998;
    border-bottom: 1px solid #e8e8e8;
    padding-bottom: 0.3rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

/* Compact hero */
.hero-section {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    padding: 1.5rem 1.5rem;
    border-radius: 6px;
    margin-bottom: 1rem;
    text-align: center;
}
.hero-section h2 {
    font-size: 1.2rem;
    margin: 0;
    color: #3b5998;
}
.hero-section p {
    font-size: 0.9rem;
    margin: 0.25rem 0;
}
.hero-section a {
    color: #3b5998;
    font-weight: 600;
}

/* Navigation buttons near top */
.nav-buttons {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.nav-buttons button {
    flex: 0 0 auto;
}

/* Publications in two-column layout */
.pub-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 0.4rem 1rem;
}

/* Compact publication items */
.pub-item {
    padding: 0.4rem 0;
    border-bottom: 1px solid #eee;
    margin-bottom: 0 !important;
}
.pub-year {
    display: inline-block;
    background: #3b5998;
    color: white;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.3rem;
}
.pub-title {
    font-weight: 600;
    font-size: 0.95rem;
}
.pub-venue {
    color: #666;
    font-size: 0.85rem;
}
.pub-citations {
    font-size: 0.8rem;
    color: #888;
    margin-left: 0.3rem;
}

/* Compact cards */
.deid-card {
    background: white;
    border-left: 3px solid #3b5998;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    border-radius: 3px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    font-size: 0.95rem;
}

/* Compact projects */
.project-badges {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.project-badge {
    display: inline-block;
    background: #f0f4f8;
    border: 1px solid #d0d5dd;
    padding: 0.4rem 0.8rem;
    border-radius: 5px;
    font-size: 0.9rem;
}
.project-badge strong {
    color: #3b5998;
    display: block;
    font-size: 0.95rem;
}

/* Compact timeline */
.timeline-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 0.3rem;
    font-size: 0.95rem;
}
.timeline-dot {
    width: 10px;
    height: 10px;
    background: #3b5998;
    border-radius: 50%;
    margin-right: 0.8rem;
    margin-top: 0.3rem;
    flex-shrink: 0;
}
.timeline-year {
    font-weight: 600;
    color: #3b5998;
    width: 3rem;
    flex-shrink: 0;
}
.timeline-content {
    flex: 1;
}
.timeline-detail {
    color: #666;
    font-size: 0.85rem;
}

/* Compact CTA buttons */
.cta-buttons {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.cta-buttons .stButton>button {
    min-width: 200px;
}
</style>
"""


def render() -> None:
    st.set_page_config(page_title="Face De-Identification Toolkit", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ================================================================== #
    # Top bar: title (matches other tabs)
    # ================================================================== #
    # Logo + title in a single row — title centered across both columns
    col_title, col_logo = st.columns([7, 1])
    with col_title:
        st.title("Face De-Identification Toolkit")
    with col_logo:
        st.image(logo_path(), width=60)

    # Hero section
    st.markdown(
        """
        <h2 style="color: #3b5998; margin-bottom: 0.25rem;">Open-source benchmark suite for privacy enhancing techniques on facial biometrics</h2>
        <p style="margin-top: 0;">
            Evaluation protocols are open and reproducible.
            Source code on
            <a href="https://github.com/blazm/deid-toolkit" target="_blank" style="color: #3b5998; font-weight: 600;">GitHub</a>.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Navigation buttons near top — DEPRECATED: unified tablist navigation in app.py
    # st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
    # col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
    # with col_nav1:
    #     if st.button("🔬 Results", key="nav_results", use_container_width=True):
    #         st.session_state.current_page = "toolkit"
    #         st.rerun()
    # with col_nav2:
    #     if st.button("📊 Benchmarks", key="nav_benchmarks", use_container_width=True):
    #         st.session_state.current_page = "public"
    #         st.rerun()
    # with col_nav3:
    #     if st.button("📖 Docs", key="nav_docs", use_container_width=True):
    #         st.session_state.current_page = "docs"
    #         st.rerun()
    # with col_nav4:
    #     if st.button("🧠 Survey", key="nav_survey", use_container_width=True):
    #         st.session_state.current_page = "survey"
    #         st.rerun()
    # st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================== #
    # RESEARCH FOUNDATION (compact)
    # ================================================================== #
    st.markdown('<div class="deid-section">Research Foundation</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 0.95rem; line-height: 1.4;">
        The DeID Toolkit originates from research at the
        <a href="https://www.fri.uni-lj.si/en/laboratory/lrv-26" target="_blank" style="color: #3b5998;">
        Computer Vision Laboratory</a>,
        Faculty of Computer and Information Science, University of Ljubljana.
        It builds on a body of work developing generative models for privacy-preserving
        facial biometrics — from early GAN-based approaches through k-Anonymity
        methods to controllable privacy protection.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ================================================================== #
    # MOTIVATION
    # ================================================================== #
    st.markdown('<div class="deid-section">Why Face De-Identification?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 0.95rem; line-height: 1.5; padding: 0.5rem 0;">
        <p>
        Facial biometric systems are increasingly used in security, healthcare, and border control.
        However, the widespread collection and sharing of face databases raises serious privacy concerns
        — once biometric data is leaked, it cannot be changed like a password.
        </p>
        <p>
        <strong>Face de-identification</strong> addresses this dilemma by transforming facial images
        to prevent unauthorized identification while preserving attributes needed for legitimate tasks
        (age estimation, emotion recognition, forensic analysis). Our research develops methods that
        achieve a controllable trade-off between privacy protection and data utility.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================================================================== #
    # COLLABORATORS
    # ================================================================== #
    st.markdown('<div class="deid-section">Collaborators</div>', unsafe_allow_html=True)

    collaborators = [
        {
            "name": "Blaž Meden",
            "role": "Principal Investigator",
            "link": "https://orcid.org/0000-0002-1690-479",
            "country": "Slovenia",
            "emoji": "🇸🇮",
        },
        {
            "name": "Žiga Emeršič",
            "role": "Senior Researcher",
            "link": "https://orcid.org/0000-0002-3726-9404",
            "country": "Slovenia",
            "emoji": "🇸🇮",
        },
        {
            "name": "Vitomir Štruc",
            "role": "Senior Researcher",
            "link": "https://orcid.org/0000-0002-3385-5780",
            "country": "Slovenia",
            "emoji": "🇸🇮",
        },
        {
            "name": "Peter Peer",
            "role": "Senior Researcher",
            "link": "https://orcid.org/0000-0001-9744-4035",
            "country": "Slovenia",
            "emoji": "🇸🇮",
        },
        {
            "name": "Sandi Ljubičić",
            "role": "Senior Researcher",
            "link": None,
            "country": "Croatia",
            "emoji": "🇭🇷",
        },
        {
            "name": "Manfred Gonzalez-Hernandez",
            "role": "Master's Student (EMAI)",
            "link": "https://orcid.org/0000-0002-5408-7901",
            "country": "Costa Rica",
            "emoji": "🇨🇷",
        },
        {
            "name": "Jernej Sabadin",
            "role": "PhD Candidate",
            "link": None,
            "country": "Slovenia",
            "emoji": "🇸🇮",
        },
        {
            "name": "Darian Tomažević",
            "role": "PhD Candidate",
            "link": None,
            "country": "Slovenia",
            "emoji": "🇸🇮",
        },
        {
            "name": "Esteban Leiva-Montenegro",
            "role": "Bachelor Student (internship)",
            "link": "https://orcid.org/0009-0004-3523-0057",
            "country": "Costa Rica",
            "emoji": "🇨🇷",
        },
        {
            "name": "Matthieu Pillot",
            "role": "Bachelor Student (internship)",
            "link": "https://orcid.org/0009-0006-3179-9152",
            "country": "France",
            "emoji": "🇫🇷",
        },
    ]

    # Build inline HTML for all collaborators
    card_html = ""
    for collab in collaborators:
        name_html = collab["name"]
        if collab["link"]:
            name_html = f'<a href="{collab["link"]}" target="_blank" style="color: #3b5998; font-weight: 600;">{collab["name"]}</a>'
        card_html += f"""
            <div style="background: #f8f9fa; padding: 0.6rem 0.8rem; border-radius: 4px;">
                <strong>{name_html}</strong><br>
                <span style="font-size: 0.85rem; color: #666;">{collab["role"]}</span><br>
                <span style="font-size: 0.85rem; color: #666;">{collab["emoji"]} {collab["country"]}</span>
            </div>"""

    # Use overflow:auto so all items are scrollable inside the iframe —
    # no matter how the grid wraps, nothing is hidden.
    st.components.v1.html(
        f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; width: 100%;">
        {card_html}
        </div>
        <style>
            body {{ margin: 0; padding: 0; overflow-y: scroll; }}
        </style>
        """,
        height=400,
    )

    # ================================================================== #
    # KEY PUBLICATIONS (two-column layout)
    # ================================================================== #
    st.markdown('<div class="deid-section">Key Publications</div>', unsafe_allow_html=True)

    st.markdown('<div class="pub-container">', unsafe_allow_html=True)

    pubs_col1 = [
        {
            "year": 2024,
            "title": "Deidentifikacija obrazov z nadzorom ravni varovanja zasebnosti",
            "venue": "Elektrotehniški vestnik",
            "citations": None,
        },
        {
            "year": 2023,
            "title": "Face deidentification with controllable privacy protection",
            "venue": "Image and Vision Computing",
            "citations": "16",
        },
        {
            "year": 2021,
            "title": "Privacy-enhancing face biometrics: a comprehensive survey",
            "venue": "IEEE TIFS",
            "citations": "149",
        },
    ]
    for p in pubs_col1:
        c_badge = f' ({p["citations"]} citations)' if p["citations"] else ""
        st.markdown(
            f"""
            <div class="pub-item">
                <span class="pub-year">{p["year"]}</span>
                <span class="pub-title">{p["title"]}</span>
                — <span class="pub-venue">{p["venue"]}</span>{c_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )
    pubs_col2 = [
        {
            "year": 2018,
            "title": "k-Same-Net: k-Anonymity via GANs",
            "venue": "Entropy",
            "citations": "59",
        },
        {
            "year": 2017,
            "title": "Face deidentification with generative deep neural networks",
            "venue": "IET Signal Processing",
            "citations": "47",
        },
    ]
    for p in pubs_col2:
        c_badge = f' ({p["citations"]} citations)' if p["citations"] else ""
        st.markdown(
            f"""
            <div class="pub-item">
                <span class="pub-year">{p["year"]}</span>
                <span class="pub-title">{p["title"]}</span>
                — <span class="pub-venue">{p["venue"]}</span>{c_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)  # Close pub-container

    # ================================================================== #
    # DOCTORAL DISSERTATION
    # ================================================================== #
    st.markdown('<div class="deid-section">Doctoral Dissertation</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="deid-card">
            <strong>Blaž Meden.</strong>
            <em>Face deidentification with generative neural networks</em>.
            UL FRI, 2023. 227 pages.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================================================================== #
    # ACTIVE PROJECTS
    # ================================================================== #
    st.markdown('<div class="deid-section">Active and Past Projects</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="project-badges">',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="project-badge">
            <strong>DeepFake DAD</strong>
            DeepFake detection via anomaly methods (ARRS J2-50065, 2023–2026)
        </div>
        <div class="project-badge">
            <strong>MIXBAI</strong>
            Interpretable biometric AI (ARRS J2-50069, 2023–2026)
        </div>
        <div class="project-badge">
            <strong>Humanities Rock!</strong>
            Real-time face de-identification showcase for European Researchers' Night 2024
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================== #
    # METHOD EVOLUTION TIMELINE (compact)
    # ================================================================== #
    st.markdown('<div class="deid-section">Method Evolution</div>', unsafe_allow_html=True)

    timeline = [
        ("2017", "GAN-based face de-identification", "IET Signal Processing"),
        ("2018", "k-Same-Net: k-Anonymity via GANs", "Entropy (59 citations)"),
        ("2021", "Comprehensive survey", "IEEE TIFS (149 citations)"),
        ("2023", "Controllable privacy protection", "Image and Vision Computing"),
        ("2023", "Doctoral dissertation", "227 pages, UL FRI"),
        ("2025", "Open-source DeID Toolkit", "This project"),
    ]

    for year, label, detail in timeline:
        st.markdown(
            f'<div class="timeline-row">'
            f'<div class="timeline-dot"></div>'
            f'<div class="timeline-year">{year}</div>'
            f'<div class="timeline-content"><strong>{label}</strong>'
            f' — <span class="timeline-detail">{detail}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ================================================================== #
    # Bottom note
    # ================================================================== #
    st.caption("Browse Docs, Results, Benchmarks, and Survey tabs above.")
