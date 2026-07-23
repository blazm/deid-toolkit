"""Crowdsourced Verification — public-facing questionnaire.

Accessible at: http://localhost:8501/?page=survey
Sharable link: http://193.2.76.178:8501/?page=survey
"""
from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from deid.explore.assets import logo_path
from deid.explore.survey_api import generate_pairs, submit_responses, reset_survey_results

RATING_LABELS = {
    1: "Different",
    2: "Probably different",
    3: "Probably same",
    4: "Same",
}


def _render_survey() -> None:
    st.set_page_config(page_title="Crowdsourced Verification", layout="wide", page_icon=logo_path())

    # Keyboard shortcut: press 1-4
    st.markdown(
        """
        <script>
        (function() {
            document.addEventListener('keydown', function(e) {
                var tag = (e.target.tagName || '').toLowerCase();
                if (tag === 'input' && e.target.type !== 'radio') return;
                if (tag === 'textarea' || tag === 'select') return;
                var num = e.key;
                if (num < '1' || num > '4') return;
                var inputs = document.querySelectorAll('input[type="range"]');
                for (var i = 0; i < inputs.length; i++) {
                    var val = parseInt(inputs[i].value);
                    if (val === 1) {
                        inputs[i].value = num;
                        inputs[i].dispatchEvent(new Event('input', {bubbles: true}));
                        e.preventDefault();
                        break;
                    }
                }
            });
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.title("Crowdsourced Verification")
    st.caption("Rate each pair: 1 = different, 4 = same. All pairs are mixed — you can't tell which are aligned vs de-identified.")
    st.markdown("---")

    # Session tracking
    if "survey_session" not in st.session_state:
        st.session_state.survey_session = {
            "started": False,
            "completed": False,
            "dataset": None,
            "attribute": None,
            "pairs": [],
            "answers": {},
            "submitted": False,
        }

    session = st.session_state.survey_session

    # --- Welcome / Consent Screen ---
    if not session["started"]:
        st.subheader("Purpose of the Survey")
        st.markdown(
            """
            This survey collects human evaluations of face de-identification techniques.
            The survey has two parts that are mixed together so you can't tell which is which:

            **Part 1 — Validation (aligned images):** Verify you can reliably distinguish faces.
            Accuracy below 80% flags unreliable responses.

            **Part 2 — De-identification (de-identified images):** Evaluate whether de-identification
            successfully hides identity.

            Your responses are completely anonymous and used only for research purposes.
            """
        )

        selected_dataset = "Arc2Face_data"
        selected_attr = "id"

        if st.button("Start Survey", type="primary", use_container_width=True):
            session["started"] = True
            session["completed"] = False
            session["dataset"] = selected_dataset
            session["attribute"] = selected_attr
            st.rerun()

    # --- Survey Screen ---
    elif not session["completed"] and session["started"]:
        dataset = session["dataset"]
        attribute = session["attribute"]

        # Generate pairs if not already generated
        if not session["pairs"]:
            try:
                session["pairs"] = generate_pairs(dataset)
            except ValueError as e:
                st.error(f"Cannot start survey: {e}")
                st.info("Ensure aligned images exist under `aligned_path/{dataset}/`.")
                return
            session["answers"] = {i: 0 for i in range(len(session["pairs"]))}

        total = len(session["pairs"])
        completed = sum(1 for v in session["answers"].values() if v != 0)
        has_deid = any(p.get("display") == "deid" for p in session["pairs"])
        if has_deid:
            st.caption(f"Progress: {completed}/{total} — all pairs mixed (validation + de-identification)")
        else:
            st.caption(f"Progress: {completed}/{total} — validation only (no de-identified images found)")
        st.caption(f"Evaluating attribute: **{attribute}**")

        # ── Compact two-column grid: each column holds one full pair ──
        col1, col2 = st.columns(2)
        cols = [col1, col2]

        for idx, pair in enumerate(session["pairs"]):
            col = cols[idx % 2]
            current = session["answers"].get(idx, 0) or 1

            with col:
                st.caption(f"Pair {idx + 1}/{total}")

                # Images inline (narrow)
                img_w = 120
                if pair["display"] == "aligned":
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(pair["image1_path"], width=img_w)
                    with c2:
                        st.image(pair["image2_path"], width=img_w)
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(pair["original1_path"], width=img_w)
                    with c2:
                        st.image(pair["original2_path"], width=img_w)

                # Slider with hidden label (just the thumb)
                rating = st.slider(
                    " ",
                    min_value=1, max_value=4, step=1,
                    value=current,
                    format="%d",
                    key=f"pair_{idx}_rating",
                    label_visibility="collapsed",
                    help="1 = different people, 4 = same person",
                )

                # Label below the slider (shows NEW value)
                st.caption(f"[{RATING_LABELS[rating]}]")

                session["answers"][idx] = rating

        st.markdown("---")

        # Submit + reset buttons
        col_sub, col_res = st.columns([1, 1])
        with col_sub:
            if st.button("Submit Responses", type="primary", use_container_width=True):
                session["completed"] = True
                session["submitted"] = True
                st.rerun()
        with col_res:
            if st.button("Reset All Survey Data", type="secondary", use_container_width=True):
                removed = reset_survey_results()
                st.info(f"Removed {removed} survey files. Rate limits reset.")
                st.rerun()

    # --- Results Screen ---
    elif session["completed"]:
        session_id = uuid.uuid4().hex[:12]
        session["session_id"] = session_id

        # Convert answers to response format
        responses = []
        for i, pair in enumerate(session["pairs"]):
            rating = session["answers"][i]
            answer_text = "Same person" if rating >= 3 else "Different people"
            pair_resp = {
                "pair_index": i,
                "pair_type": pair["pair_type"],
                "display": pair.get("display", "aligned"),
                "rating": rating,
                "answer_text": answer_text,
                "dataset": pair["dataset"],
                "ground_truth": pair["ground_truth"],
            }
            if "technique" in pair:
                pair_resp["technique"] = pair["technique"]
            responses.append(pair_resp)

        # Submit responses
        status, message = submit_responses(responses, session_id)

        if status == "success":
            st.success(message)
            st.info("Thank you for your contribution to research!")
        else:
            st.error(message)

        st.markdown("---")

        # ── Validation results ──
        val_indices = [i for i, p in enumerate(session["pairs"]) if p.get("display") == "aligned"]
        if val_indices:
            st.subheader("Validation Accuracy (Real Faces)")
            val_correct = 0
            for i in val_indices:
                pair = session["pairs"][i]
                rating = session["answers"][i]
                if pair["ground_truth"] == "same" and rating >= 3:
                    val_correct += 1
                elif pair["ground_truth"] == "different" and rating <= 2:
                    val_correct += 1
            total = len(val_indices)
            acc = val_correct / total if total > 0 else 0

            if acc >= 0.8:
                st.success(f"Your validation accuracy is **{acc:.0%}** ({val_correct}/{total}) — your responses are reliable.")
            elif acc >= 0.5:
                st.warning(f"Your validation accuracy is **{acc:.0%}** ({val_correct}/{total}) — some responses may need review.")
            else:
                st.error(f"Your validation accuracy is **{acc:.0%}** ({val_correct}/{total}) — responses may not be reliable for research data.")

        # ── De-identification results ──
        deid_indices = [i for i, p in enumerate(session["pairs"]) if p.get("display") == "deid"]
        if deid_indices:
            st.markdown("---")
            st.subheader("De-identification Results")
            deid_correct = 0
            for i in deid_indices:
                pair = session["pairs"][i]
                rating = session["answers"][i]
                if pair["ground_truth"] == "same" and rating >= 3:
                    deid_correct += 1
                elif pair["ground_truth"] == "different" and rating <= 2:
                    deid_correct += 1
            total = len(deid_indices)
            acc = deid_correct / total if total > 0 else 0

            st.markdown(f"**Identity Protection Rate:** {acc:.0%} ({deid_correct}/{total})")

            # Breakdown by technique
            tech_acc = {}
            for i in deid_indices:
                pair = session["pairs"][i]
                tech = pair.get("technique", "unknown")
                rating = session["answers"][i]
                if tech not in tech_acc:
                    tech_acc[tech] = {"correct": 0, "total": 0}
                if pair["ground_truth"] == "same" and rating >= 3:
                    tech_acc[tech]["correct"] += 1
                elif pair["ground_truth"] == "different" and rating <= 2:
                    tech_acc[tech]["correct"] += 1
                tech_acc[tech]["total"] += 1

            if tech_acc:
                st.markdown("**Protection Rate by Technique:**")
                for tech, a in tech_acc.items():
                    st.write(f"- **{tech}**: {a['correct']}/{a['total']} ({a['correct']/a['total']:.0%})")

        # Repeat or go back
        st.markdown("---")
        col_repeat, col_home = st.columns(2)
        with col_repeat:
            if st.button("Repeat Survey", use_container_width=True):
                st.session_state.survey_session = {
                    "started": False,
                    "completed": False,
                    "dataset": None,
                    "attribute": None,
                    "pairs": [],
                    "answers": {},
                    "submitted": False,
                }
                st.rerun()
        with col_home:
            if st.button("Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()

        # Reset button (below results)
        if st.button("Reset All Survey Results", type="secondary", use_container_width=True):
            removed = reset_survey_results()
            st.info(f"Removed {removed} survey files. Rate limits have been reset.")
            st.rerun()


def render() -> None:
    _render_survey()
