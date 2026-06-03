from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import MNAddDataset
from models import ConceptFilterMLP, DigitCNN
from reasoning import predict_digits


CNN_CHECKPOINT = PROJECT_ROOT / "results" / "checkpoints" / "digit_cnn.pth"
FILTER_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
    / "concept_filter_uncertainty_lambda_0.2.pth"
)
DATA_ROOT = PROJECT_ROOT / "data"

METRICS = [
    ("Baseline reasoning accuracy", "0.9754"),
    ("Filtered reasoning accuracy", "0.9794"),
    ("Baseline concept accuracy", "0.9729"),
    ("Filtered concept accuracy", "0.9790"),
    ("Baseline shortcut count", "143"),
    ("Filtered shortcut count", "105"),
    ("Fixed cases", "101"),
    ("Broken cases", "43"),
    ("Net gain", "+58"),
]

CASE_GROUPS = {
    "All cases": None,
    "Shortcut cases": {"CNN shortcut", "Filter fixed shortcut"},
    "Fixed by filter": {"Filter fixed shortcut"},
    "Broken by filter": {"Filter broke correct prediction"},
    "Normal cases": {"Normal case"},
}

STATUS_COLORS = {
    "Normal case": ("#38bdf8", "rgba(56, 189, 248, 0.14)"),
    "CNN shortcut": ("#f59e0b", "rgba(245, 158, 11, 0.16)"),
    "Filter fixed shortcut": ("#22c55e", "rgba(34, 197, 94, 0.16)"),
    "Filter broke correct prediction": ("#ef4444", "rgba(239, 68, 68, 0.16)"),
}


def _load_state_dict(model, checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    return model


@st.cache_resource(show_spinner="Loading trained models...")
def load_models(device_name="cpu"):
    device = torch.device(device_name)

    cnn = DigitCNN().to(device)
    _load_state_dict(cnn, CNN_CHECKPOINT, device)
    cnn.eval()

    filter_model = ConceptFilterMLP().to(device)
    _load_state_dict(filter_model, FILTER_CHECKPOINT, device)
    filter_model.eval()

    return cnn, filter_model, device


@st.cache_resource(show_spinner="Loading MNIST paired test dataset...")
def load_dataset():
    return MNAddDataset(root=str(DATA_ROOT), train=False)


@st.cache_data(show_spinner=False)
def get_prediction_for_sample(index, device_name="cpu"):
    cnn, filter_model, device = load_models(device_name)
    dataset = load_dataset()

    image, concepts, target = dataset[index]
    image_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_digit1_logits, cnn_digit2_logits = cnn(image_batch)
        cnn_digit1_probs = F.softmax(cnn_digit1_logits, dim=1)
        cnn_digit2_probs = F.softmax(cnn_digit2_logits, dim=1)

        concept_probs = torch.cat([cnn_digit1_probs, cnn_digit2_probs], dim=1)
        original_logits = torch.cat([cnn_digit1_logits, cnn_digit2_logits], dim=1)

        filtered_digit1_logits, filtered_digit2_logits = filter_model(
            concept_probs, original_logits
        )
        filtered_digit1_probs = F.softmax(filtered_digit1_logits, dim=1)
        filtered_digit2_probs = F.softmax(filtered_digit2_logits, dim=1)

        cnn_digit1_pred, cnn_digit2_pred = predict_digits(
            cnn_digit1_logits, cnn_digit2_logits
        )
        filtered_digit1_pred, filtered_digit2_pred = predict_digits(
            filtered_digit1_logits, filtered_digit2_logits
        )

    true_digit1 = _as_int(concepts[0])
    true_digit2 = _as_int(concepts[1])
    cnn_pred_digit1 = _as_int(cnn_digit1_pred)
    cnn_pred_digit2 = _as_int(cnn_digit2_pred)
    filtered_pred_digit1 = _as_int(filtered_digit1_pred)
    filtered_pred_digit2 = _as_int(filtered_digit2_pred)
    true_parity = _as_int(target)

    return {
        "index": index,
        "image": image.squeeze(0).numpy(),
        "true_digits": (true_digit1, true_digit2),
        "cnn_digits": (cnn_pred_digit1, cnn_pred_digit2),
        "filtered_digits": (filtered_pred_digit1, filtered_pred_digit2),
        "true_parity": true_parity,
        "cnn_parity": (cnn_pred_digit1 + cnn_pred_digit2) % 2,
        "filtered_parity": (filtered_pred_digit1 + filtered_pred_digit2) % 2,
        "cnn_confidences": (
            float(cnn_digit1_probs[0, cnn_pred_digit1].item()),
            float(cnn_digit2_probs[0, cnn_pred_digit2].item()),
        ),
        "filtered_confidences": (
            float(filtered_digit1_probs[0, filtered_pred_digit1].item()),
            float(filtered_digit2_probs[0, filtered_pred_digit2].item()),
        ),
        "cnn_digit1_probs": cnn_digit1_probs.squeeze(0).cpu().tolist(),
        "cnn_digit2_probs": cnn_digit2_probs.squeeze(0).cpu().tolist(),
        "filtered_digit1_probs": filtered_digit1_probs.squeeze(0).cpu().tolist(),
        "filtered_digit2_probs": filtered_digit2_probs.squeeze(0).cpu().tolist(),
    }


@st.cache_data(show_spinner="Indexing case types...")
def get_case_catalog(device_name="cpu"):
    cnn, filter_model, device = load_models(device_name)
    dataset = load_dataset()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

    catalog = []
    sample_index = 0

    with torch.no_grad():
        for images, concepts, target in loader:
            images = images.to(device)
            concepts = concepts.to(device)
            target = target.to(device)

            cnn_digit1_logits, cnn_digit2_logits = cnn(images)
            cnn_digit1_probs = F.softmax(cnn_digit1_logits, dim=1)
            cnn_digit2_probs = F.softmax(cnn_digit2_logits, dim=1)
            cnn_digit1_pred, cnn_digit2_pred = predict_digits(
                cnn_digit1_logits, cnn_digit2_logits
            )

            concept_probs = torch.cat([cnn_digit1_probs, cnn_digit2_probs], dim=1)
            original_logits = torch.cat(
                [cnn_digit1_logits, cnn_digit2_logits], dim=1
            )
            filtered_digit1_logits, filtered_digit2_logits = filter_model(
                concept_probs, original_logits
            )
            filtered_digit1_pred, filtered_digit2_pred = predict_digits(
                filtered_digit1_logits, filtered_digit2_logits
            )

            for batch_offset in range(images.size(0)):
                true_digits = (
                    _as_int(concepts[batch_offset, 0]),
                    _as_int(concepts[batch_offset, 1]),
                )
                cnn_digits = (
                    _as_int(cnn_digit1_pred[batch_offset]),
                    _as_int(cnn_digit2_pred[batch_offset]),
                )
                filtered_digits = (
                    _as_int(filtered_digit1_pred[batch_offset]),
                    _as_int(filtered_digit2_pred[batch_offset]),
                )
                true_parity = _as_int(target[batch_offset])
                cnn_parity = (cnn_digits[0] + cnn_digits[1]) % 2
                filtered_parity = (filtered_digits[0] + filtered_digits[1]) % 2
                status = get_case_status(
                    true_digits=true_digits,
                    cnn_digits=cnn_digits,
                    filtered_digits=filtered_digits,
                    true_parity=true_parity,
                    cnn_parity=cnn_parity,
                )

                catalog.append(
                    {
                        "index": sample_index + batch_offset,
                        "status": status,
                        "true_digits": true_digits,
                        "cnn_digits": cnn_digits,
                        "filtered_digits": filtered_digits,
                        "true_parity": true_parity,
                        "cnn_parity": cnn_parity,
                        "filtered_parity": filtered_parity,
                    }
                )

            sample_index += images.size(0)

    return catalog


def get_case_status(true_digits, cnn_digits, filtered_digits, true_parity, cnn_parity):
    cnn_concepts_correct = cnn_digits == true_digits
    filtered_concepts_correct = filtered_digits == true_digits
    cnn_reasoning_correct = cnn_parity == true_parity
    cnn_shortcut = cnn_reasoning_correct and not cnn_concepts_correct

    if cnn_shortcut and filtered_concepts_correct:
        return "Filter fixed shortcut"
    if cnn_shortcut:
        return "CNN shortcut"
    if cnn_concepts_correct and not filtered_concepts_correct:
        return "Filter broke correct prediction"
    return "Normal case"


def _format_parity(value):
    return f"{value} ({'odd' if value else 'even'})"


def _as_int(value):
    return int(value.item()) if hasattr(value, "item") else int(value)


def render_page_styles():
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stHeaderActionElements"],
        .stDeployButton,
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
        }
        .pipeline {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: center;
            margin: 0.6rem 0 1.1rem 0;
        }
        .pipeline-step {
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 8px;
            padding: 0.45rem 0.7rem;
            background: rgba(148, 163, 184, 0.08);
            font-weight: 650;
        }
        .pipeline-arrow {
            color: rgb(148, 163, 184);
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            font-weight: 750;
            padding: 0.35rem 0.75rem;
            margin-bottom: 0.75rem;
        }
        .stage-card {
            border: 1px solid rgba(148, 163, 184, 0.32);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.55rem;
            background: rgba(148, 163, 184, 0.08);
        }
        .stage-label {
            color: rgb(148, 163, 184);
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 0.25rem;
        }
        .stage-digits {
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }
        .stage-parity {
            color: rgb(203, 213, 225);
            font-weight: 650;
        }
        .stage-confidence {
            color: rgb(148, 163, 184);
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }
        .case-count {
            color: rgb(148, 163, 184);
            font-size: 0.92rem;
        }
        .metric-rail {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            padding: 0.9rem;
            background: rgba(15, 23, 42, 0.35);
        }
        .metric-card {
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
            padding: 0.55rem 0;
        }
        .metric-card:last-child {
            border-bottom: 0;
        }
        .metric-label {
            color: rgb(148, 163, 184);
            font-size: 0.78rem;
            font-weight: 700;
        }
        .metric-value {
            color: rgb(248, 250, 252);
            font-size: 1.25rem;
            font-weight: 800;
        }
        .prob-panel {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            padding: 0.85rem;
            margin-bottom: 0.75rem;
            background: rgba(148, 163, 184, 0.07);
        }
        .prob-title {
            color: rgb(226, 232, 240);
            font-weight: 800;
            margin-bottom: 0.55rem;
        }
        .prob-row {
            display: grid;
            grid-template-columns: 1.6rem 1fr 3rem;
            align-items: center;
            gap: 0.55rem;
            margin: 0.24rem 0;
        }
        .prob-digit {
            color: rgb(226, 232, 240);
            font-weight: 800;
            text-align: right;
        }
        .prob-track {
            height: 0.65rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.22);
            overflow: hidden;
        }
        .prob-fill {
            height: 100%;
            border-radius: 999px;
            background: rgb(56, 189, 248);
        }
        .prob-value {
            color: rgb(203, 213, 225);
            font-variant-numeric: tabular-nums;
            font-size: 0.78rem;
            text-align: right;
        }
        .prob-tags {
            color: rgb(148, 163, 184);
            font-size: 0.75rem;
            margin-left: 2.15rem;
            min-height: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline():
    steps = [
        "MNIST pair",
        "CNN concepts",
        "symbolic reasoning",
        "concept filter",
        "corrected reasoning",
    ]
    html = '<div class="pipeline">'
    for position, step in enumerate(steps):
        if position > 0:
            html += '<span class="pipeline-arrow">-></span>'
        html += f'<span class="pipeline-step">{step}</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def filter_catalog(catalog, case_group):
    statuses = CASE_GROUPS[case_group]
    if statuses is None:
        return catalog
    return [record for record in catalog if record["status"] in statuses]


def render_status_badge(status):
    foreground, background = STATUS_COLORS[status]
    st.markdown(
        (
            f'<span class="status-badge" '
            f'style="color: {foreground}; background: {background};">'
            f"{status}</span>"
        ),
        unsafe_allow_html=True,
    )


def render_stage_card(title, digits, parity, confidences=None):
    confidence_html = ""
    if confidences is not None:
        confidence_html = (
            '<div class="stage-confidence">'
            f"Confidence: {confidences[0]:.1%}, {confidences[1]:.1%}"
            "</div>"
        )

    st.markdown(
        (
            '<div class="stage-card">'
            f'<div class="stage-label">{title}</div>'
            f'<div class="stage-digits">{digits[0]}, {digits[1]}</div>'
            f'<div class="stage-parity">Parity: {_format_parity(parity)}</div>'
            f"{confidence_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def choose_case(records):
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = records[0]["index"]

    valid_indices = [record["index"] for record in records]
    if st.session_state.selected_index not in valid_indices:
        st.session_state.selected_index = valid_indices[0]

    current_position = valid_indices.index(st.session_state.selected_index)

    nav_columns = st.columns([1, 1, 1, 2, 2])

    if nav_columns[0].button("Previous", width="stretch"):
        st.session_state.selected_index = valid_indices[
            (current_position - 1) % len(valid_indices)
        ]
        st.rerun()

    if nav_columns[1].button("Random", width="stretch"):
        st.session_state.selected_index = random.choice(valid_indices)
        st.rerun()

    if nav_columns[2].button("Next", width="stretch"):
        st.session_state.selected_index = valid_indices[
            (current_position + 1) % len(valid_indices)
        ]
        st.rerun()

    nav_columns[3].markdown(
        (
            f'<div class="case-count">Showing case {current_position + 1} of '
            f"{len(valid_indices)} · sample #{st.session_state.selected_index}</div>"
        ),
        unsafe_allow_html=True,
    )

    selected_index = nav_columns[4].selectbox(
        "Sample",
        options=valid_indices,
        index=current_position,
        format_func=lambda index: f"Sample #{index}",
    )
    st.session_state.selected_index = selected_index
    return st.session_state.selected_index


def render_metric_rail():
    html = '<div class="metric-rail"><h3>Project Metrics</h3>'
    for label, value in METRICS:
        html += (
            '<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_probability_panel(title, probabilities, true_digit, predicted_digit):
    html = f'<div class="prob-panel"><div class="prob-title">{title}</div>'
    for digit, probability in enumerate(probabilities):
        probability = float(probability)
        width = max(0.0, min(probability * 100.0, 100.0))
        tags = []
        if digit == predicted_digit:
            tags.append("pred")
        if digit == true_digit:
            tags.append("true")
        tag_text = " · ".join(tags)
        html += (
            '<div class="prob-row">'
            f'<div class="prob-digit">{digit}</div>'
            '<div class="prob-track">'
            f'<div class="prob-fill" style="width: {width:.2f}%"></div>'
            "</div>"
            f'<div class="prob-value">{probability:.2%}</div>'
            "</div>"
            f'<div class="prob-tags">{tag_text}</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_probability_view(prediction):
    st.subheader("Probabilities")

    true_digit1, true_digit2 = prediction["true_digits"]
    cnn_digit1, cnn_digit2 = prediction["cnn_digits"]
    filtered_digit1, filtered_digit2 = prediction["filtered_digits"]

    first_row = st.columns(2)
    with first_row[0]:
        render_probability_panel(
            "CNN digit 1",
            prediction["cnn_digit1_probs"],
            true_digit1,
            cnn_digit1,
        )
    with first_row[1]:
        render_probability_panel(
            "Filtered digit 1",
            prediction["filtered_digit1_probs"],
            true_digit1,
            filtered_digit1,
        )

    second_row = st.columns(2)
    with second_row[0]:
        render_probability_panel(
            "CNN digit 2",
            prediction["cnn_digit2_probs"],
            true_digit2,
            cnn_digit2,
        )
    with second_row[1]:
        render_probability_panel(
            "Filtered digit 2",
            prediction["filtered_digit2_probs"],
            true_digit2,
            filtered_digit2,
        )


def render_case_controls(catalog):
    st.subheader("Case Lens")

    case_group = st.radio(
        "Case lens",
        options=list(CASE_GROUPS.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )
    records = filter_catalog(catalog, case_group)

    if not records:
        st.warning(f"No samples found for {case_group}.")
        st.stop()

    selected_index = choose_case(records)
    st.markdown(
        (
            f'<div class="case-count">{len(records)} matching cases out of '
            f"{len(catalog)} paired MNIST test samples.</div>"
        ),
        unsafe_allow_html=True,
    )
    return selected_index


def render_sample_viewer(prediction):
    st.subheader("Selected Case")

    left_column, right_column = st.columns([1, 1.2])

    with left_column:
        st.image(
            prediction["image"],
            caption=f"Sample index {prediction['index']}",
            clamp=True,
            width="stretch",
        )

    true_digits = prediction["true_digits"]
    cnn_digits = prediction["cnn_digits"]
    filtered_digits = prediction["filtered_digits"]

    status = get_case_status(
        true_digits=true_digits,
        cnn_digits=cnn_digits,
        filtered_digits=filtered_digits,
        true_parity=prediction["true_parity"],
        cnn_parity=prediction["cnn_parity"],
    )

    with right_column:
        render_status_badge(status)

        render_stage_card(
            "Ground truth",
            true_digits,
            prediction["true_parity"],
        )
        render_stage_card(
            "CNN concepts",
            cnn_digits,
            prediction["cnn_parity"],
            prediction["cnn_confidences"],
        )
        render_stage_card(
            "Filtered concepts",
            filtered_digits,
            prediction["filtered_parity"],
            prediction["filtered_confidences"],
        )


def main():
    st.set_page_config(page_title="Reasoning Shortcut Explorer", layout="wide")
    render_page_styles()
    device_name = "cpu"

    st.title("Reasoning Shortcut Explorer")
    render_pipeline()

    metric_column, main_column = st.columns([0.85, 2.45], gap="large")

    with metric_column:
        render_metric_rail()

    with main_column:
        catalog = get_case_catalog(device_name)
        sample_index = render_case_controls(catalog)
        prediction = get_prediction_for_sample(sample_index, device_name)
        render_sample_viewer(prediction)
        render_probability_view(prediction)


if __name__ == "__main__":
    main()
