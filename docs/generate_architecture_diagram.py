"""Generate the architecture diagram used by README and the dashboard."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "architecture_diagram.png"

WIDTH = 2400
HEIGHT = 1500

COLORS = {
    "bg": "#f6f8fb",
    "ink": "#17202a",
    "muted": "#5f6d7c",
    "line": "#cfd9e6",
    "shadow": "#dbe3ed",
    "teal": "#0b6f6a",
    "teal_light": "#e8f5f2",
    "blue": "#315a9a",
    "blue_light": "#eaf0fb",
    "green": "#2f7d4d",
    "green_light": "#eaf6e8",
    "amber": "#9a5b00",
    "amber_light": "#fff4df",
    "purple": "#6b4bb3",
    "purple_light": "#f1ecfb",
    "surface": "#ffffff",
}


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["bg"])
    draw = ImageDraw.Draw(image)

    _title(draw)
    _business_goal(draw)
    _draw_main_flow(draw)
    _draw_learning_artifacts(draw)
    _draw_serving_path(draw)
    _footer(draw)

    image.save(OUTPUT, quality=95)


def _title(draw: ImageDraw.ImageDraw) -> None:
    draw.text(
        (90, 78),
        "learning-to-rank-distillation architecture",
        fill=COLORS["ink"],
        font=_font(44, True),
    )
    draw.text(
        (90, 130),
        "A governed ranking lifecycle: teacher quality, student serving, "
        "fairness evidence, and promotion policy",
        fill=COLORS["muted"],
        font=_font(24),
    )


def _business_goal(draw: ImageDraw.ImageDraw) -> None:
    box = (1610, 58, 2260, 165)
    _round_box(draw, box, fill=COLORS["surface"], outline="#b8c8db", radius=22, shadow=True)
    _center_text(
        draw,
        "Business question",
        (box[0], box[1] + 18, box[2], box[1] + 48),
        _font(23, True),
        COLORS["blue"],
    )
    _center_text(
        draw,
        "Is this model good enough, fast enough, and fair enough to serve?",
        (box[0] + 25, box[1] + 55, box[2] - 25, box[3] - 16),
        _font(18),
        COLORS["muted"],
    )


def _draw_main_flow(draw: ImageDraw.ImageDraw) -> None:
    adapters = (90, 255, 560, 690)
    contract = (645, 255, 1085, 690)
    training = (1170, 255, 1715, 690)
    decisions = (1800, 255, 2260, 690)

    _panel(draw, adapters, "1. Data adapters", COLORS["teal_light"], "#c7ddd9")
    _small_card(
        draw,
        (125, 345, 300, 430),
        "Amazon ESCI",
        "primary public LTR data",
        COLORS["teal"],
        COLORS["teal_light"],
    )
    _small_card(
        draw,
        (325, 345, 525, 430),
        "RecTour",
        "guarded travel adapter",
        COLORS["teal"],
        COLORS["teal_light"],
    )
    _small_card(
        draw,
        (125, 460, 300, 545),
        "Synthetic",
        "CI + fairness stress",
        COLORS["teal"],
        COLORS["teal_light"],
    )
    _small_card(
        draw,
        (325, 460, 525, 545),
        "MovieLens",
        "optional quickstart",
        COLORS["teal"],
        COLORS["teal_light"],
    )

    _panel(draw, contract, "2. Shared contract", COLORS["blue_light"], "#c9d6ea")
    _wide_card(
        draw,
        (690, 335, 1040, 435),
        "RankingExample",
        "shared query-item contract",
        COLORS["blue"],
        COLORS["blue_light"],
    )
    _wide_card(
        draw,
        (690, 465, 1040, 545),
        "Feature vectorizers",
        "numeric + categorical one-hot",
        COLORS["blue"],
        COLORS["surface"],
    )
    _pill(draw, (700, 575, 825, 613), "query group", COLORS["blue"], COLORS["blue_light"])
    _pill(draw, (850, 575, 955, 613), "label", COLORS["green"], COLORS["green_light"])
    _pill(draw, (975, 575, 1068, 613), "group", COLORS["amber"], COLORS["amber_light"])

    _panel(draw, training, "3. Training and distillation", COLORS["surface"], "#cfd9e6")
    _small_card(
        draw,
        (1215, 345, 1455, 455),
        "Teacher",
        "LightGBM\nTransformer option",
        COLORS["green"],
        COLORS["green_light"],
    )
    _small_card(
        draw,
        (1215, 510, 1455, 620),
        "Student",
        "two-tower\nserving model",
        COLORS["blue"],
        COLORS["blue_light"],
    )
    _small_card(
        draw,
        (1500, 370, 1688, 490),
        "KD signals",
        "teacher behavior",
        COLORS["purple"],
        COLORS["purple_light"],
    )
    _small_card(
        draw,
        (1500, 535, 1688, 620),
        "No-KD control",
        "label baseline",
        COLORS["amber"],
        COLORS["surface"],
    )
    _arrow(draw, (1455, 400), (1498, 430), COLORS["green"])
    _arrow(draw, (1455, 560), (1498, 450), COLORS["purple"])
    _arrow(draw, (1455, 565), (1498, 575), COLORS["amber"], width=4)

    _panel(draw, decisions, "4. Decision evidence", COLORS["amber_light"], "#efd5a5")
    _decision_stack(draw, decisions)

    _arrow(draw, (560, 458), (645, 458), COLORS["teal"], width=6)
    _arrow(draw, (1085, 458), (1170, 458), COLORS["teal"], width=6)
    _arrow(draw, (1715, 458), (1800, 458), COLORS["amber"], width=6)
    _label(draw, (1720, 415, 1792, 450), "evidence", COLORS["amber"])


def _decision_stack(draw: ImageDraw.ImageDraw, panel: tuple[int, int, int, int]) -> None:
    x1, y1, x2, _ = panel
    cards = [
        ("Quality", "NDCG"),
        ("Serving cost", "latency + size"),
        ("Fairness", "exposure"),
        ("Promotion", "gate + lineage"),
    ]
    top = y1 + 96
    for index, (title, subtitle) in enumerate(cards):
        y = top + index * 82
        _decision_card(
            draw,
            (x1 + 62, y, x2 - 62, y + 68),
            title,
            subtitle,
        )


def _draw_learning_artifacts(draw: ImageDraw.ImageDraw) -> None:
    box = (90, 760, 560, 1085)
    _panel(draw, box, "Learning artifacts", COLORS["surface"], "#cfd9e6")
    _wide_card(
        draw,
        (135, 850, 515, 925),
        "HTML + Markdown guides",
        "summary, detailed, technical flows",
        COLORS["blue"],
        COLORS["surface"],
    )
    _wide_card(
        draw,
        (135, 950, 515, 1025),
        "Benchmark plots",
        "quality-latency + fairness trade-offs",
        COLORS["blue"],
        COLORS["surface"],
    )


def _draw_serving_path(draw: ImageDraw.ImageDraw) -> None:
    box = (645, 760, 1915, 1085)
    _panel(draw, box, "5. Production-shaped serving path", COLORS["purple_light"], "#d8cceb")
    y = 875
    cards = [
        ((695, y, 905, y + 85), "Student bundle", "weights, vectorizer\nmetadata, data hash"),
        ((970, y, 1180, y + 85), "Item embeddings", "precomputed item\nrepresentations"),
        ((1245, y, 1455, y + 85), "ANN search", "FAISS-compatible\nNumPy fallback"),
        ((1520, y, 1730, y + 85), "FastAPI service", "/health /rank\n/metrics"),
    ]
    for rect, title, subtitle in cards:
        _small_card(draw, rect, title, subtitle, COLORS["purple"], COLORS["purple_light"])
    for start, end in [
        ((905, y + 42), (970, y + 42)),
        ((1180, y + 42), (1245, y + 42)),
        ((1455, y + 42), (1520, y + 42)),
    ]:
        _arrow(draw, start, end, COLORS["purple"], width=6)

    _small_card(
        draw,
        (970, 1000, 1180, 1070),
        "Registry",
        "filesystem stages",
        COLORS["purple"],
        COLORS["surface"],
    )
    _small_card(
        draw,
        (1245, 1000, 1455, 1070),
        "Docker + k6",
        "container + load test",
        COLORS["purple"],
        COLORS["surface"],
    )
    _arrow(draw, (1180, 1035), (1245, 1035), COLORS["purple"], width=5)

    # Promotion to serving uses a right-angle path that avoids the model-training arrows.
    _polyline_arrow(
        draw,
        [(2030, 665), (2030, 725), (630, 725), (630, 917), (695, 917)],
        COLORS["amber"],
        width=5,
    )
    _label(draw, (1275, 682, 1840, 715), "approved model -> serving bundle", COLORS["amber"])


def _footer(draw: ImageDraw.ImageDraw) -> None:
    draw.text(
        (90, 1245),
        "Primary path: Amazon ESCI -> RankingExample -> teacher/student "
        "distillation -> benchmark/fairness -> governed serving bundle.",
        fill=COLORS["muted"],
        font=_font(22, True),
    )
    draw.text(
        (90, 1300),
        "Guardrails: RecTour stays guarded without real files; synthetic is "
        "deterministic CI/demo data; MovieLens is optional quickstart.",
        fill=COLORS["muted"],
        font=_font(22, True),
    )


def _panel(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, fill: str, outline: str
) -> None:
    _round_box(draw, box, fill=fill, outline=outline, radius=30)
    draw.text((box[0] + 28, box[1] + 25), title, fill=COLORS["muted"], font=_font(26, True))


def _small_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    accent: str,
    fill: str,
) -> None:
    _round_box(draw, box, fill=fill, outline=_soft(accent), radius=18, shadow=True)
    _center_text(
        draw,
        title,
        (box[0] + 12, box[1] + 14, box[2] - 12, box[1] + 45),
        _font(24, True),
        accent,
    )
    _center_multiline(
        draw,
        subtitle,
        (box[0] + 14, box[1] + 47, box[2] - 14, box[3] - 10),
        _font(19),
        COLORS["muted"],
    )


def _wide_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    accent: str,
    fill: str,
) -> None:
    _small_card(draw, box, title, subtitle, accent, fill)


def _decision_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
) -> None:
    _round_box(
        draw, box, fill=COLORS["surface"], outline=_soft(COLORS["amber"]), radius=18, shadow=True
    )
    _center_text(
        draw,
        title,
        (box[0] + 14, box[1] + 10, box[2] - 14, box[1] + 42),
        _font(27, True),
        COLORS["amber"],
    )
    _center_text(
        draw,
        subtitle,
        (box[0] + 14, box[1] + 43, box[2] - 14, box[3] - 8),
        _font(21),
        COLORS["muted"],
    )


def _pill(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, accent: str, fill: str
) -> None:
    _round_box(draw, box, fill=fill, outline=accent, radius=18)
    _center_text(draw, text, box, _font(18, True), accent)


def _label(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, color: str
) -> None:
    _center_multiline(draw, text, box, _font(19, True), color)


def _round_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    radius: int,
    shadow: bool = False,
) -> None:
    if shadow:
        shadow_box = (box[0] + 8, box[1] + 10, box[2] + 8, box[3] + 10)
        draw.rounded_rectangle(shadow_box, radius=radius, fill=COLORS["shadow"], outline=None)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    *,
    width: int = 4,
) -> None:
    draw.line((start, end), fill=color, width=width)
    _arrow_head(draw, start, end, color, size=17)


def _polyline_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: str,
    *,
    width: int = 4,
) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    _arrow_head(draw, points[-2], points[-1], color, size=17)


def _arrow_head(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    *,
    size: int,
) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (
        end[0] - size * math.cos(angle - math.pi / 6),
        end[1] - size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - size * math.cos(angle + math.pi / 6),
        end[1] - size * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, left, right], fill=color)


def _center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) / 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) / 2
    draw.text((x, y), text, font=font, fill=fill)


def _center_multiline(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    lines: list[str] = []
    for raw_line in text.splitlines():
        lines.extend(_wrap(draw, raw_line, font, box[2] - box[0]))
    line_height = font.size + 4 if hasattr(font, "size") else 18
    total_height = len(lines) * line_height
    y = box[1] + (box[3] - box[1] - total_height) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height


def _wrap(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    width: int,
) -> list[str]:
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) == 1:
        return lines
    return [line for wrapped in lines for line in textwrap.wrap(wrapped, width=32) or [wrapped]]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
        Path("/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _soft(color: str) -> str:
    if color == COLORS["teal"]:
        return "#86cbc4"
    if color == COLORS["blue"]:
        return "#adc1e0"
    if color == COLORS["green"]:
        return "#b6d5ad"
    if color == COLORS["amber"]:
        return "#e8bd72"
    if color == COLORS["purple"]:
        return "#c6b5e7"
    return COLORS["line"]


if __name__ == "__main__":
    main()
