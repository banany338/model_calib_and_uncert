"""Small SVG plotting helpers so the project does not require matplotlib."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#475569",
    "#be123c",
]


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#111827}'
        '.small{font-size:11px;fill:#4b5563}.title{font-size:18px;font-weight:700}'
        '.axis{stroke:#111827;stroke-width:1.5}.grid{stroke:#e5e7eb;stroke-width:1}'
        '</style>',
    ]


def save_reliability_diagram(
    model_bins: dict[str, pd.DataFrame],
    path: str | Path,
    title: str = "Reliability diagram",
) -> None:
    """Save a confidence-vs-accuracy reliability diagram as SVG."""

    width, height = 760, 560
    left, right, top, bottom = 78, 560, 54, 470
    plot_w = right - left
    plot_h = bottom - top

    def x(value: float) -> float:
        return left + value * plot_w

    def y(value: float) -> float:
        return bottom - value * plot_h

    svg = _svg_header(width, height)
    svg.append(f'<text x="{left}" y="30" class="title">{escape(title)}</text>')

    for tick in np.linspace(0, 1, 6):
        xx = x(float(tick))
        yy = y(float(tick))
        svg.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" class="grid"/>')
        svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/>')
        svg.append(f'<text x="{xx - 8:.1f}" y="{bottom + 22}" class="small">{tick:.1f}</text>')
        svg.append(f'<text x="{left - 42}" y="{yy + 4:.1f}" class="small">{tick:.1f}</text>')

    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{top}" stroke="#9ca3af" stroke-dasharray="6 5"/>')
    svg.append(f'<text x="{left + 150}" y="{height - 22}">Srednia pewnosc modelu</text>')
    svg.append(
        f'<text transform="translate(22 {top + 260}) rotate(-90)">Rzeczywista poprawnosc</text>'
    )

    legend_x = 590
    legend_y = 75
    for idx, (name, bins) in enumerate(model_bins.items()):
        color = COLORS[idx % len(COLORS)]
        points = []
        for _, row in bins.dropna(subset=["accuracy"]).iterrows():
            if int(row["count"]) == 0:
                continue
            points.append((x(float(row["mean_confidence"])), y(float(row["accuracy"]))))

        if points:
            path_points = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            svg.append(
                f'<polyline points="{path_points}" fill="none" stroke="{color}" '
                f'stroke-width="2.2"/>'
            )
            for px, py in points:
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>')

        ly = legend_y + idx * 28
        svg.append(f'<rect x="{legend_x}" y="{ly - 10}" width="16" height="4" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{ly - 4}">{escape(name)}</text>')

    svg.append("</svg>")
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def save_rejection_curve(
    curves: dict[str, pd.DataFrame],
    path: str | Path,
    title: str = "Mechanizm 'nie wiem'",
) -> None:
    """Save coverage-vs-selective-accuracy curves as SVG."""

    width, height = 760, 560
    left, right, top, bottom = 78, 560, 54, 470
    plot_w = right - left
    plot_h = bottom - top

    def x(value: float) -> float:
        return left + value * plot_w

    def y(value: float) -> float:
        return bottom - value * plot_h

    svg = _svg_header(width, height)
    svg.append(f'<text x="{left}" y="30" class="title">{escape(title)}</text>')

    for tick in np.linspace(0, 1, 6):
        xx = x(float(tick))
        yy = y(float(tick))
        svg.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" class="grid"/>')
        svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/>')
        svg.append(f'<text x="{xx - 8:.1f}" y="{bottom + 22}" class="small">{tick:.1f}</text>')
        svg.append(f'<text x="{left - 42}" y="{yy + 4:.1f}" class="small">{tick:.1f}</text>')

    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    svg.append(f'<text x="{left + 185}" y="{height - 22}">Coverage</text>')
    svg.append(
        f'<text transform="translate(22 {top + 285}) rotate(-90)">Accuracy po odrzuceniu</text>'
    )

    legend_x = 590
    legend_y = 75
    for idx, (name, curve) in enumerate(curves.items()):
        color = COLORS[idx % len(COLORS)]
        points = []
        for _, row in curve.dropna(subset=["selective_accuracy"]).iterrows():
            points.append((x(float(row["coverage"])), y(float(row["selective_accuracy"]))))

        if points:
            path_points = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            svg.append(
                f'<polyline points="{path_points}" fill="none" stroke="{color}" '
                f'stroke-width="2.2"/>'
            )
            for px, py in points:
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>')

        ly = legend_y + idx * 28
        svg.append(f'<rect x="{legend_x}" y="{ly - 10}" width="16" height="4" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{ly - 4}">{escape(name)}</text>')

    svg.append("</svg>")
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def save_rejection_threshold_panels(
    curves: dict[str, pd.DataFrame],
    path: str | Path,
    title: str = "Prog odrzucania",
) -> None:
    """Save threshold-vs-accuracy/coverage/rejected panels for the rejection rule."""

    width, height = 1120, 520
    top, bottom = 70, 405
    panel_w = 270
    gap = 58
    lefts = [76, 76 + panel_w + gap, 76 + 2 * (panel_w + gap)]
    right_margin = 120

    all_thresholds = np.concatenate([curve["threshold"].to_numpy() for curve in curves.values()])
    x_min = float(all_thresholds.min())
    x_max = float(all_thresholds.max())
    max_rejected = max(1, max(int(curve["rejected"].max()) for curve in curves.values()))

    svg = _svg_header(width, height)
    svg.append(f'<text x="38" y="34" class="title">{escape(title)}</text>')
    svg.append(
        '<text x="38" y="54" class="small">Im wyzszy prog tau, tym mniej odpowiedzi, '
        'ale zwykle wieksza trafnosc zaakceptowanych predykcji.</text>'
    )

    panel_titles = [
        "Accuracy po odrzuceniu",
        "Coverage",
        "Odrzucone przyklady",
    ]

    def x(value: float, panel_left: float) -> float:
        if x_max == x_min:
            return panel_left
        return panel_left + (value - x_min) / (x_max - x_min) * panel_w

    def y_unit(value: float) -> float:
        return bottom - value * (bottom - top)

    def y_count(value: float) -> float:
        return bottom - (value / max_rejected) * (bottom - top)

    for panel_idx, panel_left in enumerate(lefts):
        svg.append(
            f'<text x="{panel_left}" y="{top - 18}" font-weight="700">'
            f'{escape(panel_titles[panel_idx])}</text>'
        )
        for tick in np.linspace(0, 1, 6):
            yy = y_unit(float(tick))
            svg.append(
                f'<line x1="{panel_left}" y1="{yy:.1f}" x2="{panel_left + panel_w}" '
                f'y2="{yy:.1f}" class="grid"/>'
            )
            label = f"{tick:.1f}" if panel_idx < 2 else str(int(round(tick * max_rejected)))
            svg.append(f'<text x="{panel_left - 38}" y="{yy + 4:.1f}" class="small">{label}</text>')
        for threshold in np.linspace(x_min, x_max, 4):
            xx = x(float(threshold), panel_left)
            svg.append(
                f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" class="grid"/>'
            )
            svg.append(f'<text x="{xx - 12:.1f}" y="{bottom + 22}" class="small">{threshold:.2f}</text>')
        svg.append(
            f'<line x1="{panel_left}" y1="{bottom}" x2="{panel_left + panel_w}" '
            f'y2="{bottom}" class="axis"/>'
        )
        svg.append(
            f'<line x1="{panel_left}" y1="{top}" x2="{panel_left}" y2="{bottom}" class="axis"/>'
        )
        svg.append(f'<text x="{panel_left + 104}" y="{height - 46}">tau</text>')

    legend_x = width - right_margin + 10
    legend_y = 95
    for idx, (name, curve) in enumerate(curves.items()):
        color = COLORS[idx % len(COLORS)]
        for panel_idx, panel_left in enumerate(lefts):
            points = []
            for _, row in curve.iterrows():
                threshold = float(row["threshold"])
                if panel_idx == 0:
                    value = row["selective_accuracy"]
                    if pd.isna(value):
                        continue
                    yy = y_unit(float(value))
                elif panel_idx == 1:
                    yy = y_unit(float(row["coverage"]))
                else:
                    yy = y_count(float(row["rejected"]))
                points.append((x(threshold, panel_left), yy))

            if points:
                path_points = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
                svg.append(
                    f'<polyline points="{path_points}" fill="none" stroke="{color}" '
                    f'stroke-width="2.1"/>'
                )
                for px, py in points:
                    svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{color}"/>')

        short_name = name.split("/", 1)[1] if "/" in name else name
        if len(short_name) > 26:
            short_name = short_name[:23] + "..."
        ly = legend_y + idx * 26
        svg.append(f'<rect x="{legend_x}" y="{ly - 10}" width="16" height="4" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{ly - 4}" class="small">{escape(short_name)}</text>')

    svg.append("</svg>")
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def save_confusion_matrix_grid(
    metrics_df: pd.DataFrame,
    path: str | Path,
    title: str = "Macierze pomylek",
) -> None:
    """Save small TP/TN/FP/FN heatmaps for several model variants."""

    rows = metrics_df.reset_index(drop=True)
    cell = 72
    gap_x = 34
    gap_y = 74
    matrix_w = cell * 2
    matrix_h = cell * 2
    columns = 3
    panels = len(rows)
    panel_w = matrix_w + gap_x
    panel_h = matrix_h + gap_y
    width = max(760, 70 + columns * panel_w)
    grid_rows = int(np.ceil(max(1, panels) / columns))
    height = 118 + grid_rows * panel_h

    svg = _svg_header(width, height)
    svg.append(f'<text x="38" y="32" class="title">{escape(title)}</text>')
    svg.append(
        '<text x="38" y="54" class="small">Wiersze: prawdziwa klasa. '
        'Kolumny: predykcja modelu.</text>'
    )

    max_count = max(1, int(rows[["tn", "fp", "fn", "tp"]].to_numpy().max()))

    def color(value: int) -> str:
        intensity = value / max_count
        # Interpolate from light blue to saturated blue.
        r = int(239 - 202 * intensity)
        g = int(246 - 147 * intensity)
        b = int(255 - 20 * intensity)
        return f"#{r:02x}{g:02x}{b:02x}"

    for idx, row in rows.iterrows():
        col = idx % columns
        grid_row = idx // columns
        x0 = 52 + col * panel_w
        y0 = 124 + grid_row * panel_h
        model_name = str(row["model"])
        short_name = model_name.replace("_tfidf/", "/").replace("logreg_", "logreg/")
        if len(short_name) > 34:
            short_name = short_name[:31] + "..."

        svg.append(f'<text x="{x0}" y="{y0 - 30}" font-weight="700">{escape(short_name)}</text>')
        svg.append(
            f'<text x="{x0}" y="{y0 - 12}" class="small">'
            f'acc={float(row["accuracy"]):.3f}, F1={float(row["f1"]):.3f}</text>'
        )
        svg.append(f'<text x="{x0 + 45}" y="{y0 - 50}" class="small">pred 0</text>')
        svg.append(f'<text x="{x0 + 117}" y="{y0 - 50}" class="small">pred 1</text>')
        svg.append(f'<text x="{x0 - 48}" y="{y0 + 42}" class="small">true 0</text>')
        svg.append(f'<text x="{x0 - 48}" y="{y0 + 114}" class="small">true 1</text>')

        cells = [
            ("TN", int(row["tn"]), 0, 0),
            ("FP", int(row["fp"]), 1, 0),
            ("FN", int(row["fn"]), 0, 1),
            ("TP", int(row["tp"]), 1, 1),
        ]
        for label, value, cx, cy in cells:
            xx = x0 + cx * cell
            yy = y0 + cy * cell
            fill = color(value)
            svg.append(
                f'<rect x="{xx}" y="{yy}" width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="#ffffff" stroke-width="2"/>'
            )
            text_color = "#ffffff" if value / max_count > 0.55 else "#111827"
            svg.append(
                f'<text x="{xx + cell / 2}" y="{yy + 29}" text-anchor="middle" '
                f'font-weight="700" style="fill:{text_color}">{label}</text>'
            )
            svg.append(
                f'<text x="{xx + cell / 2}" y="{yy + 51}" text-anchor="middle" '
                f'style="fill:{text_color}">{value}</text>'
            )

        svg.append(
            f'<rect x="{x0}" y="{y0}" width="{matrix_w}" height="{matrix_h}" '
            'fill="none" stroke="#111827" stroke-width="1.2"/>'
        )

    svg.append("</svg>")
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def save_conformal_plot(
    conformal_df: pd.DataFrame,
    path: str | Path,
    title: str = "Conformal prediction",
) -> None:
    """Save alpha-vs-coverage/singleton/set-size panels for conformal prediction."""

    width, height = 1120, 520
    top, bottom = 70, 405
    panel_w = 270
    gap = 58
    lefts = [76, 76 + panel_w + gap, 76 + 2 * (panel_w + gap)]
    right_margin = 120
    models = conformal_df["model"].drop_duplicates().tolist()
    alphas = conformal_df["alpha"].to_numpy(dtype=float)
    x_min = float(alphas.min())
    x_max = float(alphas.max())
    max_set_size = max(1.0, float(conformal_df["avg_set_size"].max()))

    svg = _svg_header(width, height)
    svg.append(f'<text x="38" y="34" class="title">{escape(title)}</text>')
    svg.append(
        '<text x="38" y="54" class="small">Alpha to dopuszczalne ryzyko bledu. '
        'Nizsze alpha daje wieksze zbiory predykcji.</text>'
    )

    panel_titles = ["Empirical coverage", "Singleton rate", "Average set size"]

    def x(value: float, panel_left: float) -> float:
        if x_max == x_min:
            return panel_left
        return panel_left + (value - x_min) / (x_max - x_min) * panel_w

    def y_unit(value: float) -> float:
        return bottom - value * (bottom - top)

    def y_size(value: float) -> float:
        return bottom - (value / max_set_size) * (bottom - top)

    for panel_idx, panel_left in enumerate(lefts):
        svg.append(
            f'<text x="{panel_left}" y="{top - 18}" font-weight="700">'
            f'{escape(panel_titles[panel_idx])}</text>'
        )
        for tick in np.linspace(0, 1, 6):
            yy = y_unit(float(tick))
            label = f"{tick:.1f}" if panel_idx < 2 else f"{tick * max_set_size:.1f}"
            svg.append(
                f'<line x1="{panel_left}" y1="{yy:.1f}" x2="{panel_left + panel_w}" '
                f'y2="{yy:.1f}" class="grid"/>'
            )
            svg.append(f'<text x="{panel_left - 38}" y="{yy + 4:.1f}" class="small">{label}</text>')
        for alpha in np.linspace(x_min, x_max, 5):
            xx = x(float(alpha), panel_left)
            svg.append(
                f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" class="grid"/>'
            )
            svg.append(f'<text x="{xx - 12:.1f}" y="{bottom + 22}" class="small">{alpha:.2f}</text>')
        svg.append(
            f'<line x1="{panel_left}" y1="{bottom}" x2="{panel_left + panel_w}" '
            f'y2="{bottom}" class="axis"/>'
        )
        svg.append(
            f'<line x1="{panel_left}" y1="{top}" x2="{panel_left}" y2="{bottom}" class="axis"/>'
        )
        svg.append(f'<text x="{panel_left + 100}" y="{height - 46}">alpha</text>')

    legend_x = width - right_margin + 10
    legend_y = 95
    for idx, model in enumerate(models):
        color = COLORS[idx % len(COLORS)]
        sub = conformal_df[conformal_df["model"] == model].sort_values("alpha")
        for panel_idx, panel_left in enumerate(lefts):
            points = []
            for _, row in sub.iterrows():
                alpha = float(row["alpha"])
                if panel_idx == 0:
                    yy = y_unit(float(row["empirical_coverage"]))
                elif panel_idx == 1:
                    yy = y_unit(float(row["singleton_rate"]))
                else:
                    yy = y_size(float(row["avg_set_size"]))
                points.append((x(alpha, panel_left), yy))
            if points:
                path_points = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
                svg.append(
                    f'<polyline points="{path_points}" fill="none" stroke="{color}" '
                    f'stroke-width="2.1"/>'
                )
                for px, py in points:
                    svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{color}"/>')

        short_name = str(model).replace("_tfidf/", "/")
        if len(short_name) > 26:
            short_name = short_name[:23] + "..."
        ly = legend_y + idx * 26
        svg.append(f'<rect x="{legend_x}" y="{ly - 10}" width="16" height="4" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{ly - 4}" class="small">{escape(short_name)}</text>')

    svg.append("</svg>")
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def save_error_category_plot(
    summary_df: pd.DataFrame,
    path: str | Path,
    title: str = "Kategorie bledow wysokiej pewnosci",
) -> None:
    """Save a grouped bar chart of high-confidence error categories."""

    if summary_df.empty:
        Path(path).write_text("\n".join(_svg_header(760, 220) + ["</svg>"]), encoding="utf-8")
        return

    categories = summary_df["category"].drop_duplicates().tolist()
    models = summary_df["model"].drop_duplicates().tolist()
    width = 1160
    row_h = 34
    top = 78
    left = 260
    right = 960
    height = max(300, top + len(categories) * row_h + 80)
    max_count = max(1, int(summary_df["high_confidence_error_count"].max()))

    svg = _svg_header(width, height)
    svg.append(f'<text x="38" y="34" class="title">{escape(title)}</text>')
    svg.append(
        '<text x="38" y="54" class="small">Liczone sa bledne predykcje z confidence >= 0.90.</text>'
    )

    for idx, category in enumerate(categories):
        y = top + idx * row_h
        svg.append(f'<text x="38" y="{y + 16}" class="small">{escape(str(category))}</text>')
        svg.append(f'<line x1="{left}" y1="{y + 20}" x2="{right}" y2="{y + 20}" class="grid"/>')

    for tick in np.linspace(0, max_count, 5):
        x = left + (tick / max_count) * (right - left)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - 56}" class="grid"/>')
        svg.append(f'<text x="{x - 8:.1f}" y="{height - 34}" class="small">{int(round(tick))}</text>')

    bar_h = max(3.5, min(9.0, row_h / (len(models) + 1)))
    for model_idx, model in enumerate(models):
        color = COLORS[model_idx % len(COLORS)]
        sub = summary_df[summary_df["model"] == model]
        counts = {
            row["category"]: int(row["high_confidence_error_count"])
            for _, row in sub.iterrows()
        }
        for cat_idx, category in enumerate(categories):
            count = counts.get(category, 0)
            y = top + cat_idx * row_h + 4 + model_idx * (bar_h + 1.5)
            bar_w = (count / max_count) * (right - left)
            svg.append(
                f'<rect x="{left}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'fill="{color}"/>'
            )
            if count:
                svg.append(
                    f'<text x="{left + bar_w + 5:.1f}" y="{y + bar_h:.1f}" class="small">{count}</text>'
                )

    legend_x = 980
    legend_y = 92
    for idx, model in enumerate(models):
        color = COLORS[idx % len(COLORS)]
        label = str(model).split("/", 1)[1] if "/" in str(model) else str(model)
        if len(label) > 30:
            label = label[:27] + "..."
        ly = legend_y + idx * 24
        svg.append(f'<rect x="{legend_x}" y="{ly - 9}" width="16" height="4" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{ly - 4}" class="small">{escape(label)}</text>')

    svg.append("</svg>")
    Path(path).write_text("\n".join(svg), encoding="utf-8")
