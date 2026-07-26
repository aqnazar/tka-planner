"""A single-file HTML plan report.

No JavaScript, no external stylesheet, no fonts, no CDN. Everything is inline, so the
file opens from a memory stick in ten years' time and looks the same. That matters for a
document intended as supplementary material to a paper.

The report's organising idea is the provenance column. Every number is shown next to how
it was obtained -- measured from the patient's anatomy, estimated through a named
assumption, or not computable at all -- and the assumptions are printed in full rather
than referenced. A reader should be able to see, without opening any code, which parts
of a plan are anatomy and which are modelling.
"""

from __future__ import annotations

import datetime as _dt
import html
from pathlib import Path

from ..core.provenance import Quality

__all__ = ["render_report", "write_report"]


_CSS = """
:root{--ink:#16181d;--muted:#5b6472;--line:#dfe3e8;--bg:#fff;--panel:#f7f8fa;
--measured:#1a7f5a;--estimated:#8a5a00;--missing:#6b7280;
--measured-bg:#e7f4ee;--estimated-bg:#fdf3e0;--missing-bg:#f1f2f4;--warn:#a33]
}
*{box-sizing:border-box}
body{margin:0;padding:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:60rem;margin:0 auto;padding:2.5rem 1.25rem 4rem}
h1{font-size:1.6rem;margin:0 0 .25rem}
h2{font-size:1.05rem;margin:2.25rem 0 .75rem;padding-bottom:.35rem;
border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 1.5rem}
.banner{background:#fff4f4;border:1px solid #f0c9c9;color:var(--warn);
padding:.7rem .9rem;border-radius:6px;margin:0 0 1.75rem;font-size:.9rem}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;padding:.5rem .6rem;border-bottom:1px solid var(--line);
vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:.78rem;text-transform:uppercase;
letter-spacing:.04em}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tag{display:inline-block;padding:.08rem .45rem;border-radius:999px;
font-size:.74rem;font-weight:600;white-space:nowrap}
.measured{background:var(--measured-bg);color:var(--measured)}
.estimated{background:var(--estimated-bg);color:var(--estimated)}
.not_computable{background:var(--missing-bg);color:var(--missing)}
.note{color:var(--muted);font-size:.82rem;margin-top:.2rem}
.flag{color:var(--warn);font-weight:600}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;
padding:.85rem 1rem;margin:.75rem 0;font-size:.87rem}
.panel h3{margin:0 0 .3rem;font-size:.9rem}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.85em}
.kv{display:grid;grid-template-columns:12rem 1fr;gap:.2rem .9rem;font-size:.88rem}
.kv dt{color:var(--muted)}
.kv dd{margin:0;font-variant-numeric:tabular-nums}
.overflow{overflow-x:auto}
@media (prefers-color-scheme:dark){
:root{--ink:#e6e8ec;--muted:#9aa3b2;--line:#2b3038;--bg:#14161a;--panel:#1b1e24;
--measured:#6ddaa8;--estimated:#e5b661;--missing:#9aa3b2;
--measured-bg:#123026;--estimated-bg:#33270f;--missing-bg:#23262c;--warn:#ff9b9b}
.banner{background:#2a1a1a;border-color:#4a2c2c}
}
"""


def _tag(quality: Quality) -> str:
    label = {"measured": "measured", "estimated": "estimated",
             "not_computable": "not computable"}[quality.value]
    return f'<span class="tag {quality.value}">{label}</span>'


def _escape(text) -> str:
    return html.escape(str(text))


def _metric_rows(metrics: dict) -> str:
    rows = []
    for name, metric in metrics.items():
        label = _escape(name.replace("_deg", "").replace("_", " "))

        if metric.value is None:
            value_cell = '<td class="num">&mdash;</td>'
        else:
            flag = ""
            if metric.is_outside_reference_range:
                low, high = metric.reference_range
                flag = (f'<div class="flag">outside normal '
                        f'{low:g}&ndash;{high:g}</div>')
            value_cell = (f'<td class="num">{metric.value:g} '
                          f'{_escape(metric.unit)}{flag}</td>')

        notes = [f"<div>{_escape(metric.definition)}</div>"]
        if metric.sign_convention:
            notes.append(f'<div class="note">{_escape(metric.sign_convention)}</div>')
        if metric.assumptions:
            for assumption in metric.assumptions:
                notes.append(
                    f'<div class="note"><strong>Assumes:</strong> '
                    f'{_escape(assumption.description)}</div>'
                )
        if metric.missing:
            notes.append(
                f'<div class="note"><strong>Missing:</strong> '
                f'<code>{_escape(", ".join(metric.missing))}</code></div>'
            )
        if metric.reason:
            notes.append(f'<div class="note">{_escape(metric.reason)}</div>')
        if metric.would_require:
            notes.append(
                f'<div class="note"><strong>Would require:</strong> '
                f'{_escape(metric.would_require)}</div>'
            )

        rows.append(
            f"<tr><td><strong>{label}</strong></td>{value_cell}"
            f"<td>{_tag(metric.quality)}</td><td>{''.join(notes)}</td></tr>"
        )
    return "\n".join(rows)


def render_report(
    *,
    case_id: str,
    side: str,
    metrics: dict,
    femoral_frame=None,
    tibial_frame=None,
    sizing=None,
    comparison_sizing=None,
    surgical=None,
    measurements: dict | None = None,
    coverage: dict | None = None,
    qc_findings: list | None = None,
    inputs: list | None = None,
    landmark_summary: dict | None = None,
    tool_version: str = "0.1.0",
) -> str:
    """Render a complete plan report as one self-contained HTML document."""
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    counts = {tier: 0 for tier in ("measured", "estimated", "not_computable")}
    for metric in metrics.values():
        counts[metric.quality.value] += 1

    parts = [
        "<div class=\"wrap\">",
        f"<h1>TKA plan &mdash; {_escape(case_id)} ({_escape(side)})</h1>",
        f'<p class="sub">Generated {generated} by tka-planner '
        f'{_escape(tool_version)}</p>',
        '<div class="banner"><strong>Not a medical device.</strong> Research and '
        'demonstration only. This plan has not been validated for clinical use and '
        'must not inform the care of any patient.</div>',
    ]

    # -- Summary -------------------------------------------------------
    parts.append("<h2>What this plan can and cannot say</h2>")
    parts.append(
        f'<p>Of {len(metrics)} metrics, '
        f'<strong>{counts["measured"]}</strong> were measured from this patient\'s '
        f'anatomy, <strong>{counts["estimated"]}</strong> required an assumption, and '
        f'<strong>{counts["not_computable"]}</strong> could not be computed at all. '
        f'Every value below states which it is.</p>'
    )

    if coverage:
        parts.append('<div class="panel"><h3>Scan coverage</h3><dl class="kv">')
        for bone, record in coverage.items():
            for key, value in record.items():
                if key == "detection":
                    continue
                parts.append(
                    f"<dt>{_escape(bone)} {_escape(key.replace('_', ' '))}</dt>"
                    f"<dd>{_escape(value)}</dd>"
                )
        parts.append("</dl>")
        for bone, record in coverage.items():
            if record.get("detection"):
                parts.append(
                    f'<div class="note">{_escape(bone)}: '
                    f'{_escape(record["detection"])}</div>'
                )
        parts.append("</div>")

    # -- Metrics -------------------------------------------------------
    parts.append("<h2>Deformity measurements</h2>")
    parts.append('<div class="overflow"><table><thead><tr>'
                 "<th>Metric</th><th>Value</th><th>Provenance</th>"
                 "<th>Definition and basis</th></tr></thead><tbody>")
    parts.append(_metric_rows(metrics))
    parts.append("</tbody></table></div>")

    # -- Frames --------------------------------------------------------
    if femoral_frame is not None or tibial_frame is not None:
        parts.append("<h2>Anatomical frames</h2>")
        parts.append('<div class="overflow"><table><thead><tr>'
                     "<th>Bone</th><th>Axis method</th><th>Provenance</th>"
                     "<th>Diagnostics</th></tr></thead><tbody>")
        for frame in (femoral_frame, tibial_frame):
            if frame is None:
                continue
            diagnostics = ", ".join(
                f"{k.replace('_', ' ')}: {v}" for k, v in frame.diagnostics.items()
            )
            parts.append(
                f"<tr><td><strong>{_escape(frame.bone)}</strong></td>"
                f"<td><code>{_escape(frame.method)}</code></td>"
                f"<td>{_tag(frame.quality)}</td>"
                f'<td class="note">{_escape(diagnostics)}</td></tr>'
            )
        parts.append("</tbody></table></div>")

    # -- Surgical plan -------------------------------------------------
    if surgical is not None:
        parts.append("<h2>Correction and resections</h2>")
        parts.append('<dl class="kv">')
        parts.append(
            f"<dt>Distal femoral valgus cut</dt>"
            f"<dd>{surgical.distal_femoral_valgus_cut_deg:.1f}&deg; "
            f"{_tag(surgical.valgus_quality)}<div class=\"note\">"
            f"{_escape(surgical.valgus_source)}</div></dd>"
        )
        parts.append(
            f"<dt>Tibial posterior slope</dt>"
            f"<dd>{surgical.tibial_slope_deg:.1f}&deg;</dd>"
        )
        parts.append(
            f"<dt>Alignment philosophy</dt>"
            f"<dd>{_escape(surgical.philosophy)}</dd>"
        )
        parts.append("</dl>")

        parts.append('<div class="overflow"><table><thead><tr>'
                     "<th>Resection</th><th>Medial</th><th>Lateral</th>"
                     "<th>Reference</th></tr></thead><tbody>")
        for name, resection in surgical.resections.items():
            parts.append(
                f"<tr><td><strong>"
                f"{_escape(name.replace('_', ' '))}</strong></td>"
                f'<td class="num">{resection.medial_depth_mm:.1f} mm</td>'
                f'<td class="num">{resection.lateral_depth_mm:.1f} mm</td>'
                f'<td class="note">{_escape(resection.reference)}</td></tr>'
            )
        parts.append("</tbody></table></div>")

        for warning in surgical.warnings:
            parts.append(f'<div class="panel flag">{_escape(warning)}</div>')

    # -- Sizing --------------------------------------------------------
    if sizing is not None:
        parts.append("<h2>Implant sizing</h2>")
        parts.append('<dl class="kv">')
        rows = [
            ("Method", sizing.method),
            ("Size parameter", f"{sizing.size_parameter:.3f}"),
            ("Implant width (ML)", f"{sizing.implant_ml_mm:.1f} mm"),
            ("Implant depth (AP)", f"{sizing.implant_ap_mm:.1f} mm"),
            ("Measured femoral ML", f"{sizing.measured_ml_mm:.1f} mm"),
            ("Component thickness", f"{sizing.femoral_thickness_mm:.2f} mm"),
            ("Nearest discrete size", sizing.nearest_discrete_size),
        ]
        if sizing.ap_mismatch_mm is not None:
            rows.append(("AP aspect residual", f"{sizing.ap_mismatch_mm:+.1f} mm"))
        for key, value in rows:
            parts.append(f"<dt>{_escape(key)}</dt><dd>{_escape(value)}</dd>")
        parts.append("</dl>")
        parts.append(f'<div class="note">{_escape(sizing.rationale)}</div>')

        if measurements:
            parts.append(
                '<div class="panel"><h3>How the dimensions were measured</h3>'
            )
            for bone, record in measurements.items():
                parts.append(
                    f"<div><strong>{_escape(bone)}</strong>: "
                    f"ML {record['ml_mm']:.1f} mm, AP {record['ap_mm']:.1f} mm "
                    f"<code>{_escape(record['method'])}</code></div>"
                )
            naive = measurements.get("tibia", {}).get(
                "diagnostics", {}
            ).get("naive_proximal_bbox_ap_mm")
            if naive is not None:
                parts.append(
                    f'<div class="note">The tibial anteroposterior dimension is taken '
                    f'from the resection cross-section, not a bounding box. A bounding '
                    f'box of the proximal tibia would read {naive:.1f} mm here, '
                    f'inflated by the intercondylar eminence and the tubercle.</div>'
                )
            parts.append("</div>")

        if comparison_sizing is not None:
            parts.append(
                '<div class="panel"><h3>Against the traditional discrete chart</h3>'
                f'<div>{_escape(comparison_sizing.rationale)}</div>'
            )
            if comparison_sizing.flags:
                parts.append(
                    f'<div class="flag">Flags: '
                    f'{_escape(", ".join(comparison_sizing.flags))}</div>'
                )
            parts.append("</div>")

    # -- Landmarks -----------------------------------------------------
    if landmark_summary:
        parts.append("<h2>Landmarks</h2>")
        parts.append('<div class="overflow"><table><thead><tr>'
                     "<th>Status</th><th>Count</th><th>Meaning</th>"
                     "</tr></thead><tbody>")
        meanings = {
            "present": "picked by a human on the anatomy",
            "estimated": "positioned automatically; awaiting review",
            "derived": "computed from other landmarks",
            "out_of_scan": "the anatomy lies outside the imaged volume",
            "not_picked": "not yet done",
            "rejected": "picked, then failed a quality check",
        }
        for status, count in sorted(landmark_summary.items()):
            parts.append(
                f"<tr><td>{_escape(status)}</td>"
                f'<td class="num">{count}</td>'
                f'<td class="note">{_escape(meanings.get(status, ""))}</td></tr>'
            )
        parts.append("</tbody></table></div>")

    # -- Quality control -----------------------------------------------
    if qc_findings:
        parts.append("<h2>Quality control</h2>")
        parts.append('<div class="overflow"><table><thead><tr>'
                     "<th>Check</th><th>Severity</th><th>Finding</th>"
                     "</tr></thead><tbody>")
        for finding in qc_findings:
            parts.append(
                f'<tr><td><code>{_escape(finding["code"])}</code></td>'
                f'<td>{_escape(finding["severity"])}</td>'
                f'<td class="note">{_escape(finding["message"])}</td></tr>'
            )
        parts.append("</tbody></table></div>")

    # -- Inputs --------------------------------------------------------
    if inputs:
        parts.append("<h2>Inputs</h2>")
        parts.append(
            '<p class="note">Content hashes, so this plan can be tied to exactly the '
            "files it was computed from.</p>"
        )
        parts.append('<div class="overflow"><table><thead><tr>'
                     "<th>Role</th><th>File</th><th>SHA-256</th>"
                     "</tr></thead><tbody>")
        for record in inputs:
            digest = record.get("sha256") or ""
            parts.append(
                f'<tr><td>{_escape(record.get("role", ""))}</td>'
                f'<td><code>{_escape(record.get("name", ""))}</code></td>'
                f'<td><code>{_escape(digest[:16])}&hellip;</code></td></tr>'
            )
        parts.append("</tbody></table></div>")

    parts.append("</div>")

    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>TKA plan — {_escape(case_id)} ({_escape(side)})</title>"
        f"<style>{_CSS}</style></head><body>"
        + "\n".join(parts)
        + "</body></html>\n"
    )


def write_report(document: str, path: "str | Path") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
