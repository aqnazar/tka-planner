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
figure.fitmap{display:inline-block;margin:.5rem 1.5rem .5rem 0;vertical-align:top;
max-width:100%}
figure.fitmap img{max-width:100%;height:auto;image-rendering:pixelated;
border:1px solid var(--line);background:repeating-conic-gradient(#0000 0 25%,#8881 0 50%)
50%/16px 16px}
figure.fitmap figcaption{font-size:.8rem;color:var(--muted);max-width:26rem}
.legend{max-width:30rem;margin:.5rem 0 1rem}
.legend .bar{height:.8rem;border-radius:3px;border:1px solid var(--line)}
.legend .ends{display:flex;justify-content:space-between;font-size:.75rem;
color:var(--muted)}
.sev-error{color:var(--warn);font-weight:700}
.sev-warning{color:var(--estimated);font-weight:600}
.sev-info{color:var(--muted)}
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
        # A landmark measured outside the CT is ``measured`` like a CT pick, but its
        # uncertainty is not; name the source and residual next to the value.
        for landmark_id, record in metric.external_sources.items():
            detail = [str(record["source"])]
            if "rms_residual_mm" in record:
                detail.append(f'RMS {record["rms_residual_mm"]:g} mm')
            if "centre_sd_mm" in record:
                detail.append(f'predicted SD {record["centre_sd_mm"]:g} mm')
            if "registration" in record:
                detail.append(f'registered by {record["registration"]}')
            notes.append(
                f'<div class="note"><strong>Source:</strong> '
                f'<code>{_escape(landmark_id)}</code> {_escape(", ".join(detail))}</div>'
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


def _mm(value) -> str:
    return "&mdash;" if value is None else f"{value:.1f}"


def _resection_table_section(table: dict) -> list:
    from ..core.resection_table import ROW_LABELS, angle_text

    parts = ["<h2>Resection table</h2>",
             '<p class="note">Bone removed by each cut, in millimetres perpendicular to '
             "the cut, measured on this patient's bone against the planned planes. "
             "The definition of each row is beside it.</p>",
             '<div class="overflow"><table><thead><tr><th>Cut</th><th>Medial</th>'
             "<th>Lateral</th><th>Angle (deg)</th><th>Measured as</th></tr></thead>"
             "<tbody>"]
    for row in table.get("rows", []):
        definition = row.get("definition", "")
        if row.get("highest_mm"):
            highest = row["highest_mm"]
            definition += (f" Highest bone above the cut: medial "
                           f"{highest['medial']:.1f}, lateral {highest['lateral']:.1f} "
                           f"mm ({row.get('highest_definition', '')})")
        parts.append(
            f"<tr><td><strong>{_escape(ROW_LABELS.get(row['cut'], row['cut']))}"
            f"</strong></td>"
            f'<td class="num">{_mm(row["medial_mm"])}</td>'
            f'<td class="num">{_mm(row["lateral_mm"])}</td>'
            f'<td class="note">{_escape(angle_text(row.get("angle_deg", {})))}</td>'
            f'<td class="note">{_escape(definition)}</td></tr>'
        )
    parts.append("</tbody></table></div>")
    for note in table.get("notes", []):
        parts.append(f'<div class="note">{_escape(note)}</div>')
    return parts


def _register_section(diagnostics: dict) -> list:
    rows = [
        ("Femoral over tibial component: rotation", "component_rotation_mismatch_deg",
         "deg"),
        ("Femoral over tibial component: anterior offset",
         "component_offset_anterior_mm", "mm"),
        ("Femoral over tibial component: lateral offset",
         "component_offset_lateral_mm", "mm"),
        ("Tray rotation off the tibial axis", "tray_rotation_from_tibial_axis_deg",
         "deg"),
        ("Tray offset from the cut centre, anterior",
         "tray_offset_from_tibial_cut_centre_anterior_mm", "mm"),
        ("Tray offset from the cut centre, lateral",
         "tray_offset_from_tibial_cut_centre_lateral_mm", "mm"),
        ("Insert thickness", "insert_thickness_mm", "mm"),
        ("Extension gap, medial", "extension_gap_medial_mm", "mm"),
        ("Extension gap, lateral", "extension_gap_lateral_mm", "mm"),
    ]
    present = [(label, diagnostics[key], unit) for label, key, unit in rows
               if diagnostics.get(key) is not None]
    if not present:
        return []
    parts = ["<h2>Component register</h2>", '<dl class="kv">']
    for label, value, unit in present:
        parts.append(f"<dt>{_escape(label)}</dt><dd>{float(value):+.1f} {unit}</dd>")
    parts.append("</dl>")
    return parts


def _implant_section(spec: dict, fit: dict | None) -> list:
    femoral = spec.get("femoral_component", {}).get("dimensions_mm", {})
    tibial = spec.get("tibial_component", {}).get("dimensions_mm", {})
    rows = [
        ("Femoral component ML", femoral.get("ml")),
        ("Femoral component AP (overall)", femoral.get("ap_overall")),
        ("Femoral flange height", femoral.get("flange_height")),
        ("Femoral distal thickness", femoral.get("distal_thickness")),
        ("Tray ML", tibial.get("ml")),
        ("Tray AP", tibial.get("ap")),
        ("Tray thickness", tibial.get("tray_thickness")),
        ("Insert thickness", spec.get("insert", {}).get("thickness_mm")),
    ]
    parts = ["<h2>Implant</h2>",
             '<p class="note">The patient-specific implant, measured on this plan\'s '
             "cuts: what the CAD model is given to build.</p>", '<dl class="kv">']
    for label, value in rows:
        if value is not None:
            parts.append(f"<dt>{_escape(label)}</dt><dd>{float(value):.1f} mm</dd>")
    equivalent = spec.get("catalogue_equivalent", {})
    if equivalent:
        parts.append(f"<dt>Nearest catalogue size</dt>"
                     f"<dd>{_escape(equivalent.get('nearest_size'))} (parameter "
                     f"{equivalent.get('size_parameter')})</dd>")
    for name, record in ((fit or {}).get("library_sizes") or {}).items():
        parts.append(f"<dt>Library part, {_escape(name.replace('_', ' '))}</dt>"
                     f"<dd>{_escape(record.get('source_size'))} at scale "
                     f"{_escape(record.get('scale_xyz') or record.get('scale'))}</dd>")
    parts.append("</dl>")
    return parts


def _fit_section(fit: dict, grids: dict) -> list:
    from .figures import fit_map_figure, legend_bar

    parts = ["<h2>Fit on the cuts</h2>",
             f'<p class="note">Implant: {_escape(fit.get("implant_mode", ""))}. Each '
             "component's footprint just off its cut is compared with the bone's "
             "section on the cut. Overhang is component beyond bone; underhang is bone "
             "left uncovered; each is the largest distance to the other outline.</p>",
             '<div class="overflow"><table><thead><tr><th>Component</th>'
             "<th>Zone</th><th>Coverage</th><th>Overhang max</th>"
             "<th>Underhang max</th></tr></thead><tbody>"]
    names = [n for n in ("tibial_component", "femoral_component") if n in fit]
    for name in names:
        record = fit[name]
        parts.append(
            f"<tr><td><strong>{_escape(name.replace('_', ' '))}</strong></td>"
            f"<td>whole cut</td>"
            f'<td class="num">{100 * record["coverage_fraction"]:.1f}%</td>'
            f'<td class="num">{record["max_overhang_mm"]:.1f} mm</td>'
            f'<td class="num">{record["max_underhang_mm"]:.1f} mm</td></tr>'
        )
        for zone, values in record.get("zones", {}).items():
            flagged = f"OVERHANG_{zone.upper()}" in record.get("flags", [])
            parts.append(
                f"<tr><td></td><td>{_escape(zone)}</td><td></td>"
                f'<td class="num{" flag" if flagged else ""}">'
                f'{values["max_overhang_mm"]:.1f} mm</td>'
                f'<td class="num">{values["max_underhang_mm"]:.1f} mm</td></tr>'
            )
    parts.append("</tbody></table></div>")
    insert = fit.get("insert") or {}
    if insert.get("note"):
        parts.append(f'<div class="note">Insert: {_escape(insert["note"])}</div>')
    if fit.get("missing"):
        parts.append(f'<div class="note">Not in the library: '
                     f'{_escape(", ".join(fit["missing"]))}</div>')
    drawn = [name for name in names if grids.get(name) is not None]
    if drawn:
        parts.append("<h3>Fit map</h3>")
        parts.append(legend_bar())
        for name in drawn:
            # Drawn lateral-right on both knees, so it is a mirror image of the true
            # view on one of them; the caption says which way is which.
            title = ("Tibial cut" if name.startswith("tibial")
                     else "Distal femoral cut")
            parts.append(fit_map_figure(grids[name], title=title, record=fit[name]))
    return parts


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
    resection_table: dict | None = None,
    implant_spec: dict | None = None,
    fit: dict | None = None,
    fit_grids: dict | None = None,
    findings: list | None = None,
) -> str:
    """Render a complete plan report as one self-contained HTML document.

    The later arguments carry what a comparison of plans needs beside the anatomy: the
    resection table, the implant's dimensions, the fit on each cut with its colour map,
    the components' register, and every finding. ``findings`` (from
    :func:`tka_planner.pipeline.findings_case`) supersedes ``qc_findings`` when given,
    because it holds the scan's checks and the plan's together.
    """
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

    if resection_table is not None:
        parts.extend(_resection_table_section(resection_table))
    if surgical is not None:
        parts.extend(_register_section(surgical.diagnostics))

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

    if implant_spec is not None:
        parts.extend(_implant_section(implant_spec, fit))
    if fit is not None:
        parts.extend(_fit_section(fit, fit_grids or {}))

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
    if findings is not None:
        parts.append("<h2>Quality control and findings</h2>")
        parts.append('<p class="note">The scan\'s checks, then what the plan, the '
                     "sizing, the femoral box and the fit found. A location is in scan "
                     "millimetres; the application turns to it.</p>")
        parts.append('<div class="overflow"><table><thead><tr>'
                     "<th>Severity</th><th>Check</th><th>Source</th><th>Finding</th>"
                     "<th>Where</th></tr></thead><tbody>")
        for finding in findings:
            where = finding.get("location")
            where_text = ("&mdash;" if not where else
                          _escape(", ".join(f"{v:.1f}" for v in where["point_mm"])
                                  + f" ({where['set']})"))
            parts.append(
                f'<tr><td class="sev-{_escape(finding["severity"])}">'
                f'{_escape(finding["severity"])}</td>'
                f'<td><code>{_escape(finding["code"])}</code></td>'
                f'<td>{_escape(finding.get("source", ""))}</td>'
                f'<td class="note">{_escape(finding["message"])}</td>'
                f'<td class="note">{where_text}</td></tr>'
            )
        parts.append("</tbody></table></div>")
    elif qc_findings:
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
