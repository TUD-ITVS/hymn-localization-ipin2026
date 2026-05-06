"""
Plotting for the HYMN positioning evaluation.

Produces:
 - ecdf_faceted.pdf       : 4 subplots (BLE/UWB/WiFi/Fused), methods overlaid
 - ecdf_fused.pdf         : Fused-only ECDF, all methods
 - spatial_error_<tech>_<method>.pdf : per-point mean error on 2D layout
 - boxplot_per_method.pdf : overview box/violin
 - anchors_overview.pdf   : anchor & reference geometry sanity figure
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from hymn.config import EVAL, FIG_DIR
from hymn.evaluation import data_interface as di


def _setup_style(cfg):
    mpl.rcParams.update({
        "font.family": cfg["plot"]["font_family"],
        "font.size": cfg["plot"]["font_size"],
        "axes.grid": True,
        "grid.alpha": cfg["plot"]["grid_alpha"],
        "grid.linewidth": 0.4,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.dpi": cfg["plot"]["dpi"],
        "savefig.bbox": "tight",
    })


def _ecdf(errors):
    e = np.sort(np.asarray(errors, dtype=float))
    y = np.arange(1, len(e) + 1) / len(e)
    return e, y


def _hull_mask(xs, ys, gx, gy, pad=0.5):
    """Boolean mask of grid points inside the (slightly padded) convex hull
    of the measurement points. Prevents cubic extrapolation into sparse areas.
    Ported from the ICRA ranging paper.
    """
    from matplotlib.path import Path as MplPath
    from scipy.spatial import ConvexHull

    pts = np.column_stack([xs, ys])
    if len(pts) < 3:
        return np.ones(gx.shape, dtype=bool)
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    centroid = hull_pts.mean(axis=0)
    radii = np.linalg.norm(hull_pts - centroid, axis=1, keepdims=True)
    radii[radii == 0] = 1.0
    expanded = centroid + (hull_pts - centroid) * (1 + pad / radii)
    path = MplPath(np.vstack([expanded, expanded[0]]))
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    return path.contains_points(grid_pts).reshape(gx.shape)


def _method_style(method, cfg):
    c = cfg["plot"]["method_colors"].get(method, "black")
    ls = cfg["plot"]["method_linestyles"].get(method, "-")
    return c, ls


_EXTRA_METHOD_COLORS = ["#CC79A7", "#56B4E9", "#F0E442", "#999999"]


def _order_methods(available, cfg):
    """Methods known in cfg first (in cfg order), then remaining methods in the
    data appended alphabetically. Ensures ResNet variants whose names differ
    from the config defaults still appear in plots.
    """
    available = list(available)
    known = [m for m in cfg["plot"]["method_colors"].keys() if m in available]
    extra = sorted(m for m in available if m not in cfg["plot"]["method_colors"])
    return known + extra


def _method_palette(method_order, cfg):
    """Palette dict for seaborn, falling back to a small rotation for methods
    not in cfg['plot']['method_colors']."""
    pal = {}
    extra_idx = 0
    for m in method_order:
        if m in cfg["plot"]["method_colors"]:
            pal[m] = cfg["plot"]["method_colors"][m]
        else:
            pal[m] = _EXTRA_METHOD_COLORS[extra_idx % len(_EXTRA_METHOD_COLORS)]
            extra_idx += 1
    return pal


def plot_ecdf_fused(long_df, cfg=EVAL):
    """Single-panel ECDF: all methods overlaid on the fused technology."""
    _setup_style(cfg)
    sub = long_df[(long_df["technology"] == "fused") & long_df["error"].notna()]
    if len(sub) == 0:
        print("[plot_ecdf_fused] no fused-technology rows; skipping")
        return None
    method_order = _order_methods(sub["method"].unique(), cfg)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for method in method_order:
        vals = sub[sub["method"] == method]["error"].values
        e, y = _ecdf(vals)
        c, ls = _method_style(method, cfg)
        ax.plot(e, y, color=c, linestyle=ls, linewidth=1.8, label=method)
        ax.plot(np.median(vals), 0.5, marker="o", markersize=4.0,
                color=c, markeredgecolor="black", markeredgewidth=0.4)
    ax.axhline(0.5, color="gray", lw=0.4, ls=":")
    ax.axhline(0.95, color="gray", lw=0.4, ls=":")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Position error [m]")
    ax.set_ylabel("ECDF")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "ecdf_fused.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_spatial_error(long_df, cfg=EVAL):
    """
    One figure per (tech, method): scatter at reference points colored by
    per-point mean error, anchors as X markers.
    """
    _setup_style(cfg)
    paths = []
    techs_in = [t for t in cfg["technologies"] if t in long_df["technology"].unique()]
    for tech in techs_in:
        if tech == "fused":
            ids, xy, _ = di.get_anchor_xy("fused")
            anchor_markers = {
                "BLE": (xy[:5], "s"),
                "UWB": (xy[5:15], "^"),
                "WiFi": (xy[15:21], "D"),
            }
        else:
            _, xy, _ = di.get_anchor_xy(tech)
            anchor_markers = {tech.upper(): (xy, "^")}

        for method in long_df[long_df["technology"] == tech]["method"].unique():
            sub = long_df[(long_df["technology"] == tech) & (long_df["method"] == method)]
            sub = sub.dropna(subset=["error", "ref_x", "ref_y"])
            if len(sub) == 0:
                continue
            per_pt = sub.groupby(["ref_x", "ref_y"])["error"].median().reset_index()

            fig, ax = plt.subplots(figsize=(4.5, 4.0))
            sc = ax.scatter(per_pt["ref_x"], per_pt["ref_y"],
                            c=per_pt["error"], cmap="RdYlGn_r", s=50,
                            edgecolor="black", linewidth=0.4)
            for label, (a, m) in anchor_markers.items():
                ax.scatter(a[:, 0], a[:, 1], marker=m, s=80, facecolor="none",
                           edgecolor="black", linewidth=1.0, label=f"{label} anchor")
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Median error [m]")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{tech.upper() if tech != 'fused' else 'Fused'} — {method}")
            ax.legend(loc="best", fontsize=6, frameon=True)
            fig.tight_layout()
            fname = f"spatial_error_{tech}_{method.replace('/', '-')}.pdf"
            path = os.path.join(FIG_DIR, fname)
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


def _plot_methods_variant(sub, method_order, kind, out_path, cfg, ylim=None):
    """Render one distribution-style variant of the per-method comparison.

    ``kind`` is one of: 'violin_box', 'box_strip', 'violin_strip', 'boxen', 'box'.
    """
    import seaborn as sns
    palette = _method_palette(method_order, cfg)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    common = dict(data=sub, x="method", y="error", order=method_order,
                  hue="method", hue_order=method_order, palette=palette,
                  legend=False, ax=ax)
    if kind == "violin_box":
        sns.violinplot(inner="box", cut=0, linewidth=0.8, **common)
    elif kind == "box_strip":
        sns.boxplot(showfliers=False, linewidth=0.8, **common)
        sns.stripplot(data=sub, x="method", y="error", order=method_order,
                      color="black", size=1.2, alpha=0.25, jitter=0.25, ax=ax)
    elif kind == "violin_strip":
        sns.violinplot(inner=None, cut=0, linewidth=0.8, **common)
        sns.stripplot(data=sub, x="method", y="error", order=method_order,
                      color="black", size=1.2, alpha=0.3, jitter=0.25, ax=ax)
    elif kind == "boxen":
        sns.boxenplot(linewidth=0.6, **common)
    elif kind == "box":
        sns.boxplot(showfliers=False, linewidth=0.8, **common)
    else:
        raise ValueError(f"unknown boxplot variant: {kind}")

    ax.set_xlabel("")
    ax.set_ylabel("Position error [m]")
    ax.set_xticks(np.arange(len(method_order)))
    ax.set_xticklabels(method_order, rotation=15, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_boxplot_methods(long_df, cfg=EVAL):
    """Per-method distribution comparison on the fused technology.

    Renders as a violin with an inner IQR box. The y-axis is clipped to the
    overall 95th-percentile error so the violin bodies remain readable
    despite the heavy ILS tail.
    """
    _setup_style(cfg)
    sub = long_df[(long_df["technology"] == "fused") & long_df["error"].notna()].copy()
    if len(sub) == 0:
        print("[plot_boxplot_methods] no fused-technology rows; skipping")
        return []
    method_order = _order_methods(sub["method"].unique(), cfg)

    y_cap = float(np.nanpercentile(sub["error"].values, 95.0)) * 1.05
    ylim = (0, y_cap)

    out = os.path.join(FIG_DIR, "boxplot_per_method.pdf")
    _plot_methods_variant(sub, method_order, "violin_box", out, cfg, ylim=ylim)
    return [out]


def plot_anchors_overview(cfg=EVAL):
    _setup_style(cfg)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    for tech, marker, color in [("ble", "s", "#CC79A7"), ("uwb", "^", "#009E73"), ("wifi", "D", "#E69F00")]:
        _, xy, _ = di.get_anchor_xy(tech)
        ax.scatter(xy[:, 0], xy[:, 1], marker=marker, s=60, facecolor="none",
                   edgecolor=color, linewidth=1.2, label=f"{tech.upper()} anchor (n={len(xy)})")
    from hymn.io import get_reference_positions
    for tech, color in [("ble", "#CC79A7"), ("uwb", "#009E73"), ("wifi", "#E69F00")]:
        ref = get_reference_positions(tech)
        rx = [v[0] for v in ref.values()]
        ry = [v[1] for v in ref.values()]
        ax.scatter(rx, ry, marker=".", s=10, color=color, alpha=0.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("HYMN anchor & reference-point geometry")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "anchors_overview.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ecdf_ranging(rng_df, techs=("ble", "uwb", "wifi"), cfg=EVAL):
    """ECDF of absolute ranging residuals, one curve per technology."""
    _setup_style(cfg)
    tech_colors = {"ble": "#CC79A7", "uwb": "#009E73", "wifi": "#E69F00"}
    tech_linestyles = {"ble": "-", "uwb": "--", "wifi": "-."}

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for tech in techs:
        vals = rng_df[(rng_df["technology"] == tech)]["abs_residual"].dropna().values
        if len(vals) == 0:
            continue
        e, y = _ecdf(vals)
        ax.plot(e, y, color=tech_colors[tech], linestyle=tech_linestyles[tech],
                linewidth=1.8, label=tech.upper())
        ax.plot(np.median(vals), 0.5, marker="o", markersize=4.0,
                color=tech_colors[tech], markeredgecolor="black", markeredgewidth=0.4)
    ax.axhline(0.5, color="gray", lw=0.4, ls=":")
    ax.axhline(0.95, color="gray", lw=0.4, ls=":")
    xmax = cfg["plot"].get("ranging_ecdf_xmax", 20.0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Absolute ranging error [m]")
    ax.set_ylabel("ECDF")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "ecdf_ranging.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def _compute_point_errors(long_df, methods, technology="fused"):
    """For each method, aggregate per reference point (median error).

    Returns dict method -> DataFrame(ref_x, ref_y, error) with NaN rows dropped.
    Methods without any usable (ref_x, ref_y, error) rows are skipped.
    """
    out = {}
    for m in methods:
        sub = long_df[(long_df["technology"] == technology) & (long_df["method"] == m)]
        sub = sub.dropna(subset=["error", "ref_x", "ref_y"])
        if len(sub) == 0:
            continue
        agg = sub.groupby(["ref_x", "ref_y"], as_index=False)["error"].median()
        if len(agg) >= 4:
            out[m] = agg
    return out


def _draw_spatial_panel(ax, agg, gx, gy, levels_fill, levels_line, norm,
                        anchor_markers, xlim, ylim, title, hull_pad,
                        ylabel=True, xlabel=True):
    from scipy.interpolate import griddata
    xs, ys, zs = agg["ref_x"].values, agg["ref_y"].values, agg["error"].values
    grid_z = griddata((xs, ys), zs, (gx, gy), method="cubic")
    mask = _hull_mask(xs, ys, gx, gy, pad=hull_pad)
    grid_z = np.where(mask, grid_z, np.nan)

    ax.contourf(gx, gy, grid_z, levels=levels_fill,
                cmap="RdYlGn_r", norm=norm, alpha=0.85, extend="max", zorder=1)
    cs = ax.contour(gx, gy, grid_z, levels=levels_line,
                    colors="black", linewidths=0.4, alpha=0.55, zorder=2)
    ax.clabel(cs, inline=True, fontsize=5, fmt="%.1f")
    ax.scatter(xs, ys, s=6, c="white", edgecolors="black",
               linewidths=0.3, zorder=3)
    for _, (a, mk) in anchor_markers.items():
        ax.scatter(a[:, 0], a[:, 1], marker=mk, s=28, facecolor="none",
                   edgecolor="black", linewidth=0.8, zorder=4)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]" if xlabel else "")
    ax.set_ylabel("y [m]" if ylabel else "")
    ax.set_title(title, fontsize=9)


def plot_spatial_heatmap(long_df, cfg=EVAL):
    """Interpolated contour heatmap of per-point mean error, fused technology.

    Emits a single combined PDF with one panel per available method and a
    shared colorbar axis, plus per-method PDFs for reuse elsewhere.
    """
    from matplotlib.colors import Normalize

    _setup_style(cfg)

    avail_methods = long_df[long_df["technology"] == "fused"]["method"].unique()
    method_order = _order_methods(avail_methods, cfg)
    method_data = _compute_point_errors(long_df, method_order, technology="fused")
    if not method_data:
        print("[plot_spatial_heatmap] no per-point data for any method; skipping")
        return []

    vmax_cfg = cfg["plot"].get("spatial_vmax", None)
    if vmax_cfg is not None:
        vmax = float(vmax_cfg)
    else:
        pct = cfg["plot"].get("spatial_vmax_percentile", 98.0)
        all_err = np.concatenate([d["error"].values for d in method_data.values()])
        vmax = float(np.nanpercentile(all_err, pct))
        vmax = max(vmax, 1.0)
        vmax = float(np.ceil(vmax / 0.5) * 0.5)

    norm = Normalize(vmin=0.0, vmax=vmax)

    _, anchor_xy, _ = di.get_anchor_xy("fused")
    anchor_markers = {
        "BLE": (anchor_xy[:5], "s"),
        "UWB": (anchor_xy[5:15], "^"),
        "WiFi": (anchor_xy[15:21], "D"),
    }

    all_xs = np.concatenate([d["ref_x"].values for d in method_data.values()] +
                            [anchor_xy[:, 0]])
    all_ys = np.concatenate([d["ref_y"].values for d in method_data.values()] +
                            [anchor_xy[:, 1]])
    pad_xy = 1.0
    xmin, xmax = all_xs.min() - pad_xy, all_xs.max() + pad_xy
    ymin, ymax = all_ys.min() - pad_xy, all_ys.max() + pad_xy
    xlim, ylim = (xmin, xmax), (ymin, ymax)
    nx, ny = cfg["plot"].get("spatial_grid_resolution", (120, 140))
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, nx),
                         np.linspace(ymin, ymax, ny))

    levels_fill = np.linspace(0, vmax, 15)
    levels_line = cfg["plot"].get("spatial_contour_levels", [0.5, 1.0, 2.0, 3.0, 5.0])
    hull_pad = cfg["plot"].get("spatial_hull_pad", 0.5)
    step = 1.0 if vmax <= 6 else (2.0 if vmax <= 12 else 5.0)
    cbar_ticks = np.arange(0, vmax + 1e-6, step)

    paths = []

    n = len(method_data)
    panel_w = 2.8
    width_ratios = [1.0] * n + [0.06]
    fig = plt.figure(figsize=(panel_w * n + 0.9, 3.2))
    gs = fig.add_gridspec(1, n + 1, width_ratios=width_ratios, wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cax = fig.add_subplot(gs[0, n])

    last_mappable = None
    for i, (method, agg) in enumerate(method_data.items()):
        _draw_spatial_panel(axes[i], agg, gx, gy, levels_fill, levels_line, norm,
                            anchor_markers, xlim, ylim, method, hull_pad,
                            ylabel=(i == 0), xlabel=True)
        from matplotlib.cm import ScalarMappable
        last_mappable = ScalarMappable(norm=norm, cmap="RdYlGn_r")
        last_mappable.set_array([])

    cb = fig.colorbar(last_mappable, cax=cax, extend="max")
    cb.set_label("Median position error [m]")
    cb.set_ticks(cbar_ticks)
    cb.ax.tick_params(labelsize=7)

    combined = os.path.join(FIG_DIR, "spatial_heatmap_fused.pdf")
    fig.savefig(combined, bbox_inches="tight")
    plt.close(fig)
    paths.append(combined)

    for method, agg in method_data.items():
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        _draw_spatial_panel(ax, agg, gx, gy, levels_fill, levels_line, norm,
                            anchor_markers, xlim, ylim, method, hull_pad)
        fig.tight_layout()
        fname = f"spatial_heatmap_fused_{method.replace('/', '-')}.pdf"
        out = os.path.join(FIG_DIR, fname)
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)

    return paths


def plot_all(long_df, cfg=EVAL):
    paths = []
    paths.append(plot_anchors_overview(cfg))
    ecdf = plot_ecdf_fused(long_df, cfg)
    if ecdf is not None:
        paths.append(ecdf)
    paths.extend(plot_boxplot_methods(long_df, cfg))
    paths.extend(plot_spatial_error(long_df, cfg))
    paths.extend(plot_spatial_heatmap(long_df, cfg))
    return paths
