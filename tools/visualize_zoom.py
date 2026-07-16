# -*- coding: utf-8 -*-
"""Render point clouds with magnified insets, PD-Flow-figure style.

Single-cloud mode reproduces the common "paper figure" layout: the cloud is
rendered in a 3D view, a rectangle marks a region of interest, and a framed
inset re-draws the points inside that region at a larger scale (a real
re-render, not a blurry pixel crop).

Passing several clouds renders a one-row comparison (one column per method):
same camera, same zoom box position in every column, and a color scale shared
across all columns so the methods are directly comparable.

Separating groups of clouds with a lone '+' renders a full comparison GRID
(one row per model, one column per method), sharing a single color scale and
colorbar across the whole figure. Row-specific options (--mesh, --box,
--inset, --center, --index, --radius, --elev, --azim) may be repeated: give
one value to use it for every row, or exactly one value per row.

Coloring (highest priority first):
  --mesh MESH        per-point point-to-face distance to a clean mesh,
                     mapped through a light->dark-blue "Clean -> Noisy"
                     gradient (needs point_cloud_utils)
  --color-col K      0-based input column used as a color scalar
  --color HEX        flat color (default soft lavender)

Point cloud formats: .npy / .npz (first array), .txt / .xyz / .pts
(whitespace, ';' or ',' separated; first 3 columns are XYZ).

Region selection (choose one style for all rows):
  --box FX,FY,FW,FH  rectangle as fractions of one column cell, origin at the
                     TOP-LEFT (like an image viewer), e.g. 0.10,0.30,0.25,0.25
  --center X,Y,Z --radius R
                     points inside a 3D sphere; the drawn rectangle is their
                     projected bounding box (aligned across columns for free)
  --index I --radius R
                     like --center, centered on point I of the row's 1st cloud
  --pick             click twice on an interactive window (single cloud only)

Typical single-figure usage:
  python tools/visualize_zoom.py denoised.xyz out.png \
      --mesh clean_mesh.off --box 0.12,0.35,0.22,0.22 --colorbar

One comparison row (shared color scale, one colorbar):
  python tools/visualize_zoom.py noisy.xyz m1.xyz m2.xyz ours.xyz row.png \
      --mesh clean_mesh.off --box 0.55,0.2,0.25,0.25 --colorbar \
      --titles "Noisy,Method A,Method B,Ours"

Full grid, two models x three methods, per-row mesh and zoom box:
  python tools/visualize_zoom.py \
      cat_noisy.xyz cat_m1.xyz cat_ours.xyz + \
      cow_noisy.xyz cow_m1.xyz cow_ours.xyz grid.png \
      --mesh cat.off --mesh cow.off \
      --box 0.1,0.3,0.25,0.25 --box 0.6,0.5,0.25,0.25 \
      --colorbar --titles "Noisy,Method A,Ours"

To keep colors comparable across SEPARATE runs, reuse the vmin/vmax that the
script prints via --vmin/--vmax.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY") and "--pick" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from mpl_toolkits.mplot3d import proj3d

# Light -> dark blue gradient mimicking the PD-Flow "Clean -> Noisy" figures.
CLEAN_NOISY = LinearSegmentedColormap.from_list(
    "clean-noisy", ["#f5f5fc", "#8e97dd", "#2b3a9e"])


def load_points(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        data = np.load(path)
    elif ext == ".npz":
        archive = np.load(path)
        data = archive[archive.files[0]]
    else:
        data = None
        for delim in (None, ";", ","):
            try:
                data = np.loadtxt(path, delimiter=delim)
                break
            except ValueError:
                continue
        if data is None:
            raise ValueError(f"could not parse {path} as a point cloud")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"{path}: expected at least 3 columns, got {data.shape[1]}")
    return data


def parse_floats(text, n, name):
    vals = [float(v) for v in text.replace(";", ",").split(",")]
    if len(vals) != n:
        raise SystemExit(f"--{name} expects {n} comma-separated numbers, got {text!r}")
    return vals


def p2f_distance(pts, mesh_path):
    """Unsigned per-point distance to a clean mesh."""
    try:
        import point_cloud_utils as pcu
    except ImportError:
        raise SystemExit("--mesh needs point_cloud_utils: pip install point-cloud-utils")
    v, f = pcu.load_mesh_vf(mesh_path)
    sdf, _, _ = pcu.signed_distance_to_mesh(
        np.ascontiguousarray(pts, dtype=np.float64), v.astype(np.float64), f)
    return np.abs(sdf)


def pick_box(fig):
    """Two clicks on the open figure -> figure-fraction rectangle."""
    clicks = []

    def on_click(event):
        clicks.append((event.x, event.y))
        if len(clicks) == 2:
            fig.canvas.stop_event_loop()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    print("Click the two opposite corners of the region to magnify...")
    plt.show(block=False)
    fig.canvas.start_event_loop(timeout=300)
    fig.canvas.mpl_disconnect(cid)
    if len(clicks) < 2:
        raise SystemExit("need two clicks to define the region")
    w, h = fig.canvas.get_width_height()
    (x0, y0), (x1, y1) = clicks
    fx0, fx1 = sorted((x0 / w, x1 / w))
    fy0, fy1 = sorted((y0 / h, y1 / h))  # event coords are already bottom-up
    return fx0, fy0, fx1 - fx0, fy1 - fy0


def cell_rect_to_fig(cell, fx, fy, fw, fh):
    """Top-left-origin fractions of a grid cell -> figure-fraction rect."""
    cx, cy, cw, ch = cell
    return cx + fx * cw, cy + (1.0 - fy - fh) * ch, fw * cw, fh * ch


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+", metavar="INPUTS OUTPUT",
                    help="point cloud files followed by the output image; separate "
                         "rows of a grid with a lone '+'")
    ap.add_argument("--box", action="append",
                    help="zoom region as cell fractions FX,FY,FW,FH (top-left origin); "
                         "repeatable, one per row")
    ap.add_argument("--center", action="append",
                    help="zoom region center in 3D coords X,Y,Z (with --radius); "
                         "repeatable, one per row")
    ap.add_argument("--index", type=int, action="append",
                    help="zoom region centered on this point of the row's first cloud "
                         "(with --radius); repeatable, one per row")
    ap.add_argument("--radius", type=float, action="append",
                    help="3D radius for --center/--index; repeatable, one per row")
    ap.add_argument("--pick", action="store_true",
                    help="pick the region interactively, 2 clicks (single cloud only)")
    ap.add_argument("--inset", action="append",
                    help="inset position as cell fractions FX,FY,FW,FH (default "
                         "lower-right); repeatable, one per row")
    ap.add_argument("--mesh", action="append",
                    help="clean mesh (.off/.obj/.ply): color points by P2F distance; "
                         "repeatable, one per row")
    ap.add_argument("--eps", type=float, default=0.0,
                    help="P2F distances below this absolute value count as perfectly clean")
    ap.add_argument("--vmin", type=float, help="fixed lower bound of the color scale")
    ap.add_argument("--vmax", type=float,
                    help="fixed upper bound of the color scale (default: 99th percentile "
                         "over ALL inputs; the value used is printed for reuse)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="nonlinearity applied to the normalized distance (e.g. 0.7)")
    ap.add_argument("--colorbar", action="store_true",
                    help="draw a Clean->Noisy gradient legend above the figure")
    ap.add_argument("--titles", help="comma-separated per-column titles (method names)")
    ap.add_argument("--elev", type=float, action="append",
                    help="camera elevation, default 20; repeatable, one per row")
    ap.add_argument("--azim", type=float, action="append",
                    help="camera azimuth, default -60; repeatable, one per row")
    ap.add_argument("--up", action="append", choices=["x", "y", "z"],
                    help="which data axis points up on screen (default z; PUNet/"
                         "ShapeNet-style models are usually y-up); repeatable, one per row")
    ap.add_argument("--roll", type=float, action="append",
                    help="camera roll in degrees, default 0; repeatable, one per row")
    ap.add_argument("--point-size", type=float, default=2.0, help="marker area in the main view")
    ap.add_argument("--max-zoom-scale", type=float, default=200.0,
                    help="cap on the marker-area magnification in the inset")
    ap.add_argument("--color", default="#b9bce8",
                    help="flat point color used when neither --mesh nor --color-col is given")
    ap.add_argument("--edge-color", default="auto",
                    help="marker edge color; 'auto' darkens each point's own color, '' disables")
    ap.add_argument("--color-col", type=int,
                    help="0-based column of the input used as a scalar for coloring")
    ap.add_argument("--cmap", help="matplotlib colormap overriding the default "
                                   "(clean-noisy for --mesh, coolwarm for --color-col)")
    ap.add_argument("--overlay", help="second cloud drawn on top, single input only (e.g. noise)")
    ap.add_argument("--overlay-color", default="#e2574c", help="color of the overlay cloud")
    ap.add_argument("--overlay-point-size", type=float, help="marker area of the overlay cloud")
    ap.add_argument("--frame-color", default="#17798e", help="color of the box and inset frame")
    ap.add_argument("--frame-width", type=float, default=4.0, help="inset frame line width")
    ap.add_argument("--connect", action="store_true", help="draw lines linking box and inset")
    ap.add_argument("--figsize", type=float, default=8.0,
                    help="size of one square grid cell in inches")
    ap.add_argument("--dpi", type=int, default=300, help="output resolution")
    ap.add_argument("--bg", default="white", help="background color")
    return ap


def split_rows(paths):
    """INPUTS with '+' separators + trailing OUTPUT -> (rows of files, output)."""
    if len(paths) < 2:
        raise SystemExit("need at least one input cloud and the output image path")
    output = paths[-1]
    rows, current = [], []
    for token in paths[:-1]:
        if token == "+":
            if not current:
                raise SystemExit("empty row: two '+' separators with nothing between")
            rows.append(current)
            current = []
        else:
            current.append(token)
    if not current:
        raise SystemExit("the last row is empty")
    rows.append(current)
    n_cols = len(rows[0])
    for r in rows:
        if len(r) != n_cols:
            raise SystemExit(f"every row needs the same number of clouds "
                             f"(got {[len(r) for r in rows]})")
    return rows, output


def per_row(values, n_rows, name, default=None):
    """Broadcast a repeatable option: one value -> all rows, else one per row."""
    if not values:
        values = [default]
    if len(values) == 1:
        return values * n_rows
    if len(values) != n_rows:
        raise SystemExit(f"--{name} was given {len(values)} times but there are "
                         f"{n_rows} rows: give it once, or once per row")
    return values


def resolve_colors(grid_clouds, grid_data, meshes, args):
    """Per-cloud RGBA arrays; scalar-based modes share one normalization."""
    if meshes[0]:
        scalars = [[p2f_distance(pts, meshes[r]) for pts in row]
                   for r, row in enumerate(grid_clouds)]
        if args.eps > 0:
            scalars = [[np.where(d < args.eps, 0.0, d) for d in row] for row in scalars]
        default_cmap = CLEAN_NOISY
    elif args.color_col is not None:
        scalars = [[d[:, args.color_col] for d in row] for row in grid_data]
        default_cmap = plt.get_cmap("coolwarm")
    else:
        return [[np.tile(to_rgba(args.color), (len(pts), 1)) for pts in row]
                for row in grid_clouds]

    stacked = np.concatenate([s for row in scalars for s in row])
    vmin = args.vmin if args.vmin is not None else (
        0.0 if meshes[0] else float(stacked.min()))
    vmax = args.vmax if args.vmax is not None else (
        float(np.percentile(stacked, 99)) if meshes[0] else float(stacked.max()))
    print(f"color scale: --vmin {vmin:.6g} --vmax {vmax:.6g}  "
          "(pass these to other runs to keep colors comparable)")

    cmap = plt.get_cmap(args.cmap) if args.cmap else default_cmap
    return [[cmap(np.clip((s - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0) ** args.gamma)
             for s in row] for row in scalars]


def edge_colors_for(colors, args):
    if args.edge_color == "auto":
        edges = colors.copy()
        edges[:, :3] *= 0.75
        return edges
    return args.edge_color or "none"


def main():
    args = build_parser().parse_args()
    grid_files, output = split_rows(args.paths)
    n_rows, n_cols = len(grid_files), len(grid_files[0])
    n_total = n_rows * n_cols
    if n_total > 1 and args.pick:
        raise SystemExit("--pick works with a single input cloud only")
    if n_total > 1 and args.overlay:
        raise SystemExit("--overlay works with a single input cloud only")

    meshes = per_row(args.mesh, n_rows, "mesh")
    if any(meshes) and not all(meshes):
        raise SystemExit("--mesh must be given for every row (or once for all rows)")
    insets = [parse_floats(t, 4, "inset")
              for t in per_row(args.inset, n_rows, "inset", "0.52,0.50,0.44,0.44")]
    elevs = per_row(args.elev, n_rows, "elev", 20.0)
    azims = per_row(args.azim, n_rows, "azim", -60.0)
    ups = per_row(args.up, n_rows, "up", "z")
    rolls = per_row(args.roll, n_rows, "roll", 0.0)

    # Region selection: one style shared by all rows, values broadcastable.
    if args.box:
        boxes = [parse_floats(t, 4, "box") for t in per_row(args.box, n_rows, "box")]
        centers = None
    elif args.center or args.index is not None:
        radii = per_row(args.radius, n_rows, "radius")
        if radii[0] is None:
            raise SystemExit("--center/--index also needs --radius")
        boxes = None
        if args.center:
            centers = [np.array(parse_floats(t, 3, "center"))
                       for t in per_row(args.center, n_rows, "center")]
        else:
            centers = per_row(args.index, n_rows, "index")  # resolved to 3D below
    elif not args.pick:
        raise SystemExit("select a region with --box, --center/--radius, --index or --pick")

    grid_data = [[load_points(p) for p in row] for row in grid_files]
    grid_clouds = [[d[:, :3] for d in row] for row in grid_data]
    grid_colors = resolve_colors(grid_clouds, grid_data, meshes, args)
    grid_sizes = [[np.full(len(pts), args.point_size) for pts in row]
                  for row in grid_clouds]

    if args.overlay:
        over = load_points(args.overlay)[:, :3]
        grid_clouds[0][0] = np.vstack([grid_clouds[0][0], over])
        grid_colors[0][0] = np.vstack([
            grid_colors[0][0], np.tile(to_rgba(args.overlay_color), (len(over), 1))])
        over_size = args.overlay_point_size or args.point_size
        grid_sizes[0][0] = np.concatenate([grid_sizes[0][0],
                                           np.full(len(over), over_size)])

    titles = None
    if args.titles:
        titles = [t.strip() for t in args.titles.split(",")]
        if len(titles) != n_cols:
            raise SystemExit(f"--titles has {len(titles)} entries for {n_cols} columns")

    # ---- figure layout: square cells + optional top strip ------------------
    cell_in = args.figsize
    strip_in = (0.45 if titles else 0.0) + (0.55 if args.colorbar else 0.0)
    fig_w, fig_h = n_cols * cell_in, n_rows * cell_in + strip_in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=args.bg)
    cell_h = cell_in / fig_h

    axes = []
    for r in range(n_rows):
        # Shared bounds per row so all methods render at the same scale.
        lo = min(pts.min() for pts in grid_clouds[r])
        hi = max(pts.max() for pts in grid_clouds[r])
        pad = 0.05 * (hi - lo)
        row_axes = []
        for c in range(n_cols):
            cell = (c / n_cols, (n_rows - 1 - r) * cell_h, 1.0 / n_cols, cell_h)
            ax = fig.add_axes(cell, projection="3d")
            ax.set_axis_off()
            ax.view_init(elev=elevs[r], azim=azims[r], roll=rolls[r],
                         vertical_axis=ups[r])
            for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
                setter(lo - pad, hi + pad)
            ax.set_box_aspect((1, 1, 1))
            pts, cols = grid_clouds[r][c], grid_colors[r][c]
            edges = edge_colors_for(cols, args)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=grid_sizes[r][c],
                       edgecolors=edges, linewidths=0.25, depthshade=True)
            row_axes.append((cell, ax, edges))
        axes.append(row_axes)

    fig.canvas.draw()  # transforms are only valid after a draw

    # ---- zoom box + inset per cell ------------------------------------------
    for r in range(n_rows):
        center3d = None
        if not args.box and not args.pick:
            center3d = (centers[r] if args.center
                        else grid_clouds[r][0][centers[r]])
        for c in range(n_cols):
            cell, ax, edges = axes[r][c]
            pts, cols, szs = grid_clouds[r][c], grid_colors[r][c], grid_sizes[r][c]
            px, py, pz = proj3d.proj_transform(
                pts[:, 0], pts[:, 1], pts[:, 2], ax.get_proj())
            disp = ax.transData.transform(np.column_stack([px, py]))
            frac = fig.transFigure.inverted().transform(disp)

            mask = None
            if args.pick:
                bx, by, bw, bh = pick_box(fig)
                cx, cy, cw, ch = cell
                print("picked region: --box %.3f,%.3f,%.3f,%.3f" % (
                    (bx - cx) / cw, 1.0 - (by - cy + bh) / ch, bw / cw, bh / ch))
            elif args.box:
                bx, by, bw, bh = cell_rect_to_fig(cell, *boxes[r])
            else:
                mask = np.linalg.norm(pts - center3d, axis=1) <= radii[r]
                if not mask.any():
                    raise SystemExit(f"{grid_files[r][c]}: no points inside the sphere")
                sel = frac[mask]
                margin = 0.01 * cell[2]
                bx, by = sel[:, 0].min() - margin, sel[:, 1].min() - margin
                bw = sel[:, 0].max() - sel[:, 0].min() + 2 * margin
                bh = sel[:, 1].max() - sel[:, 1].min() + 2 * margin

            if mask is None:
                mask = ((frac[:, 0] >= bx) & (frac[:, 0] <= bx + bw)
                        & (frac[:, 1] >= by) & (frac[:, 1] <= by + bh))
            if not mask.any():
                raise SystemExit(f"{grid_files[r][c]}: the selected rectangle "
                                 "contains no points")

            fig.add_artist(Rectangle((bx, by), bw, bh, transform=fig.transFigure,
                                     fill=False, edgecolor=args.frame_color,
                                     linewidth=max(1.5, args.frame_width * 0.45),
                                     zorder=10))

            ix, iy, iw, ih = cell_rect_to_fig(cell, *insets[r])
            inset = fig.add_axes([ix, iy, iw, ih], facecolor=args.bg)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_edgecolor(args.frame_color)
                spine.set_linewidth(args.frame_width)

            # Inset limits in projected data coordinates = the rectangle drawn.
            inv = ax.transData.inverted()
            (dx0, dy0), (dx1, dy1) = inv.transform(
                fig.transFigure.transform([[bx, by], [bx + bw, by + bh]]))
            inset.set_xlim(dx0, dx1)
            inset.set_ylim(dy0, dy1)
            inset.set_aspect("equal", adjustable="datalim")

            # Painter's algorithm: draw far points first, like mplot3d does.
            order = np.flatnonzero(mask)[np.argsort(pz[mask])[::-1]]
            zoom = min(iw / bw, ih / bh)
            scale = min(zoom ** 2, args.max_zoom_scale)
            inset_edges = edges[order] if isinstance(edges, np.ndarray) else edges
            inset.scatter(px[order], py[order], c=cols[order], s=szs[order] * scale,
                          edgecolors=inset_edges, linewidths=0.25 * np.sqrt(scale))

            if args.connect:
                for corner_y in (by, by + bh):
                    fig.add_artist(ConnectionPatch(
                        xyA=(bx + bw, corner_y), coordsA=fig.transFigure,
                        xyB=(ix, iy if corner_y == by else iy + ih),
                        coordsB=fig.transFigure,
                        color=args.frame_color, linewidth=1.0, zorder=9))

            if titles and r == 0:
                fig.text(cell[0] + cell[2] / 2, n_rows * cell_h + 0.12 / fig_h,
                         titles[c], ha="center", va="bottom", fontsize=16)
            print(f"{grid_files[r][c]}: {int(mask.sum())} points in the inset, "
                  f"zoom x{zoom:.1f}")

    # ---- Clean -> Noisy legend ----------------------------------------------
    if args.colorbar:
        cmap = plt.get_cmap(args.cmap) if args.cmap else CLEAN_NOISY
        bar_w, bar_h = min(0.16, 1.6 / fig_w), 0.16 / fig_h
        bar_y = 1.0 - (0.36 / fig_h)
        bar_x = 1.0 - bar_w - 0.9 / fig_w
        cax = fig.add_axes([bar_x, bar_y, bar_w, bar_h])
        cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap)
        cax.set_xticks([])
        cax.set_yticks([])
        for spine in cax.spines.values():
            spine.set_linewidth(0.5)
        fig.text(bar_x - 0.08 / fig_w, bar_y + bar_h / 2, "Clean",
                 ha="right", va="center", fontsize=16)
        fig.text(bar_x + bar_w + 0.08 / fig_w, bar_y + bar_h / 2, "Noisy",
                 ha="left", va="center", fontsize=16)

    fig.savefig(output, dpi=args.dpi, facecolor=args.bg)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
