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

Inputs may also be IMAGES (.bmp/.png/.jpg/...), e.g. screenshots rendered by
another tool: they are placed in their cells unchanged, and --mark / --box /
--inset then work in fractions of the image itself (top-left origin). For an
image, the inset is an enlarged pixel crop; --center/--index/--pick and the
coloring options do not apply.

Region selection (choose one style for all rows):
  --box FX,FY,FW,FH  rectangle as fractions of one column cell, origin at the
                     TOP-LEFT (like an image viewer), e.g. 0.10,0.30,0.25,0.25
  --center X,Y,Z --radius R
                     points inside a 3D sphere; the drawn rectangle is their
                     projected bounding box (aligned across columns for free)
  --index I --radius R
                     like --center, centered on point I of the row's 1st cloud
  --pick             click twice on an interactive window (single cloud only)

Highlight boxes without insets (real-world scans with no clean mesh):
  --mark FX,FY,FW,FH[:COLOR[:LINESTYLE]] draws a dashed rectangle at the same
  relative position in every cell; repeat it for several boxes. Works alone
  (no --box needed) or together with the inset options, e.g.
  python tools/visualize_zoom.py noisy.xyz m1.xyz ours.xyz scene.png \
      --color '#d95f57' --titles "Noisy,Method 1,Ours" \
      --mark 0.05,0.1,0.3,0.5:#4a7bd0:dashed --mark 0.5,0.55,0.4,0.3:#4a8f5a:dashdot

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


IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"}


def is_image(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def load_image(path):
    img = plt.imread(path)
    if img.ndim == 2:  # grayscale -> RGB
        img = np.stack([img] * 3, axis=-1)
    rgb = img[..., :3].astype(np.float64)
    if rgb.max() > 1.0:
        rgb /= 255.0
    return rgb


def tint_image(img, color, threshold):
    """Recolor an image's foreground (the points) with `color`.

    The background color is estimated from the image corners; every pixel
    sufficiently different from it counts as foreground and is replaced by
    the tint scaled by the pixel's original luminance, so splat shading and
    anti-aliased edges survive the recoloring.
    """
    h, w = img.shape[:2]
    k = max(2, min(h, w) // 50)
    corners = np.concatenate([img[:k, :k].reshape(-1, 3), img[:k, -k:].reshape(-1, 3),
                              img[-k:, :k].reshape(-1, 3), img[-k:, -k:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    fg = np.linalg.norm(img - bg, axis=-1) > threshold
    if not fg.any():
        return img
    lum = img @ np.array([0.299, 0.587, 0.114])
    out = img.copy()
    out[fg] = np.asarray(to_rgba(color)[:3]) * lum[fg, None]
    return out


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


def to_z_up(pts, up):
    """Rotate points so the chosen data axis becomes the screen-vertical z.

    Uses proper rotations (no mirroring), so the model keeps its handedness.
    Works with any matplotlib version, unlike view_init(vertical_axis=...).
    """
    if up == "y":  # +90 deg about x: (x, y, z) -> (x, -z, y)
        return pts[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
    if up == "x":  # +90 deg about y: (x, y, z) -> (-z, y, x)
        return pts[:, [2, 1, 0]] * np.array([-1.0, 1.0, 1.0])
    return pts


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
                         "join several regions with ':' for multiple insets per cell "
                         "(pair each with an --inset rect); repeatable, one per row")
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
    ap.add_argument("--mark", action="append",
                    help="highlight rectangle drawn in EVERY cell (no inset), as cell "
                         "fractions FX,FY,FW,FH[:COLOR[:LINESTYLE]] with top-left "
                         "origin, e.g. 0.05,0.1,0.3,0.5:#4a7bd0:dashed ; linestyles: "
                         "solid/dashed/dashdot/dotted; repeat for several boxes; can "
                         "be used alone or together with --box/--center")
    ap.add_argument("--mark-width", type=float, default=2.0,
                    help="line width of --mark rectangles")
    ap.add_argument("--inset", action="append",
                    help="inset position as cell fractions FX,FY,FW,FH (default "
                         "lower-right); join with ':' to match multiple --box regions; "
                         "repeatable, one per row")
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
                         "ShapeNet-style models are usually y-up); with it, --azim "
                         "spins the model about its natural vertical axis; "
                         "repeatable, one per row")
    ap.add_argument("--roll", type=float, action="append",
                    help="camera roll in degrees, default 0 (needs matplotlib >= 3.6); "
                         "repeatable, one per row")
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
    ap.add_argument("--tint",
                    help="recolor the foreground of IMAGE inputs with this color "
                         "(e.g. '#4a6fd8' to turn white points blue); the background "
                         "is auto-detected from the image corners and kept as is")
    ap.add_argument("--tint-threshold", type=float, default=0.1,
                    help="how different from the background a pixel must be to count "
                         "as foreground for --tint (0-1, default 0.1)")
    ap.add_argument("--overlay", help="second cloud drawn on top, single input only (e.g. noise)")
    ap.add_argument("--overlay-color", default="#e2574c", help="color of the overlay cloud")
    ap.add_argument("--overlay-point-size", type=float, help="marker area of the overlay cloud")
    ap.add_argument("--frame-color", default="#17798e",
                    help="color of the box and inset frame; a comma-separated list "
                         "cycles over the zoom regions (e.g. '#17798e,#4a8f5a')")
    ap.add_argument("--frame-width", type=float, default=4.0, help="inset frame line width")
    ap.add_argument("--connect", action="store_true", help="draw lines linking box and inset")
    ap.add_argument("--fit-percentile", type=float, default=100.0,
                    help="fit the view to this central percentile of the coordinates "
                         "(default 100 = all points); lower it to ~99 when far "
                         "outliers make the model tiny")
    ap.add_argument("--figsize", type=float, default=8.0,
                    help="size of one square grid cell in inches")
    ap.add_argument("--title-fontsize", type=float, default=24.0,
                    help="font size of the per-column method titles (default 24)")
    ap.add_argument("--legend-fontsize", type=float, default=24.0,
                    help="font size of the Clean/Noisy colorbar labels (default 24)")
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
    """Per-cloud RGBA arrays (None for image cells); scalar modes share one scale."""
    colors = [[None] * len(row) for row in grid_clouds]
    cloud_cells = [(r, c) for r, row in enumerate(grid_clouds)
                   for c, pts in enumerate(row) if pts is not None]
    if not cloud_cells:
        if meshes[0]:
            print("note: --mesh has no effect, all inputs are images")
        return colors

    if meshes[0]:
        scalars = {(r, c): p2f_distance(grid_clouds[r][c], meshes[r])
                   for r, c in cloud_cells}
        if args.eps > 0:
            scalars = {rc: np.where(s < args.eps, 0.0, s) for rc, s in scalars.items()}
        default_cmap = CLEAN_NOISY
    elif args.color_col is not None:
        scalars = {(r, c): grid_data[r][c][:, args.color_col] for r, c in cloud_cells}
        default_cmap = plt.get_cmap("coolwarm")
    else:
        for r, c in cloud_cells:
            colors[r][c] = np.tile(to_rgba(args.color), (len(grid_clouds[r][c]), 1))
        return colors

    stacked = np.concatenate(list(scalars.values()))
    vmin = args.vmin if args.vmin is not None else (
        0.0 if meshes[0] else float(stacked.min()))
    vmax = args.vmax if args.vmax is not None else (
        float(np.percentile(stacked, 99)) if meshes[0] else float(stacked.max()))
    print(f"color scale: --vmin {vmin:.6g} --vmax {vmax:.6g}  "
          "(pass these to other runs to keep colors comparable)")

    cmap = plt.get_cmap(args.cmap) if args.cmap else default_cmap
    for (r, c), s in scalars.items():
        colors[r][c] = cmap(
            np.clip((s - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0) ** args.gamma)
    return colors


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
    insets = [[parse_floats(t, 4, "inset") for t in spec.split(":")]
              for spec in per_row(args.inset, n_rows, "inset", "0.52,0.50,0.44,0.44")]
    frame_colors = [c.strip() for c in args.frame_color.split(",")]
    elevs = per_row(args.elev, n_rows, "elev", 20.0)
    azims = per_row(args.azim, n_rows, "azim", -60.0)
    ups = per_row(args.up, n_rows, "up", "z")
    rolls = per_row(args.roll, n_rows, "roll", 0.0)

    marks = []
    for text in args.mark or []:
        parts = text.split(":")
        rect = parse_floats(parts[0], 4, "mark")
        color = parts[1] if len(parts) > 1 and parts[1] else frame_colors[0]
        style = parts[2] if len(parts) > 2 and parts[2] else "dashed"
        marks.append((rect, color, style))

    # Region selection: one style shared by all rows, values broadcastable.
    have_region = bool(args.box or args.center or args.index is not None or args.pick)
    if args.box:
        boxes = [[parse_floats(t, 4, "box") for t in spec.split(":")]
                 for spec in per_row(args.box, n_rows, "box")]
        centers = None
        for r in range(n_rows):
            if len(insets[r]) != len(boxes[r]):
                raise SystemExit(
                    f"row {r + 1} has {len(boxes[r])} zoom boxes but {len(insets[r])} "
                    "insets: give one --inset rect per box, joined with ':'")
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
    elif not args.pick and not marks:
        raise SystemExit("select a region with --box, --center/--radius, --index or "
                         "--pick, and/or draw highlight boxes with --mark")

    grid_data, grid_clouds = [], []
    for row in grid_files:
        drow, crow = [], []
        for p in row:
            if is_image(p):
                img = load_image(p)
                if args.tint:
                    img = tint_image(img, args.tint, args.tint_threshold)
                drow.append(img)
                crow.append(None)
            else:
                d = load_points(p)
                drow.append(d)
                crow.append(d[:, :3])
        grid_data.append(drow)
        grid_clouds.append(crow)

    if any(pts is None for row in grid_clouds for pts in row):
        if args.center or args.index is not None or args.pick:
            raise SystemExit("image inputs support --box and --mark only "
                             "(no --center/--index/--pick)")

    grid_colors = resolve_colors(grid_clouds, grid_data, meshes, args)
    grid_sizes = [[np.full(len(pts), args.point_size) if pts is not None else None
                   for pts in row] for row in grid_clouds]

    if args.overlay:
        if grid_clouds[0][0] is None:
            raise SystemExit("--overlay needs a point cloud input, not an image")
        over = load_points(args.overlay)[:, :3]
        grid_clouds[0][0] = np.vstack([grid_clouds[0][0], over])
        grid_colors[0][0] = np.vstack([
            grid_colors[0][0], np.tile(to_rgba(args.overlay_color), (len(over), 1))])
        over_size = args.overlay_point_size or args.point_size
        grid_sizes[0][0] = np.concatenate([grid_sizes[0][0],
                                           np.full(len(over), over_size)])

    # Display-only rotation, after colors are computed on original coords.
    if any(u != "z" for u in ups):
        grid_clouds = [[to_z_up(pts, ups[r]) if pts is not None else None
                        for pts in row] for r, row in enumerate(grid_clouds)]

    titles = None
    if args.titles:
        titles = [t.strip() for t in args.titles.split(",")]
        if len(titles) != n_cols:
            raise SystemExit(f"--titles has {len(titles)} entries for {n_cols} columns")

    # ---- figure layout: square cells + optional top strip ------------------
    cell_in = args.figsize
    # 标题/图例条高度随字号自适应放大，避免大字号被裁切（字号 / 72 ≈ 英寸高，留 2x 余量）
    title_strip = (args.title_fontsize / 72.0 * 2.0 + 0.12) if titles else 0.0
    cbar_strip = (args.legend_fontsize / 72.0 * 2.0 + 0.20) if args.colorbar else 0.0
    strip_in = title_strip + cbar_strip
    fig_w, fig_h = n_cols * cell_in, n_rows * cell_in + strip_in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=args.bg)
    cell_h = cell_in / fig_h

    axes = []
    for r in range(n_rows):
        # Shared bounds per row so all methods render at the same scale.
        # Per-axis and centered: real-world scans sit far from the origin
        # with unequal extents, so a global min/max would shrink the model.
        row_pts = [pts for pts in grid_clouds[r] if pts is not None]
        if row_pts:
            all_pts = np.vstack(row_pts)
            q = (100.0 - args.fit_percentile) / 2.0
            mins = np.percentile(all_pts, q, axis=0)
            maxs = np.percentile(all_pts, 100.0 - q, axis=0)
            ctr = (mins + maxs) / 2.0
            half = float((maxs - mins).max()) * 0.5 * 1.05 + 1e-12
        row_axes = []
        for c in range(n_cols):
            cell = (c / n_cols, (n_rows - 1 - r) * cell_h, 1.0 / n_cols, cell_h)
            if grid_clouds[r][c] is None:
                ax = fig.add_axes(cell)
                ax.imshow(grid_data[r][c])
                ax.set_axis_off()
                row_axes.append((cell, ax, None))
                continue
            ax = fig.add_axes(cell, projection="3d")
            ax.set_axis_off()
            try:
                ax.view_init(elev=elevs[r], azim=azims[r], roll=rolls[r])
            except TypeError:
                if rolls[r]:
                    raise SystemExit("--roll needs matplotlib >= 3.6 "
                                     "(pip install -U matplotlib)")
                ax.view_init(elev=elevs[r], azim=azims[r])
            for axis, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                setter(ctr[axis] - half, ctr[axis] + half)
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
        if have_region and not args.box and not args.pick:
            # --center is given in original coords; --index picks an already
            # rotated point, so only the former needs the display rotation.
            center3d = (to_z_up(centers[r][None, :], ups[r])[0] if args.center
                        else grid_clouds[r][0][centers[r]])
        for c in range(n_cols):
            cell, ax, edges = axes[r][c]

            if titles and r == 0:
                fig.text(cell[0] + cell[2] / 2, n_rows * cell_h + 0.12 / fig_h,
                         titles[c], ha="center", va="bottom",
                         fontsize=args.title_fontsize)

            if grid_clouds[r][c] is None:
                # Image cell: --mark and --box are fractions of the image
                # itself, drawn in pixel coordinates (origin top-left).
                img = grid_data[r][c]
                h_px, w_px = img.shape[:2]
                for rect, mcolor, mstyle in marks:
                    fx, fy, fw, fh = rect
                    ax.add_patch(Rectangle((fx * w_px, fy * h_px),
                                           fw * w_px, fh * h_px, fill=False,
                                           edgecolor=mcolor, linestyle=mstyle,
                                           linewidth=args.mark_width, zorder=10))
                if have_region:
                    for k, (fx, fy, fw, fh) in enumerate(boxes[r]):
                        fcol = frame_colors[k % len(frame_colors)]
                        x0, y0, w, h = fx * w_px, fy * h_px, fw * w_px, fh * h_px
                        ax.add_patch(Rectangle((x0, y0), w, h, fill=False,
                                               edgecolor=fcol,
                                               linewidth=max(1.5,
                                                             args.frame_width * 0.45),
                                               zorder=10))
                        inset = fig.add_axes(cell_rect_to_fig(cell, *insets[r][k]))
                        inset.imshow(img, interpolation="bilinear")
                        inset.set_xlim(x0, x0 + w)
                        inset.set_ylim(y0 + h, y0)  # image origin is top-left
                        inset.set_xticks([])
                        inset.set_yticks([])
                        for spine in inset.spines.values():
                            spine.set_edgecolor(fcol)
                            spine.set_linewidth(args.frame_width)
                        print(f"{grid_files[r][c]}: image crop "
                              f"x[{x0:.0f}:{x0 + w:.0f}] y[{y0:.0f}:{y0 + h:.0f}] px")
                continue

            for rect, mcolor, mstyle in marks:
                mx, my, mw, mh = cell_rect_to_fig(cell, *rect)
                fig.add_artist(Rectangle((mx, my), mw, mh, transform=fig.transFigure,
                                         fill=False, edgecolor=mcolor,
                                         linewidth=args.mark_width,
                                         linestyle=mstyle, zorder=10))
            if not have_region:
                continue

            pts, cols, szs = grid_clouds[r][c], grid_colors[r][c], grid_sizes[r][c]
            px, py, pz = proj3d.proj_transform(
                pts[:, 0], pts[:, 1], pts[:, 2], ax.get_proj())
            disp = ax.transData.transform(np.column_stack([px, py]))
            frac = fig.transFigure.inverted().transform(disp)

            regions = []  # (bx, by, bw, bh, mask) in figure fractions
            if args.pick:
                bx, by, bw, bh = pick_box(fig)
                cx, cy, cw, ch = cell
                print("picked region: --box %.3f,%.3f,%.3f,%.3f" % (
                    (bx - cx) / cw, 1.0 - (by - cy + bh) / ch, bw / cw, bh / ch))
                regions.append((bx, by, bw, bh, None))
            elif args.box:
                for rect in boxes[r]:
                    regions.append(cell_rect_to_fig(cell, *rect) + (None,))
            else:
                mask = np.linalg.norm(pts - center3d, axis=1) <= radii[r]
                if not mask.any():
                    raise SystemExit(f"{grid_files[r][c]}: no points inside the sphere")
                sel = frac[mask]
                margin = 0.01 * cell[2]
                bx, by = sel[:, 0].min() - margin, sel[:, 1].min() - margin
                bw = sel[:, 0].max() - sel[:, 0].min() + 2 * margin
                bh = sel[:, 1].max() - sel[:, 1].min() + 2 * margin
                regions.append((bx, by, bw, bh, mask))

            for k, (bx, by, bw, bh, mask) in enumerate(regions):
                fcol = frame_colors[k % len(frame_colors)]
                if mask is None:
                    mask = ((frac[:, 0] >= bx) & (frac[:, 0] <= bx + bw)
                            & (frac[:, 1] >= by) & (frac[:, 1] <= by + bh))
                if not mask.any():
                    raise SystemExit(f"{grid_files[r][c]}: zoom box {k + 1} "
                                     "contains no points")

                fig.add_artist(Rectangle((bx, by), bw, bh, transform=fig.transFigure,
                                         fill=False, edgecolor=fcol,
                                         linewidth=max(1.5, args.frame_width * 0.45),
                                         zorder=10))

                ix, iy, iw, ih = cell_rect_to_fig(cell, *insets[r][k])
                inset = fig.add_axes([ix, iy, iw, ih], facecolor=args.bg)
                inset.set_xticks([])
                inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_edgecolor(fcol)
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
                inset.scatter(px[order], py[order], c=cols[order],
                              s=szs[order] * scale, edgecolors=inset_edges,
                              linewidths=0.25 * np.sqrt(scale))

                if args.connect:
                    for corner_y in (by, by + bh):
                        fig.add_artist(ConnectionPatch(
                            xyA=(bx + bw, corner_y), coordsA=fig.transFigure,
                            xyB=(ix, iy if corner_y == by else iy + ih),
                            coordsB=fig.transFigure,
                            color=fcol, linewidth=1.0, zorder=9))

                print(f"{grid_files[r][c]}: box {k + 1}: {int(mask.sum())} points "
                      f"in the inset, zoom x{zoom:.1f}")

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
                 ha="right", va="center", fontsize=args.legend_fontsize)
        fig.text(bar_x + bar_w + 0.08 / fig_w, bar_y + bar_h / 2, "Noisy",
                 ha="left", va="center", fontsize=args.legend_fontsize)

    fig.savefig(output, dpi=args.dpi, facecolor=args.bg)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
