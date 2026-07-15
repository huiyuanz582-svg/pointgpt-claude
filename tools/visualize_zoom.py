# -*- coding: utf-8 -*-
"""Render a point cloud with a magnified inset of a selected region.

Reproduces the common "paper figure" style: the full cloud is rendered in a
3D view, a rectangle marks a region of interest, and a framed inset shows the
points inside that region re-drawn at a larger scale (real re-render, not a
blurry pixel crop).

Point cloud formats: .npy / .npz (first array), .txt / .xyz / .pts
(whitespace, ';' or ',' separated; first 3 columns are XYZ, an optional 4th
column can be used as a scalar for coloring via --color-col).

Region selection (choose one):
  --box FX,FY,FW,FH      rectangle in image fractions (origin at the TOP-LEFT
                         of the saved image, like an image viewer), e.g.
                         0.10,0.30,0.25,0.25
  --center X,Y,Z --radius R
                         all points within a 3D sphere; the drawn rectangle is
                         the projected bounding box of those points
  --index I --radius R   like --center but the sphere is centered on point I
  --pick                 click twice on an interactive window to define the
                         rectangle (needs a display)

Typical usage:
  python tools/visualize_zoom.py vis/denoised.txt out.png \
      --box 0.12,0.35,0.22,0.22 --inset 0.55,0.52,0.42,0.42 \
      --elev 20 --azim -60 --point-size 2

Overlay a second cloud (e.g. noisy input in red over the denoised result):
  python tools/visualize_zoom.py denoised.txt out.png --color '#6d76d8' \
      --overlay noisy.txt --overlay-color '#e2574c' --box 0.4,0.4,0.2,0.2
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
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import proj3d


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


def image_frac_to_fig(fx, fy, fw, fh):
    """Convert a top-left-origin image-fraction rect to figure coords."""
    return fx, 1.0 - fy - fh, fw, fh


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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", help="point cloud file (.txt/.xyz/.pts/.npy/.npz)")
    ap.add_argument("output", help="output image path (.png/.pdf/...)")
    ap.add_argument("--box", help="zoom region as image fractions FX,FY,FW,FH (top-left origin)")
    ap.add_argument("--center", help="zoom region center in 3D coords X,Y,Z (with --radius)")
    ap.add_argument("--index", type=int, help="zoom region centered on point INDEX (with --radius)")
    ap.add_argument("--radius", type=float, help="3D radius for --center/--index selection")
    ap.add_argument("--pick", action="store_true", help="pick the region interactively (2 clicks)")
    ap.add_argument("--inset", default="0.52,0.50,0.44,0.44",
                    help="inset position as image fractions FX,FY,FW,FH (default lower-right)")
    ap.add_argument("--elev", type=float, default=20, help="camera elevation (default 20)")
    ap.add_argument("--azim", type=float, default=-60, help="camera azimuth (default -60)")
    ap.add_argument("--point-size", type=float, default=2.0, help="marker area in the main view")
    ap.add_argument("--max-zoom-scale", type=float, default=200.0,
                    help="cap on the marker-area magnification in the inset")
    ap.add_argument("--color", default="#b9bce8",
                    help="flat point color (default soft lavender, like the reference figure)")
    ap.add_argument("--edge-color", default="#8d90c4", help="marker edge color ('' disables)")
    ap.add_argument("--color-col", type=int,
                    help="0-based column of the input used as a scalar for coloring")
    ap.add_argument("--cmap", default="coolwarm", help="colormap for --color-col")
    ap.add_argument("--overlay", help="optional second point cloud drawn on top (e.g. noise)")
    ap.add_argument("--overlay-color", default="#e2574c", help="color of the overlay cloud")
    ap.add_argument("--overlay-point-size", type=float, help="marker area of the overlay cloud")
    ap.add_argument("--frame-color", default="#17798e", help="color of the box and inset frame")
    ap.add_argument("--frame-width", type=float, default=4.0, help="inset frame line width")
    ap.add_argument("--connect", action="store_true", help="draw lines linking box and inset")
    ap.add_argument("--figsize", type=float, default=8.0, help="figure size in inches (square)")
    ap.add_argument("--dpi", type=int, default=300, help="output resolution")
    ap.add_argument("--bg", default="white", help="background color")
    args = ap.parse_args()

    data = load_points(args.input)
    pts = data[:, :3]
    n_base = len(pts)

    # Per-point colors for the base cloud.
    if args.color_col is not None:
        scalars = data[:, args.color_col]
        norm = plt.Normalize(scalars.min(), scalars.max())
        colors = plt.get_cmap(args.cmap)(norm(scalars))
    else:
        colors = np.tile(to_rgba(args.color), (n_base, 1))

    sizes = np.full(n_base, args.point_size)

    if args.overlay:
        over = load_points(args.overlay)[:, :3]
        pts = np.vstack([pts, over])
        colors = np.vstack([colors, np.tile(to_rgba(args.overlay_color), (len(over), 1))])
        over_size = args.overlay_point_size or args.point_size
        sizes = np.concatenate([sizes, np.full(len(over), over_size)])

    edge = args.edge_color or "none"

    # ---- main 3D view ----------------------------------------------------
    fig = plt.figure(figsize=(args.figsize, args.figsize), facecolor=args.bg)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    ax.set_axis_off()
    ax.view_init(elev=args.elev, azim=args.azim)
    lo, hi = pts.min(), pts.max()
    pad = 0.05 * (hi - lo)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(lo - pad, hi + pad)
    ax.set_box_aspect((1, 1, 1))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=sizes,
               edgecolors=edge, linewidths=0.25, depthshade=True)

    fig.canvas.draw()  # transforms are only valid after a draw

    # Project every point into the figure to know where it lands on screen.
    px, py, pz = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], ax.get_proj())
    disp = ax.transData.transform(np.column_stack([px, py]))
    frac = fig.transFigure.inverted().transform(disp)  # figure fractions

    # ---- resolve the zoom region into a figure-fraction rectangle --------
    mask = None
    if args.pick:
        bx, by, bw, bh = pick_box(fig)
    elif args.box:
        bx, by, bw, bh = image_frac_to_fig(*parse_floats(args.box, 4, "box"))
    elif args.center or args.index is not None:
        if args.radius is None:
            raise SystemExit("--center/--index also needs --radius")
        center = (np.array(parse_floats(args.center, 3, "center"))
                  if args.center else pts[args.index])
        mask = np.linalg.norm(pts - center, axis=1) <= args.radius
        if not mask.any():
            raise SystemExit("no points inside the given sphere")
        sel = frac[mask]
        margin = 0.01
        bx, by = sel[:, 0].min() - margin, sel[:, 1].min() - margin
        bw = sel[:, 0].max() - sel[:, 0].min() + 2 * margin
        bh = sel[:, 1].max() - sel[:, 1].min() + 2 * margin
    else:
        raise SystemExit("select a region with --box, --center/--radius, --index or --pick")

    if mask is None:
        mask = ((frac[:, 0] >= bx) & (frac[:, 0] <= bx + bw)
                & (frac[:, 1] >= by) & (frac[:, 1] <= by + bh))
    if not mask.any():
        raise SystemExit("the selected rectangle contains no points")

    # Rectangle marking the region on the main view.
    fig.add_artist(Rectangle((bx, by), bw, bh, transform=fig.transFigure,
                             fill=False, edgecolor=args.frame_color,
                             linewidth=max(1.5, args.frame_width * 0.45), zorder=10))

    # ---- inset: re-draw the selected points, magnified -------------------
    ix, iy, iw, ih = image_frac_to_fig(*parse_floats(args.inset, 4, "inset"))
    inset = fig.add_axes([ix, iy, iw, ih], facecolor=args.bg)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor(args.frame_color)
        spine.set_linewidth(args.frame_width)

    # Inset limits in projected data coordinates = the rectangle we drew.
    inv = ax.transData.inverted()
    corners_disp = fig.transFigure.transform([[bx, by], [bx + bw, by + bh]])
    (dx0, dy0), (dx1, dy1) = inv.transform(corners_disp)
    inset.set_xlim(dx0, dx1)
    inset.set_ylim(dy0, dy1)
    inset.set_aspect("equal", adjustable="datalim")

    # Painter's algorithm: draw far points first, like mplot3d does.
    order = np.flatnonzero(mask)[np.argsort(pz[mask])[::-1]]
    zoom = min(iw / bw, ih / bh)
    scale = min(zoom ** 2, args.max_zoom_scale)
    inset.scatter(px[order], py[order], c=colors[order], s=sizes[order] * scale,
                  edgecolors=edge, linewidths=0.25 * np.sqrt(scale))

    if args.connect:
        for cx, cy in ((bx + bw, by), (bx + bw, by + bh)):
            fig.add_artist(ConnectionPatch(
                xyA=(cx, cy), coordsA=fig.transFigure,
                xyB=(ix, iy if cy == by else iy + ih), coordsB=fig.transFigure,
                color=args.frame_color, linewidth=1.0, zorder=9))

    fig.savefig(args.output, dpi=args.dpi, facecolor=args.bg)
    print(f"saved {args.output}  ({int(mask.sum())} points in the inset, zoom x{zoom:.1f})")
    if args.pick:
        img_box = (bx, 1.0 - by - bh, bw, bh)
        print("re-run without --pick using: --box %.3f,%.3f,%.3f,%.3f" % img_box)


if __name__ == "__main__":
    main()
