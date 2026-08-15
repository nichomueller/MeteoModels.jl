"""
Plot FEM / RB / RB+calibration temperature-error field snapshots from the HeatEq
DA results stored in data/heateq/sol_*.vtu (fields e1 = true - FEM,
e2 = true - RB, e3 = true - RB+calib -- see the `createpvd` block at the end of
examples/HeatEq.jl).

Same low-level appended-data VTU reader and figure layout (colorbar on the left,
tripcolor with Gouraud shading, PowerNorm(gamma=0.4), no axis ticks/spines) as
plot_navier_stokes_error.py. Two differences from that script:
  - the HeatEq mesh is quadrilateral (VTK_QUAD, type 9), not triangular, so each
    cell is split into 2 triangles (using Gridap's own per-cell point layout,
    which already duplicates points at cell boundaries, so the split is exact);
  - the fields are scalar (temperature error), not vector (velocity error), so
    the "error magnitude" is simply the absolute value.
"""
import struct, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import PowerNorm
import xml.etree.ElementTree as ET

# ── paths ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__),
                        "..", "data", "heateq")
OUT_DIR  = os.path.join(os.path.dirname(__file__),
                        "..", "data", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

_MARKER = b"<AppendedData encoding=\"raw\">\n_"
_FIELDS = ["e1", "e2", "e3"]
_LABELS = ["FEM error", "RB error", "RB+calib error"]


# ── low-level VTU reader (quad topology, scalar point-data fields) ──────────

def _read_appended(ap, offset, dtype, n_elem, n_comp=1):
    nb = struct.unpack("<Q", ap[offset:offset + 8])[0]
    arr = np.frombuffer(ap[offset + 8:offset + 8 + nb], dtype=dtype).copy()
    return arr.reshape(n_elem, n_comp) if n_comp > 1 else arr


def read_vtu(path):
    with open(path, "rb") as f:
        raw = f.read()

    hdr_end = raw.find(b"<AppendedData")
    hdr_xml = raw[:hdr_end].decode("utf-8", "replace") + "</VTKFile>"
    root = ET.fromstring(hdr_xml)

    piece = root.find(".//Piece")
    n_pts = int(piece.get("NumberOfPoints"))
    n_cells = int(piece.get("NumberOfCells"))

    offsets = {}
    for da in root.iter("DataArray"):
        if da.get("format") == "appended":
            offsets[da.get("Name")] = int(da.get("offset"))

    ap = raw[raw.find(_MARKER) + len(_MARKER):]

    types = _read_appended(ap, offsets["types"], np.uint8, n_cells)
    if not np.all(types == 9):
        raise ValueError(f"expected VTK_QUAD (type 9) cells, got {np.unique(types)}")

    conn = _read_appended(ap, offsets["connectivity"], np.int32, n_cells, 4)
    # each quad's 4 corners are CCW (Gridap's createvtk convention): split the
    # quad [a,b,c,d] along the (a,c) diagonal into triangles (a,b,c),(a,c,d)
    tri = np.empty((2 * n_cells, 3), dtype=np.int64)
    tri[0::2] = conn[:, [0, 1, 2]]
    tri[1::2] = conn[:, [0, 2, 3]]

    pts = _read_appended(ap, offsets["Points"], np.float64, n_pts, 3)

    out = {"x": pts[:, 0], "y": pts[:, 1], "tri": tri}
    for name in _FIELDS:
        if name in offsets:
            out[name] = _read_appended(ap, offsets[name], np.float64, n_pts)

    return out


# ── figure builder: one row per model, one column per DA step ───────────────

def make_error_figure(vtu_files, out_path):
    n = len(vtu_files)
    rows = len(_FIELDS)

    datasets = [read_vtu(p) for p in vtu_files]
    tris = [mtri.Triangulation(d["x"], d["y"], d["tri"]) for d in datasets]
    fields = [[np.abs(d[name]) for d in datasets] for name in _FIELDS]
    norms = [
        PowerNorm(gamma=0.4, vmin=0.0, vmax=max(f.max() for f in row_fields))
        for row_fields in fields
    ]

    fig = plt.figure(figsize=(5.2 * n + 1.1, 3.2 * rows))
    gs = fig.add_gridspec(
        rows, n + 1,
        width_ratios=[0.14] + [1] * n,
        wspace=0.02, hspace=0.02,
        left=0.06, right=0.99, top=0.97, bottom=0.03,
    )

    def style_axis(ax):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    for r, (label, row_fields, norm) in enumerate(zip(_LABELS, fields, norms)):
        cbar_ax = fig.add_subplot(gs[r, 0])
        axes = [fig.add_subplot(gs[r, k + 1]) for k in range(n)]

        imgs = []
        for ax, tri, field in zip(axes, tris, row_fields):
            img = ax.tripcolor(tri, field, cmap="viridis", norm=norm,
                               shading="gouraud", rasterized=True)
            imgs.append(img)
            style_axis(ax)

        cb = fig.colorbar(imgs[0], cax=cbar_ax)
        cb.set_label(f"|{label}|", fontsize=10, labelpad=6)
        cb.ax.yaxis.set_label_position("left")
        cb.ax.yaxis.tick_left()
        cb.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────

steps = [10, 20, 30]
vtu_files = [os.path.join(DATA_DIR, f"sol_{s}.vtu") for s in steps]
missing = [p for p in vtu_files if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"missing VTU files: {missing}")

print(f"Using DA steps: {steps}")
make_error_figure(vtu_files, os.path.join(OUT_DIR, "heat_equation_error_fields.png"))
