"""
Plot velocity-error (and pressure-error, if available) field snapshots
from NavierStokes DA results stored in data/navier_stokes/sol.pvd.

Modifications vs. the original figure:
  - PowerNorm(gamma=0.4) so low-error regions are not pitch-black
  - Separate figure for pressure error (requires re-running NavierStokes.jl
    with the modified pvd section that also writes p / _p / error_p)
"""
import struct, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import PowerNorm, TwoSlopeNorm
import xml.etree.ElementTree as ET

# ── paths ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__),
                        "..", "data", "navier_stokes")
OUT_DIR  = os.path.join(os.path.dirname(__file__),
                        "..", "data", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── low-level VTU reader ─────────────────────────────────────────────────────

# Fixed mesh parameters (all sol_*.vtu share the same topology)
_N_PTS   = 4332
_N_CELLS = 1444

_MARKER = b"<AppendedData encoding=\"raw\">\n_"

_VEC3 = {"_u", "u", "error"}   # 3-component vector fields
_SCALAR = {"p", "_p", "error_p"}  # scalar fields


def _read_appended(ap, offset, dtype, n_elem, n_comp=1):
    nb = struct.unpack("<Q", ap[offset : offset + 8])[0]
    arr = np.frombuffer(ap[offset + 8 : offset + 8 + nb], dtype=dtype).copy()
    return arr.reshape(n_elem, n_comp) if n_comp > 1 else arr


def read_vtu(path):
    with open(path, "rb") as f:
        raw = f.read()

    # ── parse ALL offsets from the XML header (robust to field-order changes) ─
    hdr_end = raw.find(b"<AppendedData")
    hdr_xml = raw[:hdr_end].decode("utf-8","replace") + "</VTKFile>"
    root = ET.fromstring(hdr_xml)
    offsets = {}
    for da in root.iter("DataArray"):
        if da.get("format") == "appended":
            offsets[da.get("Name")] = int(da.get("offset"))

    ap = raw[raw.find(_MARKER) + len(_MARKER):]

    # mesh (triangles, zero-indexed)
    tri = _read_appended(ap, offsets["connectivity"], np.int32, None
                         ).reshape(_N_CELLS, 3)

    out = {"tri": tri}
    pts = _read_appended(ap, offsets["Points"], np.float64, _N_PTS, 3)
    out["x"] = pts[:, 0]
    out["y"] = pts[:, 1]

    for name in _VEC3 | _SCALAR:
        if name in offsets:
            nc = 3 if name in _VEC3 else 1
            out[name] = _read_appended(ap, offsets[name], np.float64, _N_PTS, nc)

    return out


# ── helpers ──────────────────────────────────────────────────────────────────

def vel_error_mag(data):
    e = data["error"]
    return np.sqrt(e[:, 0] ** 2 + e[:, 1] ** 2)


def pres_error(data):
    if "error_p" in data:
        return data["error_p"].ravel()
    if "p" in data and "_p" in data:
        return (data["p"] - data["_p"]).ravel()
    return None


# ── figure builder ───────────────────────────────────────────────────────────

def make_figure(vtu_files, field_fn, norm, cmap, cbar_label, out_path):
    """
    3-panel 1×N figure, colorbar on the left (matching the original layout).
    """
    n = len(vtu_files)
    datasets = [read_vtu(p) for p in vtu_files]
    fields   = [field_fn(d) for d in datasets]

    # figure: wide + narrow, leave room for the left-side colorbar
    fig = plt.figure(figsize=(5.2 * n + 1.1, 3.2))

    # one colorbar axis on the left + n field axes
    gs = fig.add_gridspec(1, n + 1,
                          width_ratios=[0.08] + [1] * n,
                          wspace=0.02, left=0.01, right=0.99,
                          top=0.97, bottom=0.03)

    cbar_ax = fig.add_subplot(gs[0, 0])
    axes    = [fig.add_subplot(gs[0, k + 1]) for k in range(n)]

    imgs = []
    for ax, data, field in zip(axes, datasets, fields):
        triang = mtri.Triangulation(data["x"], data["y"], data["tri"])
        img = ax.tripcolor(triang, field, cmap=cmap, norm=norm,
                           shading="gouraud", rasterized=True)
        imgs.append(img)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    cb = fig.colorbar(imgs[0], cax=cbar_ax)
    cb.set_label(cbar_label, fontsize=10, labelpad=6)
    cb.ax.yaxis.set_label_position("left")
    cb.ax.yaxis.tick_left()
    cb.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")

def make_combined_figure(vtu_files, out_path):
    """
    Generate velocity-error and pressure-error panels in one aligned figure.
    Rows share identical mesh axes and columns correspond to identical timesteps.
    """

    datasets = [read_vtu(p) for p in vtu_files]
    tris = [
        mtri.Triangulation(d["x"], d["y"], d["tri"])
        for d in datasets
    ]

    vel_fields = [vel_error_mag(d) for d in datasets]
    pres_fields = [pres_error(d) for d in datasets]

    have_pressure = all(f is not None for f in pres_fields)


    # norms ---------------------------------------------------------------

    vmax_vel = max(f.max() for f in vel_fields)
    vel_norm = PowerNorm(
        gamma=0.4,
        vmin=0.0,
        vmax=vmax_vel
    )

    if have_pressure:
        pmax = max(np.abs(f).max() for f in pres_fields)
        pres_norm = TwoSlopeNorm(
            vmin=-pmax,
            vcenter=0.0,
            vmax=pmax
        )


    n = len(vtu_files)

    # one shared layout ---------------------------------------------------
    rows = 2 if have_pressure else 1

    fig = plt.figure(
        figsize=(5.2*n + 1.1, 3.2*rows)
    )

    gs = fig.add_gridspec(
        rows,
        n + 1,
        width_ratios=[0.14] + [1]*n,
        wspace=0.02,
        hspace=0.02,
        left=0.06,
        right=0.99,
        top=0.97,
        bottom=0.03
    )


    # colorbars occupy first column ---------------------------------------

    cbar_vel_ax = fig.add_subplot(gs[0,0])

    vel_axes = [
        fig.add_subplot(gs[0,k+1])
        for k in range(n)
    ]


    pres_axes = []
    if have_pressure:
        cbar_pres_ax = fig.add_subplot(gs[1,0])

        pres_axes = [
            fig.add_subplot(gs[1,k+1])
            for k in range(n)
        ]


    def style_axis(ax):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)


    # velocity row --------------------------------------------------------

    vel_imgs = []

    for ax, tri, field in zip(
        vel_axes,
        tris,
        vel_fields
    ):
        im = ax.tripcolor(
            tri,
            field,
            cmap="viridis",
            norm=vel_norm,
            shading="gouraud",
            rasterized=True
        )

        vel_imgs.append(im)
        style_axis(ax)


    cb = fig.colorbar(
        vel_imgs[0],
        cax=cbar_vel_ax
    )

    cb.set_label(
        "Velocity error [m/s]",
        fontsize=10,
        labelpad=6
    )

    cb.ax.yaxis.set_label_position("left")
    cb.ax.yaxis.tick_left()


    # pressure row --------------------------------------------------------

    if have_pressure:

        pres_imgs = []

        for ax, tri, field in zip(
            pres_axes,
            tris,
            pres_fields
        ):
            im = ax.tripcolor(
                tri,
                field,
                cmap="viridis",
                norm=pres_norm,
                shading="gouraud",
                rasterized=True
            )

            pres_imgs.append(im)
            style_axis(ax)


        cb = fig.colorbar(
            pres_imgs[0],
            cax=cbar_pres_ax
        )

        cb.set_label(
            "Pressure error [Pa]",
            fontsize=10,
            labelpad=6
        )

        cb.ax.yaxis.set_label_position("left")
        cb.ax.yaxis.tick_left()


    fig.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.15
    )

    plt.close(fig)

    print(f"  → {out_path}")

# ── main ─────────────────────────────────────────────────────────────────────

# Sort files by timestep index
all_files = sorted(
    glob.glob(os.path.join(DATA_DIR, "sol_*.vtu")),
    key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
)
n_total = len(all_files)
# Pick first, middle, last DA snapshot (matching the original figure)
sel = [all_files[0], all_files[n_total // 2], all_files[-1]]

print(f"Using timesteps: 1, {n_total//2+1}, {n_total}  ({n_total} total)")

# ── combined aligned figure ────────────────────────────────────────────────

print("Generating aligned velocity/pressure error figure …")

make_combined_figure(
    vtu_files=sel,
    out_path=os.path.join(
        OUT_DIR,
        "navier_stokes_error_fields.png"
    )
)