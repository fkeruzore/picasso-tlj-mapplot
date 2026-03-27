import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import cmocean
import seaborn as sns

# Patch parameters
CENTER_RA_DEG = 20.0
CENTER_DEC_DEG = 20.0
X_SIZE_DEG = 3.0
Y_SIZE_DEG = 9.0
RESOLUTION_ARCMIN = 0.5

MAP_CONFIGS = [
    (
        # "./halos_test.h5",
        "/home/fkeruzore/SkySimz/picasso-tlj/LJLC_TSZ/"
        "8192_theta3.0t200_all-m/halos.h5",
        {"type": "halos", "cmap": "berlin", "label": "Redshift $z$"},
    ),
    (
        "/home/fkeruzore/SkySimz/picasso-tlj/LJLC_TSZ/"
        "8192_theta3.0t200_all-m/coadded_map.all.fits",
        {
            "cmap": cmocean.cm.tempo,
            "vmin": 0,
            "vmax": 5e-6,
            "label": "tSZ $y$",
        },
    ),
    (
        "/data/a/cpac/prlarsen/sharing/ksz_LJ/ksz_map_paper.fits",
        {
            "cmap": sns.diverging_palette(220, 20, as_cmap=True),
            "vmin": -5e-6,
            "vmax": 5e-6,
            "label": "kSZ $b$",
        },
    ),
    (
        "/data/a/cpac/prlarsen/sharing/ksz_LJ/kappa_CMB.fits",
        {"cmap": cmocean.cm.ice, "vmin": -0.5, "vmax": 2.0, "label": "Lensing"},
    ),
    (
        "./map_cib.fits",
        {"cmap": cmocean.cm.thermal, "vmin": 0, "vmax": 0.1, "label": "CIB"},
    ),
    (
        "./map_radio.fits",
        {"cmap": cmocean.cm.dense_r, "vmin": 0, "vmax": 0.1, "label": "Radio"},
    ),
]


def gnomonic_patch(
    m, center_ra_deg, center_dec_deg, x_size_deg, y_size_deg, resolution_arcmin
):
    """Return a gnomonic (flat-sky) projection of healpix map m as a
    2D array."""
    n_x = int(round(x_size_deg * 60 / resolution_arcmin))
    n_y = int(round(y_size_deg * 60 / resolution_arcmin))

    # Tangent-plane offsets in radians
    xi = np.linspace(
        -np.radians(x_size_deg) / 2, np.radians(x_size_deg) / 2, n_x
    )
    eta = np.linspace(
        -np.radians(y_size_deg) / 2, np.radians(y_size_deg) / 2, n_y
    )
    xi_grid, eta_grid = np.meshgrid(xi, eta)

    # Center in radians; healpy uses colatitude theta and longitude phi
    phi0 = np.radians(center_ra_deg)
    theta0 = np.radians(90.0 - center_dec_deg)  # colatitude

    # Inverse gnomonic projection
    rho = np.sqrt(xi_grid**2 + eta_grid**2)
    c = np.arctan(rho)

    cos_c = np.cos(c)
    sin_c = np.sin(c)
    cos_t0 = np.cos(theta0)
    sin_t0 = np.sin(theta0)

    # Declination of each pixel
    sin_dec = cos_c * cos_t0 + eta_grid * sin_c * sin_t0 / np.where(
        rho == 0, 1, rho
    )
    sin_dec = np.where(rho == 0, cos_t0, sin_dec)
    dec = np.arcsin(np.clip(sin_dec, -1, 1))

    # Right ascension of each pixel
    numerator = xi_grid * sin_c
    denominator = rho * sin_t0 * cos_c - eta_grid * cos_t0 * sin_c
    denominator = np.where(rho == 0, 1, denominator)
    ra = phi0 + np.arctan2(numerator, denominator)
    ra = np.where(rho == 0, phi0, ra)

    theta = np.pi / 2 - dec  # colatitude
    phi = ra % (2 * np.pi)

    return hp.get_interp_val(m, theta.ravel(), phi.ravel()).reshape(n_y, n_x)


def _symmetric_ticks(size_deg):
    step = max(1, round(size_deg / 4))
    half = np.arange(0, size_deg / 2, step)
    return np.concatenate([-half[1:][::-1], half])


def _set_degree_ticks(ax):
    x_tick_degs = _symmetric_ticks(X_SIZE_DEG)
    x_tick_pix = (x_tick_degs + X_SIZE_DEG / 2) * 60 / RESOLUTION_ARCMIN
    ax.set_xticks(x_tick_pix)
    ax.set_xticklabels([f"{v:g}" for v in x_tick_degs])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("both")

    y_tick_degs = _symmetric_ticks(Y_SIZE_DEG)
    y_tick_pix = (y_tick_degs + Y_SIZE_DEG / 2) * 60 / RESOLUTION_ARCMIN
    ax.set_yticks(y_tick_pix)
    ax.set_yticklabels([f"{v:g}" for v in y_tick_degs])
    ax.yaxis.set_ticks_position("both")


def plot_halos(catalog_path, cosmo_args, ax):
    try:
        with h5py.File(catalog_path) as f:
            ra = f["RA(deg)"][:]
            dec = f["DEC(deg)"][:]
            z = 1.0 / f["a"][:] - 1.0
            theta_200c = f["theta200c"][:] * 60 * 180 / np.pi  # arcmin
    except FileNotFoundError:
        rng = np.random.default_rng(42)
        n = 100
        ra = CENTER_RA_DEG + rng.uniform(-X_SIZE_DEG / 2, X_SIZE_DEG / 2, n)
        dec = CENTER_DEC_DEG + rng.uniform(-Y_SIZE_DEG / 2, Y_SIZE_DEG / 2, n)
        z = rng.uniform(0, 2, n)
        theta_200c = rng.uniform(1, 3, n)  # arcmin

    # Forward gnomonic projection to pixel coordinates
    dec0 = np.radians(CENTER_DEC_DEG)
    phi0 = np.radians(CENTER_RA_DEG)
    phi = np.radians(ra)
    d = np.radians(dec)

    cos_c = np.sin(dec0) * np.sin(d) + np.cos(dec0) * np.cos(d) * np.cos(
        phi - phi0
    )
    xi = np.cos(d) * np.sin(phi - phi0) / cos_c
    eta = (
        np.cos(dec0) * np.sin(d)
        - np.sin(dec0) * np.cos(d) * np.cos(phi - phi0)
    ) / cos_c

    n_x = int(round(X_SIZE_DEG * 60 / RESOLUTION_ARCMIN))
    n_y = int(round(Y_SIZE_DEG * 60 / RESOLUTION_ARCMIN))
    x_pix = (
        (xi + np.radians(X_SIZE_DEG) / 2) / np.radians(X_SIZE_DEG) * (n_x - 1)
    )
    y_pix = (
        (eta + np.radians(Y_SIZE_DEG) / 2) / np.radians(Y_SIZE_DEG) * (n_y - 1)
    )
    r_pix = theta_200c / RESOLUTION_ARCMIN

    norm = Normalize(vmin=z.min(), vmax=z.max())
    cmap = plt.get_cmap(cosmo_args["cmap"])

    ax.set_xlim(-0.5, n_x - 0.5)
    ax.set_ylim(-0.5, n_y - 0.5)
    ax.set_aspect("equal")

    mask = (
        (x_pix >= -r_pix)
        & (x_pix < n_x + r_pix)
        & (y_pix >= -r_pix)
        & (y_pix < n_y + r_pix)
    )
    for xp, yp, rp, zp in zip(x_pix[mask], y_pix[mask], r_pix[mask], z[mask]):
        ax.add_patch(
            mpatches.Circle(
                (xp, yp),
                rp,
                edgecolor=cmap(norm(zp)),
                facecolor="none",
                linewidth=0.5,
            )
        )

    _set_degree_ticks(ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    plt.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        label=cosmo_args["label"],
    )


def plot_map(map_or_path, cosmo_args, ax):
    if isinstance(map_or_path, (str, Path)):
        try:
            m = hp.read_map(map_or_path)
        except FileNotFoundError:
            m = hp.read_map("./map_test.fits")
    else:
        m = map_or_path

    patch = gnomonic_patch(
        m,
        CENTER_RA_DEG,
        CENTER_DEC_DEG,
        X_SIZE_DEG,
        Y_SIZE_DEG,
        RESOLUTION_ARCMIN,
    )

    im = ax.imshow(
        patch,
        cmap=cosmo_args["cmap"],
        vmin=cosmo_args["vmin"],
        vmax=cosmo_args["vmax"],
        origin="lower",
    )

    _set_degree_ticks(ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    plt.colorbar(
        im, cax=cax, orientation="horizontal", label=cosmo_args["label"]
    )


def main():
    fig, axes = plt.subplots(1, 6, figsize=(18, 7), sharex=True, sharey=True)
    for ax, (path, cosmo_args) in zip(axes, MAP_CONFIGS):
        if cosmo_args.get("type") == "halos":
            plot_halos(path, cosmo_args, ax)
        else:
            plot_map(path, cosmo_args, ax)

    axes[0].set_ylabel(r"$\Delta$Dec [deg]")
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01)
    renderer = fig.canvas.get_renderer()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    top = max(ax.get_tightbbox(renderer).ymax for ax in axes) / fig_h_px
    fig.text(
        0.5,
        top + 0.005,
        r"$\Delta$RA [deg]",
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    plt.savefig("sky_components.png", dpi=150)  # , bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
