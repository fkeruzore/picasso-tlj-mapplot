"""Render each sky component as a full-bleed page of a multipage PDF.

Reuses the panel-drawing code from ``main.py``; each page holds a single
component, sized to the aspect ratio of the patch, with no margins,
axes, ticks, labels, or colorbars.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from main import (
    MAP_CONFIGS,
    RESOLUTION_ARCMIN,
    X_SIZE_DEG,
    Y_SIZE_DEG,
    plot_halos,
    plot_map,
)

OUTPUT_PATH = "sky_components_multipage.pdf"
PAGE_HEIGHT_IN = 9.0
DPI = 300

N_X = int(round(X_SIZE_DEG * 60 / RESOLUTION_ARCMIN))
N_Y = int(round(Y_SIZE_DEG * 60 / RESOLUTION_ARCMIN))
PAGE_WIDTH_IN = PAGE_HEIGHT_IN * N_X / N_Y


def make_page(path, cosmo_args):
    """Draw one component on a figure that is entirely filled by the map."""
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN))
    ax = fig.add_axes([0, 0, 1, 1])

    if cosmo_args.get("type") == "halos":
        plot_halos(path, cosmo_args, ax)
        ax.set_facecolor("white")
    else:
        plot_map(path, cosmo_args, ax)

    # plot_halos/plot_map attach a colorbar through make_axes_locatable,
    # which both adds an axes and shrinks `ax`; undo both to get back to
    # full bleed.
    for extra in fig.get_axes():
        if extra is not ax:
            extra.remove()
    ax.set_axes_locator(None)
    ax.set_position([0, 0, 1, 1])

    # The page is already sized to the pixel grid, so stretching to fill it
    # is distortion-free.
    ax.set_aspect("auto")
    ax.set_xlim(-0.5, N_X - 0.5)
    ax.set_ylim(-0.5, N_Y - 0.5)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig


def main():
    with PdfPages(OUTPUT_PATH) as pdf:
        for path, cosmo_args in tqdm(MAP_CONFIGS):
            fig = make_page(path, cosmo_args)
            pdf.savefig(fig, dpi=DPI)
            plt.close(fig)


if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"
    main()
