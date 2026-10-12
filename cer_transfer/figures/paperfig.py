"""One width for every paper figure.

The manuscript is letterpaper with 1 in margins: \\textwidth = 6.5 in. Paper
(PDF) figures are drawn at exactly this width and saved without a tight
bounding box, so LaTeX includes them at their natural size, all with the same
width and the same font size. Talk (PNG) figures keep the tight bounding box.
"""

PAPER_W = 6.5  # inches = \textwidth


def bbox(ext):
    """bbox_inches argument for savefig: exact size for the paper PDF."""
    return None if ext == "pdf" else "tight"
