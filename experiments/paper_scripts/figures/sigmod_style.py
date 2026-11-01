"""The SIGMOD (acmart sigconf) figure style of the paper-figures rules: the page
geometry, upright Linux Biolinum at 9 pt on every element, TrueType embedding,
and each system's bar style, so every figure that imports it matches.

    from sigmod_style import COLUMN, TEXT, FAIL, bar_style, save

Biolinum ships as OpenType/CFF, which matplotlib embeds under a TrueType label
(pdffonts: "Mismatch between font type and embedded font file"), so a TrueType
copy is built once into ~/.cache/paper-figures/fonts under the family name
"Linux Biolinum O TTF" and every figure plots with that.  ``save`` writes the
PDF and fails unless every embedded font is upright Biolinum.

Colours: PageFlow the saturated blue, each baseline a colour of its own,
DuckDB black; the configurations that only ablations show are greys
(baselines) or lighter blues (PageFlow with mechanisms left out).  Every bar
style also has a hatch or a distinct lightness, so a figure survives
greyscale.
"""
import os
import re
import shutil
import subprocess
import zlib

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

COLUMN = 3.335      # \columnwidth, in
TEXT = 7.03         # \textwidth, in
PT = 9
FONT_OTF = "Linux Biolinum O"           # installed family (OpenType/CFF)
FONT = FONT_OTF + " TTF"                # TrueType copy built below; plot with this
PDF_KEY = "Biolinum"                    # substring of the embedded PDF font names
TEXLIVE = ("LinBiolinum_R.otf", "LinBiolinum_RB.otf")
CACHE = os.path.expanduser("~/.cache/paper-figures/fonts")

FAIL = "#d62728"    # failure text and marks: one red for every system

# name: (facecolor, hatch)
SYSTEMS = {
    "PageFlow":           ("#2a78d6", ""),
    "Sirius":             ("#eb6834", "////"),
    "cudf-Polars":        ("#1baf7a", "\\\\\\\\"),
    "Spark-RAPIDS":       ("#eda100", "xxxx"),
    "DuckDB":             ("#111111", ""),
    "UVM":                ("#8c8c8c", "xxxx"),
    "Free list":          ("#d9d9d9", "\\\\\\\\"),
    "Free list reserved": ("#a6a6a6", "////"),
}


# Line plots: SYSTEMS's colour, and a marker and line style per system.  DuckDB
# (the CPU reference) is solid black, heavier, larger and on top.
LINES = {
    "PageFlow":     dict(color="#2a78d6", marker="o", linestyle="-"),
    "Sirius":       dict(color="#eb6834", marker="D", linestyle="-"),
    "cudf-Polars":  dict(color="#1baf7a", marker="s", linestyle="--"),
    "Spark-RAPIDS": dict(color="#eda100", marker="^", linestyle="-."),
    "DuckDB":       dict(color="#111111", marker="v", linestyle="-", linewidth=1.9,
                         markersize=5.2, zorder=6),
}
INK2 = "#52514e"    # secondary text: region labels
INK3 = "#78776f"    # rules
# A point at which a query failed: a red x over a white one, which lifts it off any line.
FAIL_MARK = dict(ls="none", marker="x", ms=5.5, mew=1.6, color=FAIL)
FAIL_HALO = dict(ls="none", marker="x", ms=6.3, mew=3.0, color="white")


def bar_style(name, color=None, hatch=None):
    """``ax.bar`` kwargs for system ``name`` (``color``/``hatch`` override)."""
    c, h = SYSTEMS.get(name, ("#a6a6a6", ""))
    return dict(facecolor=color or c, hatch=h if hatch is None else hatch,
                edgecolor="black", linewidth=0.5)


def _to_truetype(src, dst, family):
    """Copy a font with TrueType outlines under a new family name."""
    from fontTools.ttLib import TTFont, newTable
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    f = TTFont(src)
    if "CFF " in f:
        order, gs = f.getGlyphOrder(), f.getGlyphSet()
        glyf = newTable("glyf")
        glyf.glyphOrder, glyf.glyphs = order, {}
        for g in order:
            pen = TTGlyphPen(gs)
            gs[g].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
            glyf.glyphs[g] = pen.glyph()
        f["loca"], f["glyf"] = newTable("loca"), glyf
        del f["CFF "]
        if "VORG" in f:
            del f["VORG"]
        glyf.compile(f)
        for g, gl in glyf.glyphs.items():
            if hasattr(gl, "xMin"):
                f["hmtx"][g] = (f["hmtx"][g][0], gl.xMin)
        maxp = f["maxp"] = newTable("maxp")
        maxp.tableVersion = 0x00010000
        maxp.maxZones, maxp.maxTwilightPoints, maxp.maxStorage = 1, 0, 0
        maxp.maxFunctionDefs = maxp.maxInstructionDefs = 0
        maxp.maxStackElements = maxp.maxSizeOfInstructions = 0
        maxp.maxComponentElements = max(len(getattr(g, "components", []) or [])
                                        for g in glyf.glyphs.values())
        maxp.compile(f)
        post = f["post"]
        post.formatType, post.extraNames, post.mapping = 2.0, [], {}
        post.glyphOrder = order
        f.sfntVersion = "\x00\x01\x00\x00"
    for rec in f["name"].names:
        if rec.nameID in (1, 16):
            rec.string = family
    f.save(dst)


def _register_font():
    srcs = [e.fname for e in fm.fontManager.ttflist
            if e.name == FONT_OTF and e.style == "normal"]
    if not srcs and shutil.which("kpsewhich"):     # TeX Live's copy (acmart's libertine)
        for n in TEXLIVE:
            p = subprocess.run(["kpsewhich", n], capture_output=True, text=True).stdout.strip()
            if p:
                srcs.append(p)
    if not srcs:
        raise SystemExit(f"{FONT_OTF} not installed: apt install fonts-linuxlibertine "
                         "&& rm -rf ~/.cache/matplotlib")
    os.makedirs(CACHE, exist_ok=True)
    for src in sorted(set(srcs)):
        dst = os.path.join(CACHE, os.path.splitext(os.path.basename(src))[0] + ".ttf")
        if not os.path.exists(dst):
            _to_truetype(src, dst, FONT)
        fm.fontManager.addfont(dst)
    if FONT not in {e.name for e in fm.fontManager.ttflist}:
        raise SystemExit(f"could not register {FONT}")


_register_font()

mpl.rcParams.update({
    # Fonts: upright Linux Biolinum, 9 pt everywhere.
    "font.family": "sans-serif",
    "font.sans-serif": [FONT],
    "font.style": "normal",
    "font.size": PT,
    "axes.labelsize": PT,
    "axes.titlesize": PT,
    "legend.fontsize": PT,
    "xtick.labelsize": PT,
    "ytick.labelsize": PT,
    "figure.titlesize": PT,
    "text.usetex": False,
    # Math: every alphabet is upright Biolinum.
    "mathtext.fontset": "custom",
    "mathtext.default": "rm",
    "mathtext.rm": FONT,
    "mathtext.it": FONT,
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.sf": FONT,
    "mathtext.tt": FONT,
    "mathtext.cal": FONT,
    "mathtext.fallback": "stixsans",
    # Spines and grid
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linewidth": 0.4,
    "grid.color": "#d0d0d0",
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
    # Ticks
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Lines, markers, hatches
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "lines.markeredgewidth": 0.6,
    "hatch.linewidth": 0.5,
    # Legend
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.2,
    # Output: TrueType embedding.  The page is exactly figsize, so
    # \includegraphics[width=\columnwidth] is a no-op: no tight bbox (it crops
    # to the ink); figures use layout="constrained".
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": None,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "figure.dpi": 150,
})
if "mathtext.bfit" in mpl.rcParams:   # matplotlib >= 3.8
    mpl.rcParams["mathtext.bfit"] = f"{FONT}:bold"


def check_fonts(pdf):
    """Fail unless every embedded font is upright Biolinum (or the upright STIX
    fallback), embedded as real TrueType.  Read from the PDF itself, whose font
    dictionaries matplotlib writes uncompressed: no Type 3 font, no CFF font
    file (an OpenType FontFile3, or a FontFile2 whose data is CFF, the
    "Mismatch" pdffonts reports), and every font's name."""
    with open(pdf, "rb") as f:
        data = f.read()
    assert b"/ObjStm" not in data, f"{pdf}: object streams; its fonts cannot be read"
    assert not re.search(rb"/Subtype\s*/Type3\b", data), f"{pdf}: Type 3 font embedded"
    assert b"/FontFile3" not in data, f"{pdf}: CFF font embedded; plot with {FONT}"
    for num in re.findall(rb"/FontFile2\s+(\d+)\s+0\s+R", data):
        m = re.search(rb"(?<!\d)" + num + rb"\s+0\s+obj\s*<<(.*?)>>\s*stream\r?\n",
                      data, re.S)
        assert m, f"{pdf}: font file {num.decode()} not found"
        font = data[m.end():]
        if b"/FlateDecode" in m.group(1):
            font = zlib.decompressobj().decompress(font)
        assert font[:4] != b"OTTO", \
            f"{pdf}: CFF font embedded under a TrueType label; plot with {FONT}"
    names = sorted({n.decode().split("+")[-1] for n in
                    re.findall(rb"/BaseFont\s*/([^\s/<>\[\]()]+)", data)})
    bad = [n for n in names
           if not (PDF_KEY in n or n.startswith("STIX"))
           or re.search(r"Ital|Oblique|LinBiolinum[OT]B?I$", n)]
    assert not bad, f"{pdf}: serif, italic, script or other stray fonts embedded: {bad}"
    assert any(PDF_KEY in n for n in names), f"{pdf}: Biolinum not embedded: {names}"


def save(fig, path):
    """The paper's copy: a vector PDF at ``path``, its fonts checked.  The
    directory is made if missing (results/ holds no tracked file)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    check_fonts(path)
    print(f"  wrote {path}")
