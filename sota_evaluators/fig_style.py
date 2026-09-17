"""
Shared (color, marker) visual encoding for the 20 DEID techniques.

Design (replaces the old name-hash % 16 palette, which produced five
collision groups -- e.g. LDFA/FAMS/GANonymization all identical, and three
yellow tones): 10 hue families, each family used twice:

    dark variant of a hue  -> circle  (o)
    light variant of a hue -> square  (s)

so all 20 techniques have a unique (color, marker) pair and no two
similarly-hued techniques share a marker. Both 4-panel figures (and any
future one) MUST use this module so encodings stay identical across
figures:  AIDPro is always the dark-blue circle, etc.

The "aligned" reference stays a black open circle.
"""

_FAMILIES = [  # (dark, light), hue name
    ("#1f77b4", "#aec7e8"),   # blue
    ("#ff7f0e", "#ffbb78"),   # orange
    ("#2ca02c", "#98df8a"),   # green
    ("#d62728", "#ff9896"),   # red
    ("#9467bd", "#c5b0d5"),   # purple
    ("#8c564b", "#c49c94"),   # brown
    ("#e377c2", "#f7b6d2"),   # magenta
    ("#c9a227", "#efe2a8"),   # gold (gray #7f7f7f reserved for aligned Original)
    ("#bcbd22", "#dbdb8d"),   # olive
    ("#17becf", "#9edae5"),   # cyan
]

_DARK = ["AIDPro", "FALCO", "G2Face", "LDFA", "NullFace", "GANonymization",
         "IPFA", "AnonNET", "DiffPrivate", "DeepPrivacy2"]
_LIGHT = ["AMT-GAN", "FADM", "CLEANIR", "PRO-Face", "RiDDLE", "DeepPrivacy",
          "FAMS", "CPP-DeID", "RP", "iFADIT"]

STYLE = {}
for (dark, light), n_dark, n_light in zip(_FAMILIES, _DARK, _LIGHT):
    STYLE[n_dark] = (dark, "o")
    STYLE[n_light] = (light, "s")


def style_for(name):
    """(color, marker) for a technique; aligned reference = black circle."""
    if name == "aligned":
        return "#000000", "o"
    return STYLE[name]


def pair_for(name):
    """(dark, light) family colors for a technique (both variants share the
    family); aligned reference = neutral gray pair."""
    if name == "aligned":
        return ("#777777", "#cccccc")
    own, _marker = STYLE[name]
    for (d, l) in _FAMILIES:
        if own in (d, l):  # resolve via either variant of the family
            return (d, l)
    return (own, own)


def color_for(name):
    return style_for(name)[0]


def marker_for(name):
    return style_for(name)[1]
