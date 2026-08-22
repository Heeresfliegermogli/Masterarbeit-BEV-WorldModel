#!/usr/bin/env python3
# =============================================================================
# thesis_style.py — gemeinsame Typografie ALLER Thesis-Figuren
# =============================================================================
# LOGIK (Redaktions-Auftrag 12.08.): Jede Figur wird mit einer NATUERLICHEN
# Breite von 17 cm (= 6.69 in) gerendert. Wird sie im Dokument schmaler
# eingebunden, skaliert LaTeX sie herunter — die Schrift wuerde dadurch zu
# klein. Deshalb werden die Font-Groessen mit 1/scale vorgehalten
# (scale = Einbaubreite / 17 cm), sodass im DRUCK ueberall dieselbe
# effektive Punktgroesse ankommt (Figuren-Basis 9 pt, Ticks 8, Titel 10).
#
# Verwendung:  from thesis_style import apply_style, S, FIG
#              apply_style(17.0)      # oder 14.0 / 10.0 je nach Einbau
#              fig, ax = plt.subplots(figsize=FIG(11.5, 4.6))
# Bei mehreren Figuren je Skript apply_style() VOR jeder Figur aufrufen.
# =============================================================================
import matplotlib

WIDTH_IN = 6.69          # 17 cm natuerliche Breite aller Figuren
_scale = 1.0

BASE = {"font": 9.0, "title": 10.0, "label": 9.0, "tick": 8.0, "legend": 8.0}
MIN_PT = 7.0             # kleinste gedruckte Schrift


def apply_style(width_cm=17.0):
    """rcParams fuer eine im Dokument width_cm breit eingebundene Figur."""
    global _scale
    _scale = float(width_cm) / 17.0
    matplotlib.rcParams.update({
        "font.size":        BASE["font"] / _scale,
        "axes.titlesize":   BASE["title"] / _scale,
        "axes.labelsize":   BASE["label"] / _scale,
        "xtick.labelsize":  BASE["tick"] / _scale,
        "ytick.labelsize":  BASE["tick"] / _scale,
        "legend.fontsize":  BASE["legend"] / _scale,
        "figure.titlesize": BASE["title"] / _scale,
        "pdf.fonttype":     42,      # TrueType einbetten (kein Type-3)
        "ps.fonttype":      42,
        "savefig.dpi":      300,
    })


def S(pt):
    """Explizite fontsize-Werte der Alt-Skripte aufs Druckniveau (x0.85),
    mit 7-pt-Untergrenze."""
    return round(max(MIN_PT, float(pt) * 0.85) / _scale, 2)


_UML = {"Kapazitaet": "Kapazität", "duenn": "dünn", "gross": "groß",
        "geloest": "gelöst", "Praediktion": "Prädiktion", "ueber": "über",
        "Straenge": "Stränge", "Flaeche": "Fläche", "haelt": "hält"}


def de(s):
    """Normiert ASCII-Umlaute in Labels, die aus DATEN-Artefakten (CSV/JSON)
    stammen — nur fuer die Anzeige; die Artefakte bleiben unveraendert."""
    for a, b in _UML.items():
        s = s.replace(a, b)
    return s


def FIG(w, h):
    """Normiert auf 17 cm natuerliche Breite, Seitenverhaeltnis bleibt."""
    return (WIDTH_IN, WIDTH_IN * float(h) / float(w))


# =============================================================================
# LiDAR-Panel-Inversion (Render-Auftrag 11.08.)
# =============================================================================
# tools/visualize.py rendert helle Punktwolken auf SCHWARZ. Fuer den Druck
# (und neben den uebrigen hellen Thesis-Figuren) wird invertiert: dunkle
# Punkte auf WEISS. Die farbigen Boxen duerfen NICHT mitinvertiert werden
# (sonst kippt die Farbcodierung) -> Trennung ueber die Farbsaettigung:
# unbunte Pixel = Punktwolke/Hintergrund (invertieren), gesaettigte Pixel =
# Boxen (Farbe erhalten, leicht abgedunkelt fuer Kontrast auf Weiss).
SAT_THRESH = 18 / 255.0   # darunter gilt ein Pixel als unbunt
DENSITY = 0.82            # <1 laesst dichte Bereiche dunkelgrau statt tiefschwarz
BOX_DARKEN = 0.88         # Boxfarben leicht abdunkeln (Kontrast auf Weiss)


def invert_lidar(img):
    """RGB-Array [0..1] -> invertierte Darstellung (dunkel auf weiss),
    Box-Farben erhalten. Erwartet float-Bild wie von mpimg.imread."""
    import numpy as np
    a = np.asarray(img, dtype=float)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.max() > 1.001:
        a = a / 255.0
    mx = a.max(axis=2)
    sat = mx - a.min(axis=2)
    colored = sat > SAT_THRESH
    out = 1.0 - a * DENSITY                    # unbunt: invertiert
    box = np.clip(a * BOX_DARKEN, 0, 1)        # bunt: Originalfarbe, dunkler
    out[colored] = box[colored]
    return np.clip(out, 0, 1)


# =============================================================================
# Deutsche Dezimalkommas (Render-Auftrag 22.08.) — ZENTRAL fuer alle Figuren
# =============================================================================
# Ein Patch auf Text.set_text erwischt ALLE gerenderten Zahlen (Achsen-Ticks,
# Annotationen, Balkenwerte, Legenden), ohne dass die einzelnen Skripte ihre
# f-Strings aendern muessen: Text.__init__ ruft set_text auf, Tick-Labels
# laufen beim Draw ueber set_text. Ersetzt NUR Punkt zwischen zwei Ziffern.
import re as _re
import matplotlib.text as _mtext

_DEC_PUNKT = _re.compile(r"(?<=\d)\.(?=\d)")
_set_text_orig = _mtext.Text.set_text

def _set_text_dezimalkomma(self, s):
    if isinstance(s, str):
        s = _DEC_PUNKT.sub(",", s)
    return _set_text_orig(self, s)

_mtext.Text.set_text = _set_text_dezimalkomma


# =============================================================================
# Explizite kaufmaennische Rundung (Render-Auftrag 22.08., RUNDUNG)
# =============================================================================
# f"{x:.3f}" rundet auf dem BINAERWERT (0.6295 ist binaer knapp darunter ->
# "0.629"). fmt() geht ueber die Dezimal-Kurzdarstellung + ROUND_HALF_UP,
# damit Labels mit den kaufmaennisch gerundeten Berichtswerten uebereinstimmen.
from decimal import Decimal as _Decimal, ROUND_HALF_UP as _RHU

def fmt(x, nd=3, sign=False):
    q = _Decimal(str(float(x))).quantize(_Decimal("1." + "0" * nd), rounding=_RHU)
    return f"{q:+}" if sign else f"{q}"
