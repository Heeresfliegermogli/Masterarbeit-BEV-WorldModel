#!/bin/bash
# =============================================================================
# optimize_pdfs.sh — Font-Subsetting der Figuren-PDFs (nach dem Rendern)
# =============================================================================
# matplotlib bettet mit pdf.fonttype=42 die VOLLE Schrift je PDF ein (~700 kB
# pro Figur). Ghostscript subsettet sie auf die tatsaechlich benutzten Glyphen
# (~8x kleiner) — Text bleibt durchsuchbar, Vektoren bleiben Vektoren,
# Rasterpanels bleiben bei /prepress unangetastet.
# Aufruf aus dem Projektroot: bash scripts_render/optimize_pdfs.sh
# =============================================================================
set -u
TMP=$(mktemp -d); OK=0; SKIP=0
for f in visualizations/*.pdf; do
    out="$TMP/$(basename "$f")"
    if gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dSubsetFonts=true \
          -dEmbedAllFonts=true -dPDFSETTINGS=/prepress -dAutoRotatePages=/None \
          -sOutputFile="$out" "$f" 2>/dev/null; then
        # Verifikation: Datei nicht leer UND Textinhalt erhalten
        a=$(pdftotext "$f" - 2>/dev/null | tr -d '[:space:]' | wc -c)
        b=$(pdftotext "$out" - 2>/dev/null | tr -d '[:space:]' | wc -c)
        if [ -s "$out" ] && [ "$b" -ge $((a * 9 / 10)) ]; then
            mv "$out" "$f"; OK=$((OK+1))
        else
            SKIP=$((SKIP+1)); echo "  uebersprungen (Text-Diff): $(basename "$f")"
        fi
    else
        SKIP=$((SKIP+1)); echo "  uebersprungen (gs-Fehler): $(basename "$f")"
    fi
done
rm -rf "$TMP"
echo "optimiert: $OK | uebersprungen: $SKIP | Gesamt: $(du -ch visualizations/*.pdf | tail -1 | cut -f1)"
