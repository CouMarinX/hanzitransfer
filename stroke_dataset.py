import json
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen


def glyph_to_strokes(glyph, scale=1.0):
    """Convert a glyph to a list of stroke commands."""
    pen = RecordingPen()
    glyph.draw(pen)
    strokes = []
    current = []
    for cmd, coords in pen.value:
        if cmd == "moveTo":
            if current:
                strokes.append(current)
                current = []
            x, y = coords[0]
            current.append({"op": "M", "points": [[x * scale, y * scale]]})
        elif cmd == "lineTo":
            x, y = coords[0]
            current.append({"op": "L", "points": [[x * scale, y * scale]]})
        elif cmd in ("qCurveTo", "curveTo"):
            pts = [[x * scale, y * scale] for x, y in coords]
            op = "Q" if cmd == "qCurveTo" else "C"
            current.append({"op": op, "points": pts})
        elif cmd == "closePath":
            current.append({"op": "Z", "points": []})
            strokes.append(current)
            current = []
    if current:
        strokes.append(current)
    return strokes


def extract_strokes(font_path, chars):
    """Extract stroke data for a set of characters from a font file."""
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    data = {}
    for ch in chars:
        glyph_name = cmap.get(ord(ch))
        if not glyph_name:
            continue
        glyph = glyph_set[glyph_name]
        data[ch] = glyph_to_strokes(glyph)
    return data


def save_dataset(font_path, chars, output_path):
    data = extract_strokes(font_path, chars)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _chars_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return list(dict.fromkeys(content.strip()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract stroke vectors from font")
    parser.add_argument("font", help="Path to TTF/OTF font file")
    parser.add_argument("output", help="Output JSON file")
    parser.add_argument("--chars", dest="chars", help="Characters to extract")
    parser.add_argument("--textfile", dest="textfile", help="Text file containing characters")
    args = parser.parse_args()

    if args.chars:
        chars = list(args.chars)
    elif args.textfile:
        chars = _chars_from_file(args.textfile)
    else:
        parser.error("Provide --chars or --textfile")

    save_dataset(args.font, chars, args.output)
