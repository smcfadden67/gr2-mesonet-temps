#!/usr/bin/env python3
"""
Build GR2Analyst camera placefiles for Indiana (511in.org) and Illinois
(travelmidwest.com).

Every snapshot is downloaded, shrunk to a small thumbnail, and packed into
sprite sheets. GR2 allows only 32 image files per placefile, but each sprite
sheet holds hundreds of thumbnails, so one regional placefile can show every
camera in that region.

Output (in --out-dir):
    cams_<region>.txt          one placefile per region
    img/<region>_<n>.jpg       sprite sheets referenced by the placefiles

Usage:
    python build_cameras.py --base-url https://raw.githubusercontent.com/USER/REPO/cameras
    python build_cameras.py --base-url ... --demo   # offline test with fake cameras
Requires Pillow (pip install pillow).
"""

import argparse
import io
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from PIL import Image, ImageDraw

# ------------------------------------------------------------------ settings

THUMB_W, THUMB_H = 144, 96          # base (1x) thumbnail size, 3:2; also used for scoring
JPEG_QUALITY = 72
DOWNLOAD_THREADS = 24
DOWNLOAD_TIMEOUT = 15
REFRESH_MINUTES = 5

# Priority tiers: (grid cell in degrees, GR Threshold nm, thumbnail size, sheet cols x rows).
# A camera lands in the first tier where it is the best-scoring camera in its
# grid cell. Each tier has a fixed thumbnail size at every zoom:
#   top priority  4x area (2x width)    appears at 60 nm
#   decent        2x area (~1.4x width) appears at 25 nm
#   rest          1x                    appears at 12 nm
# Lower tiers are written first so GR2 draws the top tier over them.
THUMB_TIERS = [
    (0.20, 60, (288, 192), (7, 10)),    # 2016 x 1920 sheets, 70 per sheet
    (0.07, 25, (204, 136), (10, 15)),   # 2040 x 2040 sheets, 150 per sheet
    (None, 12, (144, 96), (14, 21)),    # 2016 x 2016 sheets, 294 per sheet
]
SRC_W, SRC_H = 288, 192             # keep each snapshot at the largest tier size
# Small camera-shaped markers (hover for the name) show at this zoom and closer.
MARKER_THRESHOLD = 60
CAM_ICON_BODY = (45, 45, 45)     # dark gray
CAM_ICON_EDGE = (110, 110, 110)  # faint outline so it shows on black
CAM_ICON_LENS = (20, 20, 20)

# Regions: (key, title, state, lat_min, lat_max, lon_min, lon_max).
# A camera goes in the FIRST region whose box contains it.
REGIONS = [
    ("indy",       "Indianapolis cameras",      "IN", 39.45, 40.15, -86.60, -85.70),
    ("in_north",   "Northern Indiana cameras",  "IN", 40.60, 42.00, -88.50, -84.50),
    ("in_central", "Central Indiana cameras",   "IN", 39.00, 40.60, -88.50, -84.50),
    ("in_south",   "Southern Indiana cameras",  "IN", 37.50, 39.00, -88.50, -84.50),
    ("chicago",    "Chicago metro cameras",     "IL", 41.35, 42.55, -88.75, -87.40),
    ("il_north",   "Northern Illinois cameras", "IL", 40.30, 42.60, -91.60, -87.00),
    ("il_south",   "Southern Illinois cameras", "IL", 36.90, 40.30, -91.60, -87.00),
    ("milwaukee",  "Milwaukee metro cameras",   "WI", 42.75, 43.40, -88.50, -87.70),
    ("wi_south",   "Southern Wisconsin cameras", "WI", 42.40, 44.00, -90.70, -86.80),
    ("wi_other",   "Wisconsin cameras (other)", "WI", 44.00, 47.20, -92.90, -86.80),
]

# ------------------------------------------------------------------ camera quality
# Each camera gets a "view quality" score from its daytime snapshots:
#   - vegetation: share of green/grassy pixels in the lower 2/3 of the frame
#     (close-in views of the roadside have lots; wide highway views have little)
#   - brightness: well-lit images score higher
#   - sky: views dominated by sky score lower
# Scores are averaged over many runs (saved in scores.json on the published
# branch) so the ranking is stable and night/IR images don't reshuffle it.
# The best-scoring camera in each grid cell gets the zoomed-out thumbnail slot.
SCORE_WEIGHTS = dict(veg=0.60, bright=0.25, sky=0.15)
SCORE_SMOOTHING = 0.15        # weight of each new daytime snapshot in the average
DAYLIGHT_MIN_BRIGHTNESS = 0.22
DAYLIGHT_MIN_SATURATION = 0.06  # below this the image is IR/grayscale (night)

UA = "gr2-camera-placefile (GR2Analyst placefile builder)"

# ------------------------------------------------------------------ sources

IN_QUERY = (
    "query MapFeatures($input: MapFeaturesArgs!) { mapFeaturesQuery(input: $input) "
    "{ mapFeatures { title uri features { geometry } __typename "
    "... on Camera { active views(limit: 5) { uri ... on CameraView { url } category } } } "
    "error { message type } } }"
)


def http(url, data=None, headers=None, timeout=60):
    h = {"User-Agent": UA}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_indiana():
    body = json.dumps({"query": IN_QUERY, "variables": {"input": {
        "north": 41.8, "south": 37.7, "east": -84.7, "west": -88.2, "zoom": 16,
        "layerSlugs": ["normalCameras"], "nonClusterableUris": ["dashboard"]}}}).encode()
    raw = http("https://511in.org/api/graphql", body,
               {"Content-Type": "application/json", "Origin": "https://511in.org",
                "Referer": "https://511in.org/"})
    feats = json.loads(raw)["data"]["mapFeaturesQuery"]["mapFeatures"]
    cams = []
    for f in feats:
        if f.get("__typename") != "Camera" or not f.get("active"):
            continue
        url = next((v.get("url") for v in f.get("views") or []
                    if str(v.get("url", "")).startswith("http")), None)
        if not url:
            continue
        lon, lat = f["features"][0]["geometry"]["coordinates"][:2]
        title = f.get("title", "")
        road, _, rest = title.partition(":")
        name = rest.strip() or title
        # Drop INDOT's internal ID prefix like "1-465-010-6-1 " when present.
        parts = name.split(" ", 1)
        if len(parts) == 2 and parts[0].count("-") >= 3:
            name = parts[1]
        cams.append(dict(state="IN", lat=lat, lon=lon,
                         name=f"{road.strip()}: {name.strip()}", src_name="INDOT", url=url))
    return cams


def fetch_travelmidwest():
    """Illinois and Wisconsin cameras from Travel Midwest."""
    body = json.dumps({"bbox": [-92.9, 36.9, -86.8, 47.2]}).encode()
    raw = http("https://travelmidwest.com/lmiga/cameraMap.json", body,
               {"Content-Type": "application/json", "Accept": "application/json",
                "Origin": "https://travelmidwest.com",
                "Referer": "https://travelmidwest.com/"})
    cams = []
    for f in json.loads(raw).get("features", []):
        p = f.get("properties", {})
        st = str(p.get("id", ""))[:2]
        if st not in ("IL", "WI") or p.get("dis"):
            continue
        urls = [u for u in p.get("remUrls") or [] if str(u).startswith("http")]
        if not urls:
            continue
        lon, lat = f["geometry"]["coordinates"][:2]
        dirs = p.get("dirs") or []
        d = f" ({dirs[0]})" if dirs and dirs[0] not in ("NONE", "") else ""
        cams.append(dict(state=st, lat=lat, lon=lon,
                         name=f"{p.get('locDesc', p['id'])}{d}",
                         src_name=p.get("src", "Travel Midwest"), url=urls[0]))
    return cams


# ------------------------------------------------------------------ images

def make_thumb(src, w, h):
    im = src.copy()
    im.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    # thin border so thumbnails stand out on the radar
    ImageDraw.Draw(canvas).rectangle([0, 0, w - 1, h - 1], outline=(200, 200, 200))
    return canvas


def set_thumbs(cam, data):
    src = Image.open(io.BytesIO(data)).convert("RGB")
    src.thumbnail((SRC_W, SRC_H), Image.LANCZOS)   # keep memory small
    cam["src"] = src
    cam["thumb"] = make_thumb(src, THUMB_W, THUMB_H)


def make_camera_icon(path):
    """Tiny dark camera glyph (body, viewfinder bump, lens) with transparency."""
    W, H = 12, 9
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    body, edge, lens = CAM_ICON_BODY + (255,), CAM_ICON_EDGE + (255,), CAM_ICON_LENS + (255,)
    d.rectangle([3, 0, 5, 1], fill=edge)                     # viewfinder bump
    d.rectangle([0, 2, W - 1, H - 1], fill=body, outline=edge)
    d.ellipse([4, 3, 8, 7], fill=lens, outline=edge)          # lens
    im.save(path, "PNG")
    return W, H


def cam_key(cam):
    return f'{cam["state"]}|{cam["lat"]:.4f}|{cam["lon"]:.4f}|{cam["name"]}'


def image_metrics(thumb):
    """Return (vegetation, brightness, sky, saturation) in 0..1 from a thumbnail."""
    hsv = thumb.convert("HSV")
    w, h = hsv.size
    raw = hsv.tobytes()
    px = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
    n = len(px)
    bright = sum(p[2] for p in px) / (255 * n)
    sat = sum(p[1] for p in px) / (255 * n)
    top_end = h // 3
    veg = sky = 0
    lower_n = upper_n = 0
    for i, (hh, ss, vv) in enumerate(px):
        y = i // w
        if y < top_end:
            upper_n += 1
            # sky: bright, low-saturation or bluish
            if vv > 140 and (ss < 50 or 130 <= hh <= 175):
                sky += 1
        else:
            lower_n += 1
            # vegetation: green to yellow-green hues, some saturation and light
            # (PIL hue 0-255: ~35-110 covers yellow-green through green)
            if 35 <= hh <= 110 and ss > 40 and vv > 40:
                veg += 1
    return veg / max(lower_n, 1), bright, sky / max(upper_n, 1), sat


def update_score(cam, history):
    veg, bright, sky, sat = image_metrics(cam["thumb"])
    key = cam_key(cam)
    prev = history.get(key)
    daylight = bright >= DAYLIGHT_MIN_BRIGHTNESS and sat >= DAYLIGHT_MIN_SATURATION
    if daylight:
        now = (SCORE_WEIGHTS["veg"] * min(veg / 0.50, 1.0)
               + SCORE_WEIGHTS["bright"] * min(bright / 0.6, 1.0)
               + SCORE_WEIGHTS["sky"] * (1 - sky))
        score = now if prev is None else prev + SCORE_SMOOTHING * (now - prev)
    else:
        score = prev   # keep the daytime average; don't learn from night images
    cam["score"] = score if score is not None else 0.0
    if score is not None:
        history[key] = round(score, 4)


def load_history(base_url):
    try:
        return json.loads(http(f"{base_url}/scores.json?t={int(time.time())}", timeout=20))
    except Exception as e:
        print(f"no previous scores ({e}); starting fresh")
        return {}


def download_thumb(cam):
    for attempt in range(2):
        try:
            set_thumbs(cam, http(cam["url"], timeout=DOWNLOAD_TIMEOUT))
            return cam
        except Exception:
            time.sleep(1)
    cam["thumb"] = None
    return cam


def demo_cameras(n_per_region=120):
    random.seed(1)
    cams = []
    for key, _, st, la0, la1, lo0, lo1 in REGIONS:
        for i in range(n_per_region):
            c = dict(state=st, lat=random.uniform(la0, la1), lon=random.uniform(lo0, lo1),
                     name=f"Demo {key} #{i}", src_name="demo", url="")
            im = Image.new("RGB", (352, 240), tuple(random.randint(0, 255) for _ in range(3)))
            ImageDraw.Draw(im).text((10, 10), f"{key} {i}", fill=(255, 255, 255))
            buf = io.BytesIO()
            im.save(buf, "JPEG")
            set_thumbs(c, buf.getvalue())
            cams.append(c)
    return cams


def make_thumb_from_image(im):
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    c = {}
    set_thumbs(c, buf.getvalue())
    return c["thumb"]


# ------------------------------------------------------------------ layout

def region_of(cam):
    for key, _, st, la0, la1, lo0, lo1 in REGIONS:
        if cam["state"] == st and la0 <= cam["lat"] <= la1 and lo0 <= cam["lon"] <= lo1:
            return key
    return None


def dedupe(cams, tol=0.0015):
    """Drop cameras sitting on top of an earlier one (same pole, 2nd view)."""
    kept, seen = [], set()
    for c in cams:
        k = (round(c["lat"] / tol), round(c["lon"] / tol))
        if k in seen:
            continue
        seen.add(k)
        kept.append(c)
    return kept


def assign_tiers(cams):
    # Best view quality first, so it wins its grid cell in the zoomed-out tiers.
    remaining = sorted(cams, key=lambda c: -c.get("score", 0.0))
    out = []
    for cell, thr, size, layout in THUMB_TIERS:
        if cell is None:
            out.append((thr, size, layout, remaining))
            break
        taken, rest, cells = [], [], set()
        for c in remaining:
            k = (int((c["lat"] + 90) // cell), int((c["lon"] + 180) // cell))
            (rest if k in cells else taken).append(c)
            cells.add(k)
        out.append((thr, size, layout, taken))
        remaining = rest
    return out


def clean(s):
    return str(s).replace('"', "'").replace("\n", " ").strip()


def pack_sheets(ordered, w, h, cols, rows, key, tag, first_file,
                out_dir, base_url, epoch):
    """Pack each camera's thumbnail (at w x h) into sprite sheets. Returns
    IconFile lines and records (file, icon) on each camera as c['ref']."""
    per = cols * rows
    lines = []
    for n in range(-(-len(ordered) // per)):
        chunk = ordered[n * per:(n + 1) * per]
        used_rows = -(-len(chunk) // cols)
        sheet = Image.new("RGB", (w * cols, h * used_rows))
        for idx, c in enumerate(chunk):
            r, col = divmod(idx, cols)
            sheet.paste(make_thumb(c["src"], w, h), (col * w, r * h))
            c["ref"] = (first_file + n, idx + 1)
        fname = f"{key}_{tag}{n + 1}.jpg"
        sheet.save(os.path.join(out_dir, "img", fname), "JPEG",
                   quality=JPEG_QUALITY, optimize=True)
        # ?t= busts caches so GR2 and GitHub serve the new sheet each run.
        lines.append(f'IconFile: {first_file + n}, {w}, {h}, {w // 2}, {h // 2}, '
                     f'"{base_url}/img/{fname}?t={epoch}"')
    return lines


def build_region(key, title, cams, out_dir, base_url, stamp, epoch):
    tiers = assign_tiers(cams)

    # GR2 allows 32 image files per placefile (1 camera icon + sheets).
    def files_needed(ts):
        return 1 + sum(-(-len(g) // (lay[0] * lay[1])) for _, _, lay, g in ts)
    while files_needed(tiers) > 32:
        t, sz, lay, g = tiers[-1]
        if not g:
            tiers = tiers[:-1]
            continue
        tiers[-1] = (t, sz, lay, g[:-1])

    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    lines = [
        f"; {title} - snapshots shrunk and packed by build_cameras.py, built {stamp}",
        "; Sources: INDOT (511in.org), Travel Midwest (travelmidwest.com), WisDOT",
        f"Title: {title} ({stamp})",
        f"Refresh: {REFRESH_MINUTES}",
        'Font: 1, 11, 1, "Arial"',
        "",
    ]
    iw, ih = make_camera_icon(os.path.join(out_dir, "img", "camera.png"))
    lines.append(f'IconFile: 1, {iw}, {ih}, {iw // 2}, {ih // 2}, "{base_url}/img/camera.png"')
    next_file, n_sheets = 2, 0
    for t_i, (thr, (w, h), (cols, rows), group) in enumerate(tiers):
        group.sort(key=lambda c: c.get("score", 0.0))    # best drawn last
        fl = pack_sheets(group, w, h, cols, rows, key, f"t{t_i + 1}_", next_file,
                         out_dir, base_url, epoch)
        lines += fl
        next_file += len(fl)
        n_sheets += len(fl)
    lines.append("")

    ordered = [c for *_, g in tiers for c in g]
    # Camera markers first, so thumbnails draw over them.
    lines.append(f"Threshold: {MARKER_THRESHOLD}")
    for c in ordered:
        lines.append(f'Icon: {c["lat"]:.5f}, {c["lon"]:.5f}, 0, 1, 1, '
                     f'"{clean(c["name"])}\\n{clean(c["src_name"])} - snapshot {stamp}"')
    lines.append("")

    # Lowest-priority tier first, top tier last, so priority cameras sit on top.
    for thr, _, _, group in reversed(tiers):
        if not group:
            continue
        lines.append(f"Threshold: {thr}")
        for c in group:
            f, i = c["ref"]
            lines.append(f'Icon: {c["lat"]:.5f}, {c["lon"]:.5f}, 0, {f}, {i}, '
                         f'"{clean(c["name"])}\\n{clean(c["src_name"])} - {stamp}'
                         f'\\nView score: {round(100 * c.get("score", 0))}"')
        lines.append("")

    path = os.path.join(out_dir, f"cams_{key}.txt")
    with open(path, "w", encoding="cp1252", errors="replace", newline="\r\n") as f:
        f.write("\n".join(lines))
    return len(ordered), n_sheets


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="cam_out")
    ap.add_argument("--base-url", required=True,
                    help="public URL of the folder the output is published to")
    ap.add_argument("--demo", action="store_true", help="use fake cameras (offline test)")
    args = ap.parse_args()
    base_url = args.base_url.rstrip("/")

    if args.demo:
        cams = demo_cameras()
    else:
        cams = []
        for name, fn in (("Indiana", fetch_indiana), ("Illinois/Wisconsin", fetch_travelmidwest)):
            try:
                got = fn()
                print(f"{name}: {len(got)} cameras listed")
                cams += got
            except Exception as e:
                print(f"{name}: camera list FAILED ({e})", file=sys.stderr)
        if not cams:
            sys.exit("No cameras from any source; not publishing.")
        t0 = time.time()
        with ThreadPoolExecutor(DOWNLOAD_THREADS) as ex:
            cams = list(ex.map(download_thumb, cams))
        ok = [c for c in cams if c["thumb"] is not None]
        print(f"downloaded {len(ok)}/{len(cams)} snapshots in {time.time() - t0:.0f}s")
        cams = ok

    history = {} if args.demo else load_history(base_url)
    for c in cams:
        update_score(c, history)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "scores.json"), "w") as f:
        json.dump(history, f)
    scored = sum(1 for c in cams if c["score"] > 0)
    print(f"view scores: {scored}/{len(cams)} cameras have a daytime score")

    now = datetime.now(timezone.utc)
    stamp, epoch = now.strftime("%H:%MZ %b %d"), int(now.timestamp())
    by_region = {}
    for c in dedupe(cams):
        k = region_of(c)
        if k:
            by_region.setdefault(k, []).append(c)

    total = 0
    for key, title, *_ in REGIONS:
        group = by_region.get(key, [])
        if not group:
            print(f"  {key}: no cameras, skipped")
            continue
        n, s = build_region(key, title, group, args.out_dir, base_url, stamp, epoch)
        total += n
        print(f"  {key}: {n} cameras on {s} sheet(s)")
    if total < 50 and not args.demo:
        sys.exit(f"Only {total} cameras built; not publishing.")


if __name__ == "__main__":
    main()
