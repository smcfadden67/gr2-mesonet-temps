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

THUMB_W, THUMB_H = 144, 96          # thumbnail size in pixels (3:2)
SHEET_COLS, SHEET_ROWS = 14, 21     # 2016 x 2016 px sheets, 294 thumbs each
JPEG_QUALITY = 72
DOWNLOAD_THREADS = 24
DOWNLOAD_TIMEOUT = 15
REFRESH_MINUTES = 5

# Zoom tiers for thumbnails: (grid cell in degrees, GR Threshold in nm).
# A camera shows its thumbnail at the first tier where it wins its grid cell,
# so zoomed out you see a spread-out sample and more appear as you zoom in.
THUMB_TIERS = [
    (0.20, 60),
    (0.07, 25),
    (None, 12),
]
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
]

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
                         name=f"{road.strip()}: {name.strip()}", src="INDOT", url=url))
    return cams


def fetch_illinois():
    body = json.dumps({"bbox": [-91.6, 36.9, -87.0, 42.6]}).encode()
    raw = http("https://travelmidwest.com/lmiga/cameraMap.json", body,
               {"Content-Type": "application/json", "Accept": "application/json",
                "Origin": "https://travelmidwest.com",
                "Referer": "https://travelmidwest.com/"})
    cams = []
    for f in json.loads(raw).get("features", []):
        p = f.get("properties", {})
        if not str(p.get("id", "")).startswith("IL-") or p.get("dis"):
            continue
        urls = [u for u in p.get("remUrls") or [] if str(u).startswith("http")]
        if not urls:
            continue
        lon, lat = f["geometry"]["coordinates"][:2]
        dirs = p.get("dirs") or []
        d = f" ({dirs[0]})" if dirs and dirs[0] not in ("NONE", "") else ""
        cams.append(dict(state="IL", lat=lat, lon=lon,
                         name=f"{p.get('locDesc', p['id'])}{d}",
                         src=p.get("src", "Travel Midwest"), url=urls[0]))
    return cams


# ------------------------------------------------------------------ images

def make_thumb(data):
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), (0, 0, 0))
    canvas.paste(im, ((THUMB_W - im.width) // 2, (THUMB_H - im.height) // 2))
    # thin border so thumbnails stand out on the radar
    ImageDraw.Draw(canvas).rectangle([0, 0, THUMB_W - 1, THUMB_H - 1],
                                     outline=(200, 200, 200))
    return canvas


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


def download_thumb(cam):
    for attempt in range(2):
        try:
            cam["thumb"] = make_thumb(http(cam["url"], timeout=DOWNLOAD_TIMEOUT))
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
                     name=f"Demo {key} #{i}", src="demo", url="")
            im = Image.new("RGB", (352, 240), tuple(random.randint(0, 255) for _ in range(3)))
            ImageDraw.Draw(im).text((10, 10), f"{key} {i}", fill=(255, 255, 255))
            c["thumb"] = make_thumb_from_image(im)
            cams.append(c)
    return cams


def make_thumb_from_image(im):
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    return make_thumb(buf.getvalue())


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
    remaining = list(cams)
    out = []
    for cell, thr in THUMB_TIERS:
        if cell is None:
            out.append((thr, remaining))
            break
        taken, rest, cells = [], [], set()
        for c in remaining:
            k = (int((c["lat"] + 90) // cell), int((c["lon"] + 180) // cell))
            (rest if k in cells else taken).append(c)
            cells.add(k)
        out.append((thr, taken))
        remaining = rest
    return out


def clean(s):
    return str(s).replace('"', "'").replace("\n", " ").strip()


def build_region(key, title, cams, out_dir, base_url, stamp, epoch):
    per_sheet = SHEET_COLS * SHEET_ROWS
    max_cams = per_sheet * 31   # file 1 is the camera icon
    cams = cams[:max_cams]

    # Pack sheets in tier order so the zoomed-out tier lives in sheet 1.
    tiers = assign_tiers(cams)
    ordered = [c for _, group in tiers for c in group]
    sheets = []
    for i, c in enumerate(ordered):
        s, idx = divmod(i, per_sheet)
        if idx == 0:
            sheets.append(Image.new("RGB", (THUMB_W * SHEET_COLS, THUMB_H * SHEET_ROWS)))
        r, col = divmod(idx, SHEET_COLS)
        sheets[-1].paste(c["thumb"], (col * THUMB_W, r * THUMB_H))
        c["file"], c["icon"] = s + 2, idx + 1

    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    lines = [
        f"; {title} - snapshots shrunk and packed by build_cameras.py, built {stamp}",
        "; Sources: INDOT (511in.org), Travel Midwest (travelmidwest.com)",
        f"Title: {title} ({stamp})",
        f"Refresh: {REFRESH_MINUTES}",
        'Font: 1, 11, 1, "Arial"',
        "",
    ]
    iw, ih = make_camera_icon(os.path.join(out_dir, "img", "camera.png"))
    lines.append(f'IconFile: 1, {iw}, {ih}, {iw // 2}, {ih // 2}, "{base_url}/img/camera.png"')
    for n, sheet in enumerate(sheets, 1):
        # Crop the last sheet to the rows actually used.
        used = len(ordered) - (n - 1) * per_sheet
        rows = min(SHEET_ROWS, -(-used // SHEET_COLS))
        sheet = sheet.crop((0, 0, THUMB_W * SHEET_COLS, THUMB_H * rows))
        fname = f"{key}_{n}.jpg"
        sheet.save(os.path.join(out_dir, "img", fname), "JPEG",
                   quality=JPEG_QUALITY, optimize=True)
        # ?t= busts caches so GR2 and GitHub serve the new sheet each run.
        lines.append(f'IconFile: {n + 1}, {THUMB_W}, {THUMB_H}, {THUMB_W // 2}, '
                     f'{THUMB_H // 2}, "{base_url}/img/{fname}?t={epoch}"')
    lines.append("")

    lines.append(f"Threshold: {MARKER_THRESHOLD}")
    for c in ordered:
        lines.append(f'Icon: {c["lat"]:.5f}, {c["lon"]:.5f}, 0, 1, 1, '
                     f'"{clean(c["name"])}\\n{clean(c["src"])} - snapshot {stamp}"')
    lines.append("")

    for thr, group in tiers:
        if not group:
            continue
        lines.append(f"Threshold: {thr}")
        for c in group:
            lines.append(f'Icon: {c["lat"]:.5f}, {c["lon"]:.5f}, 0, {c["file"]}, '
                         f'{c["icon"]}, "{clean(c["name"])}\\n{clean(c["src"])} - {stamp}"')
        lines.append("")

    path = os.path.join(out_dir, f"cams_{key}.txt")
    with open(path, "w", encoding="cp1252", errors="replace", newline="\r\n") as f:
        f.write("\n".join(lines))
    return len(ordered), len(sheets)


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
        for name, fn in (("Indiana", fetch_indiana), ("Illinois", fetch_illinois)):
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
