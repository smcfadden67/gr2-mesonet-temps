#!/usr/bin/env python3
"""
Build a GR2Analyst placefile of CONUS mesonet air temperatures.

Data source: NOAA MADIS public surface data (no account or token needed).
METAR/ASOS/AWOS airport stations and maritime/tide stations are excluded, so
what's left is CWOP, MesoWest-aggregated state mesonets, RAWS, DOT/RWIS,
agricultural networks, etc.

Stations are thinned into zoom tiers with GR "Threshold" values: zoomed out
you see one station per grid cell (best network wins), and more fill in as
you zoom in.

Usage:
    python build_placefile.py                     # fetch live, write mesonet_temps.txt
    python build_placefile.py --input sample.html # parse a saved MADIS response
    python build_placefile.py --out path/to/file.txt
Standard library only.
"""

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

MADIS_URL = "https://madis-data.ncep.noaa.gov/madisPublic1/cgi-bin/madisXmlPublicDir"

# CONUS box (lat/lon corners).
BBOX = dict(latll=24.0, lonll=-125.0, latur=49.5, lonur=-66.5)

# Providers that are airport (METAR) or marine/tide stations, not mesonets.
EXCLUDE_PROVIDERS = {
    "ASOS", "ASOS-HFM", "AWOS", "OTHER-MTR", "OTHER-HFM", "NonFedAWOS",
    "MARITIME", "NOS-NWLON", "NOS-PORTS", "COOP",
}

# When thinning, better-sited networks win a grid cell over backyard stations.
# Lower number = higher priority. Anything not listed gets 5; CWOP gets 9.
PROVIDER_PRIORITY = {
    # research-grade state/regional mesonets
    "NJWxNet": 1, "NC-ECONet": 1, "IEM": 1, "CoAgMet": 1, "MOComAgNet": 1,
    "NEPP": 1, "LCRA": 1, "SFWMD": 1, "MAP": 1, "CAIC": 1, "OKMESO": 1,
    "NYSM": 1, "KSMESO": 1, "WTXMESO": 1, "DEOS": 1, "FAWN": 1,
    # federal mesonets
    "RAWS": 2, "NRCS": 2, "SNOTEL": 2,
    # MesoWest-aggregated networks and DOT/RWIS
    "MesoWest": 3, "ITD": 3, "OHDOT": 3, "MADOT": 3, "NHDOT": 3,
    "VTDOT": 3, "CO_E-470": 3,
    # hydro networks
    "HADS": 4, "CA-Hydro": 4,
    # citizen weather
    "APRSWXNET": 9,
}

# Zoom tiers: (grid cell size in degrees, GR Threshold in nautical miles).
# A station appears in the first (coarsest) tier where it wins its cell.
# None = every remaining station.
TIERS = [
    (None, 999),
]

# Drop obs that MADIS QC flagged as questioned/rejected/bad.
BAD_QC = {"Q", "X", "B"}

# Sanity limits (°F) as a last line of defense.
T_MIN_F, T_MAX_F = -60.0, 130.0

REFRESH_MINUTES = 5
FONT = '1, 11, 1, "Arial"'

# Temperature color scale (°F). Stepped bands: each entry is
# (lower bound, RGB); a temp uses the first band whose lower bound it meets.
COLOR_BANDS = [
    ( 125, (254, 254, 254)),
    ( 120, (251, 223, 223)),
    ( 115, (247, 192, 192)),
    ( 110, (244, 160, 160)),
    ( 105, (240, 128, 128)),
    ( 100, (202,  88,  87)),
    (  95, (150,  16,   0)),
    (  90, (184,  65,   0)),
    (  85, (219, 115,   0)),
    (  80, (254, 163,   1)),
    (  75, (254, 221,   0)),
    (  70, (254, 254,   0)),
    (  65, (188, 221,   0)),
    (  60, (129, 192,   0)),
    (  55, ( 66, 160,   0)),
    (  50, (  2, 128,   1)),
    (  45, ( 34, 153,  57)),
    (  40, ( 67, 178, 112)),
    (  35, (101, 204, 170)),
    (  32, (  0, 250, 254)),
    (  30, (  0, 168, 254)),
    (  25, (  0,  86, 255)),
    (  20, (  0,   1, 254)),
    (  15, ( 59,  45, 240)),
    (  10, (103,  79, 230)),
    (   5, (147, 112, 219)),
    (   0, (149,   1, 211)),
    (  -5, (163,  32, 213)),
    ( -10, (177,  64, 215)),
    ( -15, (192,  96, 217)),
    ( -20, (253, 108, 181)),
    ( -30, (235, 138, 194)),
    ( -40, (215, 171, 209)),
    ( -50, (196, 203, 215)),
    ( -60, (175, 238, 238)),
]

MIN_STATIONS_OK = 1000   # fail instead of publishing if far fewer than expected


def temp_color(t):
    for lo, rgb in COLOR_BANDS:
        if t >= lo:
            return rgb
    return COLOR_BANDS[-1][1]


def fetch_madis(retries=3):
    params = [
        ("rdr", ""), ("time", "0"), ("minbck", "-59"), ("minfwd", "0"),
        ("recwin", "3"), ("timefilter", "0"), ("dfltrsel", "1"), ("state", "PA"),
        ("latll", BBOX["latll"]), ("lonll", BBOX["lonll"]),
        ("latur", BBOX["latur"]), ("lonur", BBOX["lonur"]),
        ("stanam", ""), ("stasel", "0"), ("pvdrsel", "0"), ("varsel", "1"),
        ("qctype", "0"), ("qcsel", "1"), ("xml", "5"), ("csvmiss", "0"),
        ("nvars", "LAT"), ("nvars", "LON"), ("nvars", "T"),
    ]
    url = MADIS_URL + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "gr2-mesonet-temps placefile builder"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read().decode("latin-1")
        except Exception as e:
            last = e
            print(f"fetch attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(15)
    raise RuntimeError(f"MADIS fetch failed: {last}")


ROW_RE = re.compile(r"^\s*\S+\s*,\d\d/\d\d/\d{4},\d\d:\d\d,")


def parse(text):
    """Return dict stid -> station, keeping the newest valid ob per station."""
    stations = {}
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        c = [x.strip() for x in line.split(",")]
        # STAID,OBDATE,OBTIME,PVDR,SUBPVDR,LAT,D,LON,D,T,D
        if len(c) < 11:
            continue
        stid, obdate, obtime, pvdr, subpvdr = c[0], c[1], c[2], c[3], c[4]
        if pvdr in EXCLUDE_PROVIDERS:
            continue
        try:
            lat, lon, tk = float(c[5]), float(c[7]), float(c[9])
        except ValueError:
            continue
        if tk < -9000 or c[10] in BAD_QC:
            continue
        tf = (tk - 273.15) * 9 / 5 + 32
        if not (T_MIN_F <= tf <= T_MAX_F):
            continue
        if not (BBOX["latll"] <= lat <= BBOX["latur"] and
                BBOX["lonll"] <= lon <= BBOX["lonur"]):
            continue
        try:
            ts = datetime.strptime(f"{obdate} {obtime}", "%m/%d/%Y %H:%M").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        prev = stations.get(stid)
        if prev and prev["time"] >= ts:
            continue
        stations[stid] = dict(stid=stid, pvdr=pvdr, sub=subpvdr, lat=lat,
                              lon=lon, tf=tf, time=ts)
    return stations


def assign_tiers(stations):
    """Return list of (threshold, [stations]) using grid thinning."""
    remaining = sorted(
        stations.values(),
        key=lambda s: (PROVIDER_PRIORITY.get(s["pvdr"], 5), -s["time"].timestamp()))
    out = []
    for cell, thr in TIERS:
        if cell is None:
            out.append((thr, remaining))
            break
        taken, rest, cells = [], [], set()
        for s in remaining:
            key = (int((s["lat"] + 90) // cell), int((s["lon"] + 180) // cell))
            if key in cells:
                rest.append(s)
            else:
                cells.add(key)
                taken.append(s)
        out.append((thr, taken))
        remaining = rest
    return out


def clean(s):
    return s.replace('"', "'")


def build(stations):
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%MZ")
    lines = [
        f"; CONUS mesonet air temperatures from NOAA MADIS public data, built {stamp}",
        "; METAR/ASOS/AWOS and marine stations excluded.",
        f"Title: Mesonet Temps (F) {stamp}",
        f"Refresh: {REFRESH_MINUTES}",
        f"Font: {FONT}",
        "",
    ]
    for thr, group in assign_tiers(stations):
        lines.append(f"Threshold: {thr}")
        last_color = None
        for s in group:
            color = temp_color(round(s["tf"]))   # color matches the number shown
            if color != last_color:
                lines.append("Color: %d %d %d" % color)
                last_color = color
            net = s["pvdr"] + (f"/{s['sub']}" if s["sub"] else "")
            hover = (f"{clean(s['stid'])} ({clean(net)})\\n"
                     f"{s['tf']:.1f}°F at {s['time']:%H:%MZ}")
            lines.append(f'Text: {s["lat"]:.4f}, {s["lon"]:.4f}, 1, '
                         f'"{round(s["tf"])}", "{hover}"')
        lines.append("")
    return "\r\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="parse a saved MADIS response instead of fetching")
    ap.add_argument("--out", default="mesonet_temps.txt")
    ap.add_argument("--min-stations", type=int, default=MIN_STATIONS_OK)
    args = ap.parse_args()

    if args.input:
        with open(args.input, encoding="latin-1") as f:
            text = f.read()
    else:
        text = fetch_madis()

    stations = parse(text)
    print(f"{len(stations)} mesonet stations after filtering")
    if len(stations) < args.min_stations:
        sys.exit(f"Only {len(stations)} stations (< {args.min_stations}); not publishing.")

    body = build(stations)
    # GR2 reads placefiles as Windows ANSI.
    with open(args.out, "w", encoding="cp1252", errors="replace", newline="") as f:
        f.write(body)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
