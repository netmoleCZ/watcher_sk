"""
Fetch SHMU ALADIN-SK NWP GRIB files and write shmu_aladin.json.

URL pattern (confirmed 2026-06-22):
  https://opendata.shmu.sk/meteorology/weather/nwp/aladin/sk/4.5km/
  {YYYYMMDD}/{HHMM}/al-grib_sk_{STEP:03d}-{YYYYMMDD}-{HHMM}-nwp-.grb

Stations are discovered from SHMU's own observation feed rather than a pinned list, so
the forecast keys match the station IDs Watcher stores verbatim (`11:11801`), and any
station a user selects is covered without a config edit.

That matters: the pinned list previously held 8 points chosen independently of the
observation network, so its IDs matched no station Watcher tracked and half the tracked
stations had no forecast point within the consumer's 30 km cutoff. Every Slovak forecast
failed. config/stations.json is retained only as a fallback for a feed outage.

Output format (identical to Czech ALADIN JSON, plus lat/lon):
  {
    "generated_at": "...",
    "stations": {
      "11:11813": {
        "lat": 48.167778,            # published so a consumer can match on position
        "lon": 17.105833,            # without depending on a shared ID scheme
        "times":    ["2026-06-21T12:00:00Z", ...],
        "temp":     [296.2, ...],    # Kelvin  — ForecastService subtracts 273.15
        "humidity": [0.61,  ...],    # fraction 0–1 — ForecastService multiplies by 100
        "pressure": [101140.0, ...]  # Pa — ForecastService divides by 100
      }
    }
  }

Run: python scripts/fetch_shmu_aladin.py
"""

import json
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import urllib3

import eccodes
import numpy as np
import requests

# opendata.shmu.sk uses a Slovak government CA not present in Ubuntu's default
# trust store, causing SSL verification to fail on GitHub Actions runners.
# Disabling verification is acceptable here: public NWP data, read-only, no auth.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://opendata.shmu.sk/meteorology/weather/nwp/aladin/sk/4.5km"

# SHMU's open observation feed — the same source Watcher's ShmuService reads, so the
# station IDs and coordinates here are exactly the ones it stores.
OBS_URL = "http://meteo.shmu.sk/customer/home/opendata/"

# The feed carries WMO-exchange stations from neighbouring countries too; keep Slovakia
# plus a ~1° buffer, matching ShmuService::stations().
SK_LAT_MIN, SK_LAT_MAX = 47.0, 50.5
SK_LON_MIN, SK_LON_MAX = 15.5, 23.5

# How many hours back to look for a populated observation file before giving up.
OBS_LOOKBACK_HOURS = 6

# GRIB shortName aliases for each field — tried in listed order; first match wins.
# Exact names are confirmed on first successful run and logged to stdout.
TEMP_NAMES = ("2t", "t2m", "t")
HUM_NAMES  = ("r", "2r", "rh", "q")
PRES_NAMES = ("prmsl", "msl", "pres", "sp")

# Steps to download: 000–048 gives a 48-hour window (sufficient for ForecastService's 9-day
# request; SHMU provides up to 072 but the extra steps add download time for no current benefit).
STEPS = range(0, 49)

# Network resilience. A ConnectTimeout/5xx right after a run publishes usually means SHMU is
# briefly overloaded, not that the data is missing — so retry a few times with backoff before
# giving up. A 404 is NOT retried (the file simply isn't published for that step yet).
MAX_RETRIES  = 3
BACKOFF_BASE = 2.0   # seconds between attempts: 2, 4 (exponential)

# How far back to shift from `now` before flooring to a 6-h model-run boundary.
SAFETY_HOURS = 3

# How many successively older runs to try when the newest is not published yet.
# Three covers an 18 h window, far beyond any plausible Actions scheduling delay.
RUN_CANDIDATES = 3


class HostUnreachable(Exception):
    """The SHMU host could not be contacted at all after MAX_RETRIES attempts."""


def candidate_runs(now: datetime) -> list[datetime]:
    """Model runs to try, newest first.

    SHMU publishes output ~2–3 h after init, so the newest run is not always there.
    Picking a single run made publication timing load-bearing on *when the job happened
    to start*: a scheduled 20:30 job targets 12Z (8.5 h old, safely published), but
    GitHub Actions routinely delays scheduled runs, and a start at 21:12 pushes
    `now - SAFETY_HOURS` across the 18:00 boundary onto a run only 3.2 h old that SHMU
    has not published yet. Every step then 404s and the whole run fails.

    Shifting the cron time cannot fix that — any start time can be delayed across a
    boundary. So try the newest run and walk back 6 h at a time instead.
    """
    safe     = now - timedelta(hours=SAFETY_HOURS)
    run_hour = (safe.hour // 6) * 6
    latest   = safe.replace(hour=run_hour, minute=0, second=0, microsecond=0)

    return [latest - timedelta(hours=6 * i) for i in range(RUN_CANDIDATES)]


def grib_url(date: str, run: str, step: int) -> str:
    filename = f"al-grib_sk_{step:03d}-{date}-{run}-nwp-.grb"
    return f"{BASE_URL}/{date}/{run}/{filename}"


def download_grib(url: str) -> bytes | None:
    """Download one GRIB step.

    Returns the file bytes on success, or None if the file is not published (404)
    or fails with a non-retryable error. Raises HostUnreachable if the server
    cannot be contacted (connect timeout / connection error / 5xx) after MAX_RETRIES,
    so the caller can abort the whole run instead of timing out on every step.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=30, verify=False)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            if r.status_code >= 500:
                # Server-side hiccup (often overload just after publish) — retryable.
                last_exc = Exception(f"HTTP {r.status_code}")
            else:
                print(f"  HTTP {r.status_code}: {url}", file=sys.stderr)
                return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
        except Exception as exc:
            # Non-network error (e.g. malformed URL) — retrying won't help.
            print(f"  Download error ({url}): {exc}", file=sys.stderr)
            return None

        if attempt < MAX_RETRIES:
            delay = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  connection failed (attempt {attempt}/{MAX_RETRIES}), "
                  f"retrying in {delay:.0f}s: {last_exc}", file=sys.stderr)
            time.sleep(delay)

    raise HostUnreachable(f"{url} — {last_exc}")


def parse_observation_csv(text: str) -> list[dict]:
    """Extract unique {id, name, lat, lon} from one hour of SHMU's observation CSV.

    Column layout (semicolon-separated, header starts with 'obs_stn'):
      0 station id ('11:11801')   2 name   3 lat   4 lon   5 elevation   6 timestamp
    """
    stations: dict[str, dict] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("obs_stn"):
            continue

        cols = line.split(";")
        if len(cols) < 13:
            continue

        sid = cols[0].strip()
        if not sid or sid in stations:
            continue

        try:
            lat = float(cols[3])
            lon = float(cols[4])
        except (ValueError, IndexError):
            continue

        if not (SK_LAT_MIN <= lat <= SK_LAT_MAX and SK_LON_MIN <= lon <= SK_LON_MAX):
            continue

        stations[sid] = {"id": sid, "name": cols[2].strip(), "lat": lat, "lon": lon}

    return list(stations.values())


def discover_stations(now: datetime) -> list[dict]:
    """Station list from SHMU's observation feed, newest populated hour wins.

    Walks back hour by hour because the most recent file is often still empty. Returns
    an empty list if nothing usable is found, leaving the caller to fall back.
    """
    for back in range(OBS_LOOKBACK_HOURS):
        dt  = now - timedelta(hours=back)
        url = f"{OBS_URL}?observations;date={dt.strftime('%Y-%m-%d')}:{dt.strftime('%H')}"

        try:
            r = requests.get(url, timeout=30, verify=False)
        except Exception as exc:
            print(f"  observation feed {dt:%Y-%m-%d %H}Z: {exc}", file=sys.stderr)
            continue

        if r.status_code != 200:
            continue

        stations = parse_observation_csv(r.text)
        if stations:
            print(f"  stations from observation feed {dt:%Y-%m-%d %H}Z: {len(stations)}")
            return stations

    return []


def load_stations(now: datetime) -> tuple[list[dict], str]:
    """Discovered stations, or the pinned config if the feed is unusable."""
    stations = discover_stations(now)
    if stations:
        return stations, "observation feed"

    config_path = Path("config/stations.json")
    if not config_path.exists():
        print("Observation feed unusable and config/stations.json not found", file=sys.stderr)
        sys.exit(1)

    stations = json.loads(config_path.read_text(encoding="utf-8"))["stations"]
    if not stations:
        print("Observation feed unusable and config/stations.json is empty", file=sys.stderr)
        sys.exit(1)

    # Not fatal: a stale forecast for the pinned points beats no forecast at all, and the
    # next run four hours later will almost certainly reach the feed.
    print("  WARNING: observation feed unusable — falling back to pinned config/stations.json",
          file=sys.stderr)
    return stations, "config/stations.json (fallback)"


def nearest_value(lats: np.ndarray, lons: np.ndarray,
                  vals: np.ndarray, target_lat: float, target_lon: float) -> float:
    dist = (lats - target_lat) ** 2 + (lons - target_lon) ** 2
    return float(vals[dist.argmin()])


def parse_grib_step(data: bytes) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Parse one GRIB step file and return a dict with keys "temp", "humidity", "pressure".
    Each value is a (lats, lons, values) tuple of flat numpy arrays.
    Missing fields are omitted from the returned dict.
    """
    found: dict[str, tuple] = {}
    found_names: dict[str, str] = {}

    with tempfile.NamedTemporaryFile(suffix=".grb", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        with open(tmp_path, "rb") as f:
            while True:
                msg = eccodes.codes_grib_new_from_file(f)
                if msg is None:
                    break
                try:
                    short = eccodes.codes_get(msg, "shortName", ktype=str)
                    lats  = eccodes.codes_get_array(msg, "latitudes")
                    lons  = eccodes.codes_get_array(msg, "longitudes")
                    vals  = eccodes.codes_get_array(msg, "values").astype(float)

                    if "temp" not in found and short in TEMP_NAMES:
                        found["temp"] = (lats, lons, vals)
                        found_names["temp"] = short

                    elif "humidity" not in found and short in HUM_NAMES:
                        # Normalise to fraction 0–1: ALADIN may publish % (0–100)
                        if vals.max() > 1.5:
                            vals = vals / 100.0
                        found["humidity"] = (lats, lons, vals)
                        found_names["humidity"] = short

                    elif "pressure" not in found and short in PRES_NAMES:
                        found["pressure"] = (lats, lons, vals)
                        found_names["pressure"] = short

                finally:
                    eccodes.codes_release(msg)
    finally:
        tmp_path.unlink(missing_ok=True)

    return found, found_names


def collect_run(base_dt: datetime, stations: list[dict]) -> tuple[dict[str, dict], int]:
    """Download and extract every step of one model run.

    Returns (station_data, ok_steps). ok_steps == 0 means this run is unusable — most
    often because SHMU has not published it yet, in which case the caller should try an
    older one.
    """
    date = base_dt.strftime("%Y%m%d")
    run  = base_dt.strftime("%H%M")

    # Initialise per-station accumulators. lat/lon are published so a consumer can match
    # on position without having to share an ID scheme with this pipeline.
    station_data: dict[str, dict] = {
        s["id"]: {
            "lat":      s["lat"],
            "lon":      s["lon"],
            "times":    [],
            "temp":     [],
            "humidity": [],
            "pressure": [],
        }
        for s in stations
    }

    confirmed_names: dict[str, str] = {}
    ok_steps = 0

    for step in STEPS:
        url  = grib_url(date, run, step)
        try:
            data = download_grib(url)
        except HostUnreachable as exc:
            # The server is down/unreachable, not just missing this file — every
            # remaining step would time out identically. Stop and use whatever we have.
            print(f"  step {step:03d}: host unreachable — aborting remaining steps ({exc})",
                  file=sys.stderr)
            break
        if data is None:
            # Step 000 missing means the run itself is not published — no point issuing
            # another 48 requests that will all 404. Bail so the caller can try an
            # older run.
            if step == min(STEPS):
                print(f"  step {step:03d}: not available — run not published yet")
                return station_data, 0
            print(f"  step {step:03d}: not available")
            continue

        fields, names = parse_grib_step(data)

        # Log shortNames on first successful step so they can be verified
        if not confirmed_names and names:
            confirmed_names = names
            print(f"  GRIB shortNames confirmed: {names}")

        missing = [k for k in ("temp", "humidity", "pressure") if k not in fields]
        if missing:
            print(f"  step {step:03d}: missing fields {missing} — skipped")
            continue

        step_ts = (base_dt + timedelta(hours=step)).strftime("%Y-%m-%dT%H:%M:%SZ")
        lats_t, lons_t, vals_t = fields["temp"]
        lats_h, lons_h, vals_h = fields["humidity"]
        lats_p, lons_p, vals_p = fields["pressure"]

        for s in stations:
            sid  = s["id"]
            lat  = s["lat"]
            lon  = s["lon"]
            sd   = station_data[sid]
            sd["times"].append(step_ts)
            sd["temp"].append(    round(nearest_value(lats_t, lons_t, vals_t, lat, lon), 2))
            sd["humidity"].append(round(nearest_value(lats_h, lons_h, vals_h, lat, lon), 4))
            sd["pressure"].append(round(nearest_value(lats_p, lons_p, vals_p, lat, lon), 1))

        ok_steps += 1
        print(f"  step {step:03d}: {step_ts} ok")

    return station_data, ok_steps


def main() -> None:
    now              = datetime.now(timezone.utc)
    stations, source = load_stations(now)

    print(f"Steps: {min(STEPS)}–{max(STEPS)}  Stations: {len(stations)} ({source})")

    station_data: dict[str, dict] = {}
    ok_steps = 0
    used_run: datetime | None = None

    for base_dt in candidate_runs(now):
        print(f"Trying run {base_dt:%Y%m%d %H%M}Z "
              f"({(now - base_dt).total_seconds() / 3600:.1f} h after init)")

        station_data, ok_steps = collect_run(base_dt, stations)
        if ok_steps:
            used_run = base_dt
            break

        print(f"  run {base_dt:%Y%m%d %H%M}Z unusable — falling back to the previous run")

    if ok_steps == 0 or used_run is None:
        print("ERROR: no usable model run found — not writing output", file=sys.stderr)
        sys.exit(1)

    output = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_run":    used_run.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stations":     station_data,
    }

    out_path = Path("shmu_aladin.json")
    out_path.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path}  ({size_kb:.1f} KB)  "
          f"run={used_run:%Y%m%d %H%M}Z  ok_steps={ok_steps}/{len(list(STEPS))}")


if __name__ == "__main__":
    main()
