#!/usr/bin/env python3
"""Fetch the TPB TAZ polygon layer and build an interactive Leaflet map."""

from __future__ import annotations

import argparse
import html
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_QUERY_URL = "https://gis.mwcog.org/wa/rest/services/RTDC/TAZ/MapServer/1/query"
DEFAULT_GEOJSON = Path("06x_stage1/tpb_taz_polygons.geojson")
DEFAULT_HTML = Path("06x_stage1/tpb_taz_map.html")
OUT_FIELDS = "TAZ,STATE,STFIPS,CNTYFIPS,FIPSSTCO,REGION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-url", default=DEFAULT_QUERY_URL)
    parser.add_argument("--geojson-out", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--html-out", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--page-size", type=int, default=2000)
    parser.add_argument("--sleep", type=float, default=0.1)
    return parser.parse_args()


def fetch_page(query_url: str, page_size: int, offset: int) -> dict[str, Any]:
    params = {
        "f": "geojson",
        "where": "1=1",
        "outFields": OUT_FIELDS,
        "returnGeometry": "true",
        "outSR": "4326",
        "resultRecordCount": str(page_size),
        "resultOffset": str(offset),
    }
    url = f"{query_url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def fetch_geojson(args: argparse.Namespace) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = fetch_page(args.query_url, args.page_size, offset)
        page = payload.get("features", [])
        if not page:
            break
        features.extend(page)
        print(f"downloaded {len(features)} TAZ polygons")
        if len(page) < args.page_size and not payload.get("exceededTransferLimit"):
            break
        offset += len(page)
        time.sleep(args.sleep)

    features.sort(key=lambda feature: feature.get("properties", {}).get("TAZ", 0))
    return {
        "type": "FeatureCollection",
        "name": "TPB TAZ polygons",
        "source": args.query_url,
        "features": features,
    }


def feature_bounds(feature: dict[str, Any]) -> tuple[float, float, float, float] | None:
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates") or []
    points: list[tuple[float, float]] = []

    def collect(value: Any) -> None:
        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            points.append((float(value[0]), float(value[1])))
            return
        if isinstance(value, list):
            for item in value:
                collect(item)

    collect(coords)
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def collection_bounds(geojson: dict[str, Any]) -> list[list[float]]:
    bounds = [feature_bounds(feature) for feature in geojson.get("features", [])]
    bounds = [item for item in bounds if item is not None]
    if not bounds:
        return [[38.0, -78.5], [39.5, -76.0]]
    min_x = min(item[0] for item in bounds)
    min_y = min(item[1] for item in bounds)
    max_x = max(item[2] for item in bounds)
    max_y = max(item[3] for item in bounds)
    return [[min_y, min_x], [max_y, max_x]]


def state_counts(geojson: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature in geojson.get("features", []):
        state = str(feature.get("properties", {}).get("STATE") or "Unknown")
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))


def write_map(geojson: dict[str, Any], html_out: Path, source_url: str) -> None:
    data_json = json.dumps(geojson, separators=(",", ":"))
    bounds_json = json.dumps(collection_bounds(geojson))
    counts = state_counts(geojson)
    count_text = ", ".join(f"{html.escape(state)}: {count}" for state, count in counts.items())
    feature_count = len(geojson.get("features", []))
    feature_count_text = f"{feature_count:,}"
    source_text = html.escape(source_url)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TPB TAZ Polygon Layer</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
    }}
    body {{
      font-family: Arial, sans-serif;
      color: #1f2933;
    }}
    .info {{
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid #c9d2dc;
      border-radius: 4px;
      box-shadow: 0 1px 6px rgba(0, 0, 0, 0.18);
      line-height: 1.35;
      max-width: 340px;
      padding: 10px 12px;
    }}
    .info h1 {{
      font-size: 15px;
      margin: 0 0 6px;
    }}
    .info p {{
      font-size: 12px;
      margin: 4px 0;
    }}
    .info code {{
      overflow-wrap: anywhere;
    }}
    .legend {{
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const tazData = {data_json};
    const bounds = {bounds_json};
    const colors = {{
      "District of Columbia": "#1b9e77",
      "Maryland": "#d95f02",
      "Virginia": "#7570b3",
      "West Virginia": "#e7298a"
    }};

    const map = L.map("map", {{ preferCanvas: true }});
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    function style(feature) {{
      const state = feature.properties.STATE || "Unknown";
      return {{
        color: colors[state] || "#606f7b",
        fillColor: colors[state] || "#606f7b",
        fillOpacity: 0.14,
        opacity: 0.9,
        weight: 1
      }};
    }}

    function popupHtml(properties) {{
      return `
        <strong>TAZ ${{properties.TAZ ?? ""}}</strong><br>
        State: ${{properties.STATE ?? ""}}<br>
        County FIPS: ${{properties.FIPSSTCO ?? ""}}<br>
        Region: ${{properties.REGION ?? ""}}
      `;
    }}

    let selectedLayer = null;
    const layer = L.geoJSON(tazData, {{
      style,
      onEachFeature: (feature, lyr) => {{
        lyr.bindPopup(popupHtml(feature.properties || {{}}));
        lyr.on("mouseover", () => lyr.setStyle({{ weight: 3, fillOpacity: 0.28 }}));
        lyr.on("mouseout", () => {{
          if (lyr !== selectedLayer) layer.resetStyle(lyr);
        }});
        lyr.on("click", () => {{
          if (selectedLayer) layer.resetStyle(selectedLayer);
          selectedLayer = lyr;
          lyr.setStyle({{ weight: 4, fillOpacity: 0.32 }});
        }});
      }}
    }}).addTo(map);

    map.fitBounds(bounds, {{ padding: [18, 18] }});

    const info = L.control({{ position: "topright" }});
    info.onAdd = () => {{
      const div = L.DomUtil.create("div", "info");
      div.innerHTML = `
        <h1>TPB TAZ Polygon Layer</h1>
        <p><strong>{feature_count_text}</strong> polygons</p>
        <p>{count_text}</p>
        <p>Source: <code>{source_text}</code></p>
      `;
      return div;
    }};
    info.addTo(map);

    const legend = L.control({{ position: "bottomright" }});
    legend.onAdd = () => {{
      const div = L.DomUtil.create("div", "info legend");
      div.innerHTML = Object.entries(colors)
        .map(([state, color]) => `<div><span style="display:inline-block;width:12px;height:12px;background:${{color}};margin-right:6px;"></span>${{state}}</div>`)
        .join("");
      return div;
    }};
    legend.addTo(map);
  </script>
</body>
</html>
"""
    html_out.write_text(html_text)


def main() -> int:
    args = parse_args()
    args.geojson_out.parent.mkdir(parents=True, exist_ok=True)
    args.html_out.parent.mkdir(parents=True, exist_ok=True)

    geojson = fetch_geojson(args)
    args.geojson_out.write_text(json.dumps(geojson, separators=(",", ":")))
    print(f"wrote {args.geojson_out}")

    write_map(geojson, args.html_out, args.query_url)
    print(f"wrote {args.html_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
