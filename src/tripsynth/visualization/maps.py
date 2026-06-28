"""Coverage maps for observed roadway counts."""

from __future__ import annotations

import html
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def write_observed_counts_sample(gdf: gpd.GeoDataFrame, path: str | Path, n: int = 20) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = gdf.head(n).copy()
    sample["geometry_wkt"] = sample.geometry.to_wkt() if len(sample) else []
    sample.drop(columns=["geometry"]).to_csv(path, index=False)


def _aadt_quantile_labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < 2:
        return pd.Series(["unknown"] * len(values), index=values.index)
    try:
        return pd.qcut(numeric, q=5, labels=["q1", "q2", "q3", "q4", "q5"], duplicates="drop")
    except ValueError:
        return pd.Series(["unknown"] * len(values), index=values.index)


def write_observed_counts_coverage_html(
    gdf: gpd.GeoDataFrame,
    path: str | Path,
    *,
    title: str = "Observed Counts Coverage",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        body = "<p>No observed-count features were available after filtering.</p>"
        path.write_text(f"<html><body><h1>{html.escape(title)}</h1>{body}</body></html>", encoding="utf-8")
        return

    map_gdf = gdf.to_crs("EPSG:4326").copy()
    map_gdf["aadt_quantile"] = _aadt_quantile_labels(map_gdf["observed_aadt"]).astype(str)
    keep = [
        "source",
        "state",
        "county_fips",
        "route_name",
        "observed_aadt",
        "aadt_quantile",
        "geometry",
    ]
    geojson = json.loads(map_gdf[keep].to_json())
    bounds = map_gdf.total_bounds
    center_lat = float((bounds[1] + bounds[3]) / 2)
    center_lon = float((bounds[0] + bounds[2]) / 2)
    geojson_text = json.dumps(geojson)

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; }}
    #map {{ width: 100vw; height: 100vh; }}
    .panel {{
      position: absolute; top: 12px; left: 56px; z-index: 500;
      background: white; padding: 8px 10px; border: 1px solid #999;
      max-width: 380px; font-size: 13px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <strong>{html.escape(title)}</strong><br>
    Features: {len(map_gdf):,}<br>
    Source temporal type: annual-average/proxy AADT.
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const data = {geojson_text};
    const colors = {{q1: "#d4e6f1", q2: "#85c1e9", q3: "#3498db", q4: "#f39c12", q5: "#c0392b", unknown: "#7f8c8d"}};
    const map = L.map("map").setView([{center_lat:.6f}, {center_lon:.6f}], 9);
    L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);
    const layer = L.geoJSON(data, {{
      style: feature => {{
        const q = feature.properties.aadt_quantile || "unknown";
        return {{color: colors[q] || colors.unknown, weight: 2, opacity: 0.75}};
      }},
      onEachFeature: (feature, layer) => {{
        const p = feature.properties;
        layer.bindPopup(`<b>${{p.route_name || "route unknown"}}</b><br>AADT: ${{p.observed_aadt ?? "missing"}}<br>State: ${{p.state}}<br>County: ${{p.county_fips || "missing"}}`);
      }}
    }}).addTo(map);
    map.fitBounds(layer.getBounds());
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_observed_counts_coverage_png(gdf: gpd.GeoDataFrame, path: str | Path) -> str | None:
    if gdf.empty:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_gdf = gdf.to_crs("EPSG:4326").copy()
        plot_gdf["aadt_quantile"] = _aadt_quantile_labels(plot_gdf["observed_aadt"]).astype(str)
        fig, ax = plt.subplots(figsize=(9, 7))
        plot_gdf.plot(column="aadt_quantile", linewidth=0.7, legend=True, ax=ax)
        ax.set_title("Observed AADT coverage in study area")
        ax.set_axis_off()
        fig.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path)
    except Exception:
        return None
