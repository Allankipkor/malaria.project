"""
Read ERA5/GRIB-derived NetCDF (e.g. data_stream-moda_stepType-avgua.nc), export CSVs,
and build a table keyed by (year, UN M49) for merging with malaria_cases.csv.

The avgua file uses valid_time x latitude x longitude (2m temperature in Kelvin).

- Domain features (grid mean, by calendar year): merged on year; same value for all countries.
- climate_t2m_local_mean_c: annual mean at nearest grid point to country centroid, only when
  the centroid lies inside the NetCDF lat/lon bounds.

Run from the malaria.project directory:

    python extract_climate_features.py

Requires: pandas, xarray, netCDF4
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

NC_DEFAULT = "data_stream-moda_stepType-avgua.nc"
MALARIA_CSV = "malaria_cases.csv"
OUTPUT_DIR = "outputs"
DOMAIN_YEARLY_CSV = os.path.join(OUTPUT_DIR, "climate_avgua_domain_by_year.csv")
YEAR_M49_CSV = os.path.join(OUTPUT_DIR, "climate_avgua_by_year_m49.csv")
MONTHLY_DOMAIN_CSV = os.path.join(OUTPUT_DIR, "climate_avgua_t2m_monthly_domain_mean.csv")
YEARLY_GRID_CSV = os.path.join(OUTPUT_DIR, "data_stream_avgua_t2m_yearly_gridded.csv")
CENTROIDS_CACHE = os.path.join(OUTPUT_DIR, "country_m49_centroids_cache.csv")

ISO3166_URL = (
    "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/"
    "master/all/all.csv"
)
CENTROIDS_URL = (
    "https://raw.githubusercontent.com/gavinr/world-countries-centroids/"
    "master/dist/countries.csv"
)


def kelvin_to_c(t: xr.DataArray | float) -> xr.DataArray | float:
    return t - 273.15


def download_centroid_table(cache_path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    if os.path.isfile(cache_path):
        return pd.read_csv(cache_path)

    with urllib.request.urlopen(ISO3166_URL, timeout=60) as r:
        iso = pd.read_csv(r)
    with urllib.request.urlopen(CENTROIDS_URL, timeout=60) as r:
        geo = pd.read_csv(r)

    iso = iso.rename(columns={"alpha-2": "iso2"})
    geo = geo.rename(columns={"ISO": "iso2", "latitude": "lat", "longitude": "lon"})
    merged = iso.merge(geo[["iso2", "lat", "lon"]], on="iso2", how="inner")
    merged["m49"] = merged["country-code"].astype(str).str.lstrip("0").replace("", "0").astype(int)
    out = merged[["m49", "lat", "lon", "name"]].drop_duplicates(subset=["m49"])
    out.to_csv(cache_path, index=False)
    return out


def domain_monthly_series(ds: xr.Dataset) -> pd.DataFrame:
    da_k = ds["t2m"]
    spatial_mean = da_k.mean(dim=["latitude", "longitude"])
    df = spatial_mean.to_dataframe(name="t2m_k").reset_index()
    df["t2m_c"] = kelvin_to_c(df["t2m_k"])
    return df[["valid_time", "t2m_k", "t2m_c"]]


def domain_yearly_features(monthly_domain: pd.DataFrame) -> pd.DataFrame:
    monthly_domain = monthly_domain.copy()
    monthly_domain["year"] = pd.to_datetime(monthly_domain["valid_time"]).dt.year
    g = monthly_domain.groupby("year")["t2m_c"]
    return g.agg(
        climate_t2m_domain_mean_c="mean",
        climate_t2m_domain_std_monthly_c="std",
        climate_t2m_domain_min_monthly_c="min",
        climate_t2m_domain_max_monthly_c="max",
    ).reset_index()


def bounds_inside_grid(lat: float, lon: float, ds: xr.Dataset) -> bool:
    lat_min = float(ds.latitude.min())
    lat_max = float(ds.latitude.max())
    lon_min = float(ds.longitude.min())
    lon_max = float(ds.longitude.max())
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def local_annual_nearest(ds: xr.Dataset, centroids: pd.DataFrame) -> pd.DataFrame:
    da = ds["t2m"]
    rows = []
    for _, r in centroids.iterrows():
        lat0, lon0, m49 = float(r["lat"]), float(r["lon"]), int(r["m49"])
        if not bounds_inside_grid(lat0, lon0, ds):
            continue
        point = da.sel(latitude=lat0, longitude=lon0, method="nearest")
        dfp = point.to_dataframe(name="t2m_k").reset_index()
        dfp["year"] = pd.to_datetime(dfp["valid_time"]).dt.year
        dfp["t2m_c"] = kelvin_to_c(dfp["t2m_k"])
        yearly = dfp.groupby("year", as_index=False)["t2m_c"].mean()
        yearly = yearly.rename(columns={"t2m_c": "climate_t2m_local_mean_c"})
        yearly["m49"] = m49
        rows.append(yearly)

    if not rows:
        return pd.DataFrame(columns=["year", "m49", "climate_t2m_local_mean_c"])

    return pd.concat(rows, ignore_index=True)


def yearly_gridded_csv(ds: xr.Dataset, out_path: str) -> None:
    da_k = ds["t2m"]
    years = pd.to_datetime(ds["valid_time"].values).year
    da_k = da_k.assign_coords(year=("valid_time", years))
    annual_k = da_k.groupby("year").mean(dim="valid_time")
    annual_c = kelvin_to_c(annual_k)
    df = annual_c.to_dataframe(name="t2m_c").dropna().reset_index()
    df.to_csv(out_path, index=False)


def malaria_feature_table(
    domain_y: pd.DataFrame, local_y: pd.DataFrame, malaria_path: str
) -> pd.DataFrame:
    mal = pd.read_csv(malaria_path)
    keys = mal[["DIM_TIME", "DIM_GEO_CODE_M49"]].drop_duplicates()
    keys = keys.rename(columns={"DIM_TIME": "year", "DIM_GEO_CODE_M49": "m49"})
    out = keys.merge(domain_y, on="year", how="left")
    if not local_y.empty:
        out = out.merge(local_y, on=["year", "m49"], how="left")
    else:
        out["climate_t2m_local_mean_c"] = np.nan
    return out


def main():
    if not os.path.isfile(NC_DEFAULT):
        raise SystemExit(f"NetCDF not found: {NC_DEFAULT}")
    if not os.path.isfile(MALARIA_CSV):
        raise SystemExit(f"Malaria CSV not found: {MALARIA_CSV}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds = xr.open_dataset(NC_DEFAULT)
    monthly = domain_monthly_series(ds)
    domain_y = domain_yearly_features(monthly)

    centroids = download_centroid_table(CENTROIDS_CACHE)
    mal = pd.read_csv(MALARIA_CSV)
    m49_need = mal["DIM_GEO_CODE_M49"].drop_duplicates()
    centroids_sub = centroids[centroids["m49"].isin(m49_need)]
    local_y = local_annual_nearest(ds, centroids_sub)

    feat = malaria_feature_table(domain_y, local_y, MALARIA_CSV)

    monthly.to_csv(MONTHLY_DOMAIN_CSV, index=False)
    domain_y.to_csv(DOMAIN_YEARLY_CSV, index=False)
    feat.to_csv(YEAR_M49_CSV, index=False)
    yearly_gridded_csv(ds, YEARLY_GRID_CSV)
    ds.close()

    print(f"Wrote monthly domain series: {MONTHLY_DOMAIN_CSV}")
    print(f"Wrote domain yearly features: {DOMAIN_YEARLY_CSV}")
    print(f"Wrote malaria merge table (year x M49): {YEAR_M49_CSV}")
    print(f"Wrote yearly gridded t2m (C): {YEARLY_GRID_CSV}")


if __name__ == "__main__":
    main()
