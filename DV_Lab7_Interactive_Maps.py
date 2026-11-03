from pathlib import Path
from io import BytesIO
import shutil
import urllib.request
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "ne_110m_admin_0_countries"
DATA_STEM = "ne_110m_admin_0_countries"
SHAPEFILE = DATA_DIR / f"{DATA_STEM}.shp"
DOWNLOAD_URL = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/"
    "ne_110m_admin_0_countries.zip"
)
COLOR_SCHEMES = ["OrRd", "YlGn", "Blues", "Purples", "Greens"]
PROJECTIONS = {
    "Equirectangular (WGS 84)": "EPSG:4326",
    "Robinson": "+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs",
    "Mollweide": "+proj=moll +lon_0=0 +datum=WGS84 +units=m +no_defs",
    "Mercator": "EPSG:3857",
    "Equal Earth": "EPSG:8857",
}


def ensure_dataset():
    """Download the public Natural Earth shapefile if it is not present."""
    if SHAPEFILE.exists() and all(
        SHAPEFILE.with_suffix(ext).exists() for ext in (".shx", ".dbf")
    ):
        return True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=25) as response:
            zip_bytes = response.read()

        with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
            # Extract only the known, required Natural Earth files.
            # Avoid extracting arbitrary paths from the archive.
            allowed_exts = {".shp", ".shx", ".dbf", ".prj", ".cpg"}
            for member in archive.infolist():
                filename = Path(member.filename).name
                if (
                    Path(filename).stem == DATA_STEM
                    and Path(filename).suffix.lower() in allowed_exts
                ):
                    with archive.open(member) as source, open(
                        DATA_DIR / filename, "wb"
                    ) as target:
                        shutil.copyfileobj(source, target)

        return SHAPEFILE.exists() and SHAPEFILE.with_suffix(".dbf").exists()
    except Exception as exc:
        st.error(f"The Natural Earth dataset could not be downloaded: {exc}")
        st.info(
            "Download the 'Admin 0 – Countries' 110m shapefile manually, "
            "extract its files into a folder named "
            "'ne_110m_admin_0_countries' next to this Python file, "
            "and rerun the app."
        )
        st.markdown(
            "Dataset website: "
            "https://www.naturalearthdata.com/downloads/110m-cultural-vectors/"
        )
        return False


@st.cache_data(show_spinner="Loading Natural Earth country boundaries...")
def load_data(shapefile_path):
    """Load world boundaries and calculate valid population density values."""
    gdf = gpd.read_file(shapefile_path)

    # Natural Earth column names vary by version (NAME/name, POP_EST/pop_est).
    columns = {col.lower(): col for col in gdf.columns}
    name_col = columns.get("name") or columns.get("admin")
    pop_col = columns.get("pop_est")

    if name_col is None:
        raise ValueError("Country name column (NAME or ADMIN) was not found.")

    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.crs is None:
        # Natural Earth countries are normally supplied in WGS84.
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["Country"] = gdf[name_col].astype(str)

    rng = np.random.default_rng(0)
    fallback_populations = pd.Series(
        rng.integers(1_000_000, 1_000_000_000, size=len(gdf)),
        index=gdf.index,
        dtype="float64",
    )
    if pop_col is not None:
        populations = pd.to_numeric(gdf[pop_col], errors="coerce")
        invalid = populations.isna() | (populations <= 0)
        gdf["Population"] = populations.where(~invalid, fallback_populations)
        population_source = "Natural Earth estimates (missing values simulated)"
    else:
        gdf["Population"] = fallback_populations
        population_source = "Simulated population data (fixed seed = 0)"

    # Use an equal-area CRS to obtain square kilometres.
    # Area measured in longitude/latitude degrees is NOT a valid land area.
    projected = gdf.to_crs("EPSG:6933")
    gdf["Area_km2"] = projected.geometry.area / 1_000_000
    gdf["Density"] = gdf["Population"] / gdf["Area_km2"].replace(0, np.nan)
    gdf["Density"] = gdf["Density"].replace([np.inf, -np.inf], np.nan)
    return gdf, population_source


def styled_axes(title, *, background="white"):
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(background)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.set_axis_off()
    return fig, ax


def show_map(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_cosmetics(gdf):
    st.subheader("1. Map Cosmetics")
    st.write(
        "Change the map's visual appearance using fill colors, borders, "
        "and a background color."
    )
    fill_color = st.color_picker("Country fill color", "#87CEEB")
    background_color = st.color_picker("Map background color", "#D3D3D3")
    border_color = st.color_picker("Country border color", "#222222")
    border_width = st.slider("Border thickness", 0.1, 1.5, 0.5, 0.1)

    fig, ax = styled_axes("Styled World Map", background=background_color)
    gdf.plot(ax=ax, color=fill_color, edgecolor=border_color, linewidth=border_width)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    show_map(fig)


def plot_hierarchy(gdf):
    st.subheader("2. Map Visual Hierarchy")
    st.write(
        "Selected countries appear in red while the remaining countries "
        "stay light blue, drawing attention to the selected areas."
    )
    options = sorted(gdf["Country"].dropna().unique().tolist())
    suggested = ["Brazil", "India", "China", "Australia", "South Africa", "Russia"]
    default = [country for country in suggested if country in options]
    highlighted = st.multiselect(
        "Select countries to highlight in red", options, default=default
    )
    colors = np.where(gdf["Country"].isin(highlighted), "#DC2626", "#A6CEE3")

    fig, ax = styled_axes("Visual Hierarchy: Highlighted Countries")
    gdf.plot(ax=ax, color=colors, edgecolor="#374151", linewidth=0.35)
    ax.legend(
        handles=[
            Patch(facecolor="#DC2626", label="Selected countries"),
            Patch(facecolor="#A6CEE3", label="Other countries"),
        ],
        loc="lower left",
        frameon=True,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    show_map(fig)
    st.caption(f"Countries selected: {len(highlighted)}")


def plot_density(gdf, *, projection=None):
    is_projection = projection is not None
    st.subheader("4. Geo Projection" if is_projection else "3. Choropleth Map")
    st.write(
        "Compare estimated population density across countries. "
        "The color palette and logarithmic transformation can be changed interactively."
    )

    if is_projection:
        projection_name = st.selectbox("Select a Projection", list(PROJECTIONS.keys()))
    else:
        projection_name = None
    color_choice = st.selectbox("Choose Color Scheme", COLOR_SCHEMES, index=0)
    use_log = st.checkbox("Use Log Transformation", value=True)

    map_data = gdf.copy()
    map_data["Map_value"] = (
        np.log1p(map_data["Density"]) if use_log else map_data["Density"]
    )
    units = "log(1 + people per km²)" if use_log else "people per km²"

    if is_projection:
        # Mercator cannot represent Antarctica and very high polar latitudes well.
        if projection_name == "Mercator":
            map_data = map_data[map_data["Country"] != "Antarctica"].copy()
            st.caption("Antarctica is excluded from Mercator to avoid extreme pole distortion.")
        map_data = map_data.to_crs(PROJECTIONS[projection_name])

    title = (
        f"Population Density – {projection_name} Projection"
        if is_projection else "Population Density by Country"
    )
    fig, ax = styled_axes(title)
    map_data.plot(
        column="Map_value",
        cmap=color_choice,
        linewidth=0.28,
        edgecolor="#374151",
        legend=True,
        legend_kwds={"label": units, "shrink": 0.65},
        missing_kwds={"color": "#E5E7EB", "label": "Missing data"},
        ax=ax,
    )
    if not is_projection:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
    show_map(fig)

    st.caption(
        "Density = estimated population ÷ land area (km²). "
        "Density is calculated before map reprojection for fair comparison."
    )
    if is_projection:
        st.info(
            "Changing the projection alters how the world appears on the map; "
            "it does not change the underlying population-density values."
        )


def main():
    st.set_page_config(page_title="DV Lab 7 - Interactive Maps", layout="wide")
    st.title("Lab 7: Interactive Map Visualization Examples")
    st.caption("Nur Aidina binti Mohd Nasir | 22005974")

    st.sidebar.header("Choose a Map Type")
    map_type = st.sidebar.radio(
        "Select Map Type",
        ["Map Cosmetics", "Map Visual Hierarchy", "Choropleth Map", "Geo Projection"],
    )

    if not ensure_dataset():
        st.stop()

    try:
        gdf, population_source = load_data(str(SHAPEFILE))
    except Exception as exc:
        st.error(f"Unable to read or process the Natural Earth dataset: {exc}")
        st.stop()

    st.sidebar.success("Natural Earth dataset loaded")
    st.sidebar.caption(population_source)
    st.sidebar.caption(f"Countries/regions in dataset: {len(gdf)}")

    if map_type == "Map Cosmetics":
        plot_cosmetics(gdf)
    elif map_type == "Map Visual Hierarchy":
        plot_hierarchy(gdf)
    elif map_type == "Choropleth Map":
        plot_density(gdf)
    else:
        plot_density(gdf, projection=True)

    with st.expander("View country population and density data"):
        st.dataframe(
            gdf[["Country", "Population", "Area_km2", "Density"]]
            .sort_values("Country")
            .rename(
                columns={
                    "Area_km2": "Area (km²)",
                    "Density": "Population density (people/km²)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.caption("Map boundaries and population estimates: Natural Earth (1:110m).")


if __name__ == "__main__":
    main()
