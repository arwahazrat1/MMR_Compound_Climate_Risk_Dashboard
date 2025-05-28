import streamlit as st
import folium
import geopandas as gpd
import os
import pandas as pd
from streamlit_folium import st_folium
import rasterio
from folium.raster_layers import ImageOverlay
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import transform_bounds
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import base64


# Set page configuration to wide mode by default
st.set_page_config(layout="wide")

# Default map center coordinates (Mumbai Coordinates)
DEFAULT_MAP_CENTER = [19.149636, 73.075294]
CITY_LOCATIONS = {"Mumbai": [19.1311, 72.8926]}


def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


st.markdown(
    """
    <style>
    .css-ffhzg2 {
        height: 100vh;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True
)

color_map = {
    "Airport": "#D3D3D3",  # Light Grey
    "Coastal Zone": "#ADD8E6",  # Light Blue
    "Eco-Sensitive Zone": "#FFFF00",  # Yellow
    "Forest Zone": "#228B22",  # Green
    "Green Zone": "#006400",  # Dark Green
    "Harbour": "#D3D3D3",  # Light Grey
    "Heritage and Conservation Area": "#FFC0CB",  # Pink
    "Industrial": "#800080",  # Purple
    "Primary Activity": "#FFFF00",  # Yellow
    "Railway": "#FFFF00",  # Yellow
    "Recreation Area": "#006400",  # Dark Green
    "Roads": "#000000",  # Black
    "Urbanisable": "#FFA500",  # Orange
    "Water Body": "#0000FF"  # Blue
}

# Layer color specifications with inverted color ramp for some layers
layer_colors = {
    'MMR Boundary': '#FF00C5',  # Dashed outline, no fill (transparent)
    'ULB Boundaries': '#4E4E4E',  # Dashed outline, no fill (transparent)
    'Tehsil Boundaries': '#4E4E4E',  # Dashed outline, no fill (transparent)
    # Line only, no dashes, no fill (transparent)
    'Village Boundaries': '#4E4E4E',
    'SPA Boundaries': '#FF00C5',  # Line, not dashed, no fill (transparent)
    'Coastal Regulatory Zone': '#005CE6',  # No outline, only fill
    'Existing Land Use': '#00FF00',  # Green, fill
    'Slum Clusters': '#E64C00',  # Outline only, not dashed, no fill
}

exposure_vulnerability_colors = {
    # Inverted colormap (RdYlBu_r) # Inverted colormap (RdYlBu_r)
    'Exposed Area': {'colormap': 'RdYlBu_r', 'low': 0, 'high': 100},
}
# Hazard layer colors
hazard_layer_colors = {
    # Example for Storm Surge
    'Storm Surge': {'colormap': 'RdYlBu_r', 'low': 0, 'high': 100},
    'Flood Depth': {'colormap': 'Blues', 'low': 0, 'high': 50},
    'Thunderstorm Frequency': {'colormap': 'Blues', 'low': 0, 'high': 200},
    'Cyclone Density': {'colormap': 'Blues', 'low': 0, 'high': 50},
    'Drought Index': {'colormap': 'Blues', 'low': -5, 'high': 5},
    'Agricultural Drought': {'colormap': 'YlOrRd', 'low': 0, 'high': 100},
    'Nighttime Heat Retention': {'colormap': 'RdYlBu_r', 'low': 0, 'high': 100},
    # Example PM10 in µg/m3
    'Wet Bulb Globe Temperature': {'colormap': 'RdYlBu_r', 'low': 0, 'high': 100},
    # Example NO2 in µg/m3
    'NO2': {'colormap': 'RdYlBu_r', 'low': 0, 'high': 100},
    # Example O3 in µg/m3
    'O3': {'colormap': 'RdYlBu_r', 'low': 0, 'high': 100},
    # Orange to Red Gradient  # Red to Green Gradient  # Categorical Layer
    'Wildfire Susceptibility': {'colormap': 'Oranges', 'low': 0, 'high': 3},
    # Blue-Yellow-Red Gradient
    'Sea Level Rise Probability': {'colormap': 'RdYlBu_r', 'low': 0, 'high': 100},
    # Blue-Yellow-Red Gradient
    'Landslide Susceptibility': {'colormap': 'RdYlBu', 'low': 0, 'high': 1},
    # Light Blue to Dark Blue Gradient
    'Sea Level Rise and Storm Surge': {'colormap': 'Blues', 'low': 0, 'high': 100},
    # Light Blue to Dark Blue Gradient
    'Sea Level Rise and Coastal Erosion': {'colormap': 'Blues', 'low': 0, 'high': 100},
    # Light Blue to Dark Blue Gradient
    'Compound Sea Level Rise and Storm Surge': {'colormap': 'Blues', 'low': 0, 'high': 100},
    # Light Blue to Dark Blue Gradient
    'Compound Sea Level Rise and Coastal Erosion': {'colormap': 'Blues', 'low': 0, 'high': 100},
}

# Tehsil coordinates
tehsil_coordinates = {
    "Vasai": [19.47, 72.80],
    "Palghar": [19.70, 72.77],
    "Mumbai City": [18.96, 72.82],
    "Uran": [18.88, 72.94],
    "Khalapur": [18.83, 73.29],
    "Kalyan": [19.24, 73.13],
    "Karjat": [18.91, 73.33],
    "Bhiwandi": [19.30, 73.06],
    "Ulhasnagar": [19.22, 73.15]
}

ulb_coordinates = {
    "Bhiwandi-Nizampur": [19.296664, 73.063121],
    "Kalyan-Dombivali": [19.238374326792115, 73.13517754446264],
    "Mira-Bhayander": [19.28960, 72.81739],
    "Thane": [19.19722, 72.97222],
    "Navi Mumbai": [19.03681, 73.01582],
    "Ambernath": [19.2090, 73.1860],
    "Kulgaon-Badlapur": [19.16266502989267, 73.2351639126853],
    "Uran": [18.8772, 72.9283],
    "Khalapur": [18.832012599951543, 73.28460221239206],
    "Khopoli": [18.7856, 73.3459],
    "Matheran": [18.9880, 73.2712],
    "Pen": [18.7373, 73.0960],
    "Panvel": [18.9907, 73.1168],
    "Alibag": [18.6511, 72.8683],
    "Karjat": [18.9107, 73.3235],
    "Vasai-Virar": [19.384135044454453, 72.87257181945087],
    "Palghar": [19.7000, 72.7000],
    "Greater Mumbai": [19.0760, 72.8777],
    "Ulhasnagar": [19.234916415736343, 73.1691130626613]
}

# Define file paths for layers based on your data
layer_file_mapping = {
    "MMR Boundary": "assets/MMR_Boundary.shp",
    "ULB Boundaries": "assets/ULB_Boundary.shp",
    "Tehsil Boundaries": "assets/Tehsil_Boundary.shp",
    "Village Boundaries": "assets/Village_Boundary.shp",
    "SPA Boundaries": "assets/SPA_Boundary.shp",
    "Coastal Regulatory Zone": "assets/CRZ_Boundary.shp",
    "Existing Land Use": "assets/Existing_Land_Use.shp",
    "Slum Clusters": "assets/CISF_boundary.shp",
    "Flood Frequency": "assets/Flood_Frequency.tif",
    "Flood Depth": "assets/Flood_depth_resampled.tif",
    "Thunderstorm Frequency": "assets/ThunderstormFrequency.tif",
    "Storm Surge": "assets/StormSurge.tif",
    "Cyclone Density": "assets/Cyclone.tif",
    "Drought Index": "assets/Drought_PDSI.tif",
    "Agricultural Drought": "assets/Drought_Crop.tif",
    "Nighttime Heat Retention": "assets/Nighttime_Heat.tif",
    "Wet Bulb Globe Temperature": "assets/WBGT.tif",
    "PM2.5": "assets/PM25.tif",
    "PM10": "assets/PM10.tif",
    "NO2": "assets/NO2.tif",
    "O3": "assets/O3.tif",
    "Wildfire Susceptibility": "assets/Wildfire.tif",
    "Deforestation and Wetland Health": "assets/DeforestationWetland.tif",
    "Coastal Erosion": "assets/Coastline.tif",
    "Riverbank Erosion": "assets/Riverbank.tif",
    "Sea Level Rise Probability": "assets/SLR.tif",
    "Landslide Susceptibility": "assets/Landslide.tif",
    "Exposed Area": "assets/Exposure.shp",
    "Socio-Economic Vulnerability": "assets/SocioVulnerability.tif",
    "Infrastructure Vulnerability": "assets/InfraVulnerability.tif",
    "Compound Sea Level Rise and Storm Surge": "assets/Comp_SLR_Storm.tif",
    "Compound Sea Level Rise and Coastal Erosion": "assets/Comp_SLR_Coast.tif",
    "Compound High WBGT and Urban Heat Island": "assets/Comp_Heat_UHI.tif",
    "Compound Heat Stress and Ozone Formation": "assets/Comp_Heat_Ozone.tif",
    "Compound Landslide and Flood": "assets/Comp_Land_Flood.tif",
    "Compound Landslide and Vegetation Loss": "assets/Comp_Land_Vegetation.tif",
    "Compound Flood and Wetland Degradation": "assets/Comp_Flood_Wetland.tif",
    "Compound Flood and Riverbank Erosion": "assets/Comp_Flood_River_Ero_resample.tif",
}

# Caching shapefile loading to avoid re-reading files


@st.cache_data
def load_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    # Rename columns without inplace to avoid side effects
    gdf = gdf.rename(columns={'name': 'NAME', 'tehsil': 'TEHSIL'})
    return gdf


@st.cache_data
def simplify_geometries(gdf, tolerance=0.001):
    # Create a copy to avoid modifying original
    gdf_copy = gdf.copy()
    gdf_copy['geometry'] = gdf_copy['geometry'].simplify(tolerance)
    return gdf_copy


def load_vector(shapefile_path, map_obj, layer_name, color="#0000FF", opacity=1.0, dashed=False):
    gdf = load_shapefile(shapefile_path)
    if gdf is None:
        return

    try:
        folium.GeoJson(
            gdf,
            name=layer_name,
            style_function=lambda: {
                'color': color,
                'weight': 2,
                'dashArray': '5, 5' if dashed else '0',
                'opacity': opacity,
                'fillOpacity': 0
            }
        ).add_to(map_obj)

    except Exception as e:
        st.error(f"❌ Error loading vector layer {layer_name}: {e}")


@st.cache_data
def load_raster(raster_path):
    """Checks if the raster file exists and loads it."""
    if not os.path.exists(raster_path):
        st.error(f"Raster file does not exist at {raster_path}")
        return None
    try:
        with rasterio.open(raster_path) as src:
            img = src.read(1)
            return img, src.bounds  # or return what you need from the raster
    except Exception as e:
        st.error(f"Error loading raster {raster_path}: {str(e)}")
        return None


def get_color(value):
    """Map raster values to a color based on the specified Red-Green gradient."""
    if 0.0995 <= value <= 1.65:
        return (0, 0, 0, 0)  # Transparent
    # Normalize value between 0 and 1
    norm_value = (value + 99.939) / (99.2745 + 99.939)
    cmap = plt.get_cmap('RdYlGn')  # Red to Green gradient
    rgba = cmap(norm_value)
    return tuple(int(c * 255) for c in rgba[:3]) + (255,)  # Convert to RGBA


def add_socio_economic_vulnerability_layer(map_obj, raster_file, transparency_value=0.7):

    if not os.path.exists(raster_file):
        st.error(f"Raster file not found: {raster_file}")
        return

    try:
        with rasterio.open(raster_file) as src:
            bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            bounds_latlon = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

            nodata_value = src.nodata if src.nodata is not None else 0
            img = src.read(1)
            img_masked = np.ma.masked_equal(img, nodata_value)

            valid_data = img_masked.compressed()
            if valid_data.size == 0:
                st.warning(
                    "No valid data in Socio-Economic Vulnerability layer.")
                return

            vmin = valid_data.min()
            vmax = valid_data.max()
            if vmin == vmax:
                st.warning(
                    "No variation in Socio-Economic Vulnerability data (min == max).")
                return

            img_norm = (img_masked - vmin) / (vmax - vmin)
            img_norm = np.clip(img_norm, 0, 1)

            # Use Green—Yellow—Red gradient
            cmap = plt.get_cmap('RdYlGn_r')  # reversed RdYlGn for Green to Red

            rgba_img = cmap(img_norm)

            alpha = np.where(img_masked.mask, 0, 1)
            rgba_img[..., -1] = alpha

            rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)

            ImageOverlay(
                image=rgba_img_uint8,
                bounds=bounds_latlon,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(f"Error loading Socio-Economic Vulnerability layer: {e}")


def add_infrastructure_vulnerability_layer(map_obj, raster_file, transparency_value=0.0000):

    if not os.path.exists(raster_file):
        st.error(f"Raster file not found: {raster_file}")
        return

    try:
        with rasterio.open(raster_file) as src:
            bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            bounds_latlon = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

            nodata_value = src.nodata if src.nodata is not None else 0
            img = src.read(1)
            img_masked = np.ma.masked_equal(img, nodata_value)

            valid_data = img_masked.compressed()
            if valid_data.size == 0:
                st.warning(
                    "No valid data in Infrastructure Vulnerability layer.")
                return

            vmin = valid_data.min()
            vmax = valid_data.max()
            if vmin == vmax:
                st.warning(
                    "No variation in Infrastructure Vulnerability data (min == max).")
                return

            img_norm = (img_masked - vmin) / (vmax - vmin)
            img_norm = np.clip(img_norm, 0, 1)

            # Blue—Yellow—Red gradient
            # reversed to match blue->yellow->red
            cmap = plt.get_cmap('RdYlBu_r')

            rgba_img = cmap(img_norm)

            alpha = np.where(img_masked.mask, 0, 1)
            rgba_img[..., -1] = alpha

            rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)

            ImageOverlay(
                image=rgba_img_uint8,
                bounds=bounds_latlon,
                # 4 decimal places exactly
                opacity=float(f"{transparency_value:.4f}"),
                interactive=True
            ).add_to(map_obj)
    except Exception as e:
        st.error(f"Error loading Infrastructure Vulnerability layer: {e}")


def add_deforestation_wetland_layer(map_obj, raster_file, transparency_value=0.7):

    try:
        if not os.path.exists(raster_file):
            st.error(f"❌ Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            nodata_value = src.nodata if src.nodata is not None else 0

            # Read raster band
            img = src.read(1)

            # Mask NoData values
            img_masked = np.ma.masked_equal(img, nodata_value)

            # Normalize data between 0 and 1 for colormap (only valid data)
            valid_data = img_masked.compressed()  # Only valid data without mask
            if valid_data.size == 0:
                st.warning(f"No valid data in raster: {raster_file}")
                return
            vmin = valid_data.min()
            vmax = valid_data.max()
            if vmin == vmax:
                st.warning(
                    f"Raster has no variation in data (min == max) in {raster_file}. Skipping.")
                return

            img_norm = (img_masked - vmin) / (vmax - vmin)
            img_norm = np.clip(img_norm, 0, 1)

            # Define bins and colors excluding the transparent range (0.0995 to 1.65)
            bins = [-99.939, -27.9284, -7.891, -2.3154, -0.764, -
                    0.3323, -0.0994, 1.6508, 7.2263, 27.2637, 99.2745]
            colors = [
                "#d73027",  # Red shades for lowest bin
                "#fc8d59",
                "#fee08b",
                "#ffffbf",
                "#d9ef8b",
                "#a6d96a",
                "#66bd63",
                "#1a9850",
                "#006837",
                "#004529"
            ]

            cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm(bins, ncolors=cmap.N, clip=True)

            # Create alpha mask for transparency: transparent if value is between 0.0995 and 1.65 OR nodata masked
            alpha = np.ones_like(img_masked, dtype=np.float32)
            transparent_mask = ((img_masked >= 0.0995) & (
                img_masked <= 1.65)) | img_masked.mask
            alpha[transparent_mask] = 0

            # Map image values to colors using colormap and norm
            filled_img = img_masked.filled(bins[0])
            rgba_img = cmap(norm(filled_img))

            rgba_img[..., -1] = alpha  # Apply alpha mask

            rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)

            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]

            ImageOverlay(
                image=rgba_img_uint8,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(
            f"❌ Error loading Deforestation and Wetland Health layer: {e}")


def add_flood_frequency_layer(map_obj, raster_file, transparency_value=0.7):
    import numpy as np
    import matplotlib.pyplot as plt
    import rasterio
    from folium.raster_layers import ImageOverlay
    import os

    try:
        if not os.path.exists(raster_file):
            st.error(f"❌ Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            img = src.read(1).astype(np.float32)
            # ✅ This line filters invalid extreme values
            img[img > 1e20] = np.nan

            nodata = src.nodata
            if nodata is not None:
                img_masked = np.ma.masked_equal(img, nodata)
            else:
                img_masked = np.ma.masked_invalid(img)

            # Normalize full data range for valid data only
            valid_data = img_masked.compressed()
            if valid_data.size == 0:
                st.warning(f"No valid data in raster: {raster_file}")
                return
            vmin, vmax = 0, 8
            if vmin == vmax:
                st.warning(
                    f"Raster has no variation in data (min == max) in {raster_file}. Skipping.")
                return

            norm = (img_masked - vmin) / (vmax - vmin)
            norm = np.clip(norm.filled(0), 0, 1)

            # Create alpha mask: fully transparent for nodata or zero or values <= 0 (adjust if needed)
            transparent_mask = (img_masked <= 0) | img_masked.mask
            alpha = np.ones_like(norm, dtype=np.float32)
            alpha[transparent_mask] = 0

            # Apply the Blue-Yellow-Red colormap
            cmap = plt.get_cmap('coolwarm')  # This is a Blue to Red colormap.
            rgba_img = cmap(norm)
            rgba_img[..., -1] = alpha  # Set transparency for 0 values

            rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)

            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]

            ImageOverlay(
                image=rgba_img_uint8,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

            st.write(
                "Flood Frequency layer added successfully with transparent borders.")

    except Exception as e:
        st.error(f"❌ Error loading Flood Frequency layer {raster_file}: {e}")


def add_flood_depth_layer(
    map_obj,
    raster_file,
    transparency_value=0.7,
    rotation_k=0,       # number of 90 degree CCW rotations, 0-3
    flip_vertical=False,
    flip_horizontal=False,
):

    if not os.path.exists(raster_file):
        st.error(f"Raster file not found: {raster_file}")
        return

    try:
        with rasterio.open(raster_file) as src:
            bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            # [[south, west], [north, east]]
            bounds_latlon = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

            nodata_value = src.nodata if src.nodata is not None else 0
            img = src.read(1)

            # Mask nodata pixels
            img_masked = np.ma.masked_equal(img, nodata_value)

            valid_data = img_masked.compressed()
            if valid_data.size == 0:
                st.warning("No valid data for Flood Depth.")
                return

            vmin, vmax = valid_data.min(), valid_data.max()
            if vmin == vmax:
                st.warning("No variation in Flood Depth data (min == max).")
                return

            # Normalize data to 0-1 for colormap
            img_norm = (img_masked - vmin) / (vmax - vmin)
            img_norm = np.clip(img_norm, 0, 1)

            # Apply rotation k times 90 degrees CCW
            img_norm = np.rot90(img_norm, k=rotation_k)
            mask = np.rot90(img_masked.mask, k=rotation_k)

            # Apply flips if specified
            if flip_vertical:
                img_norm = np.flipud(img_norm)
                mask = np.flipud(mask)
            if flip_horizontal:
                img_norm = np.fliplr(img_norm)
                mask = np.fliplr(mask)

            # Create zero mask and apply same transformations
            zero_mask = (img == 0)
            zero_mask = np.rot90(zero_mask, k=rotation_k)
            if flip_vertical:
                zero_mask = np.flipud(zero_mask)
            if flip_horizontal:
                zero_mask = np.fliplr(zero_mask)

            # Combine masks: nodata and zero both transparent
            combined_mask = mask | zero_mask

            cmap = plt.get_cmap('Blues')
            rgba_img = cmap(img_norm)

            # Set alpha 0 where masked (nodata or zero), else 1
            alpha = np.where(combined_mask, 0, 1)
            rgba_img[..., -1] = alpha

            rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)

            ImageOverlay(
                image=rgba_img_uint8,
                bounds=bounds_latlon,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(f"Error loading Flood Depth layer: {e}")


def add_wet_bulb_globe_temperature_layer(map_obj, raster_file, transparency_value=0.7):
    """Adds the Wet Bulb Globe Temperature (WBGT) layer with Blue-Yellow-Red gradient and transparent zero/NaN values."""
    try:
        if not os.path.exists(raster_file):
            st.error(f"❌ Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            img = src.read(1)  # Read the first band

            # Mask zeros and NaNs (set them transparent)
            mask = (img == 0) | np.isnan(img)
            img_masked = np.ma.array(img, mask=mask)

            # Normalize only the valid data (masked values excluded)
            min_val = img_masked.min()
            max_val = img_masked.max()
            img_normalized = (img_masked - min_val) / (max_val - min_val)

            # Apply Blue-Yellow-Red colormap (use 'RdYlBu_r' to get blue to red gradient)
            cmap = plt.get_cmap('RdYlBu_r')
            img_colored = cmap(img_normalized.filled(0))[
                :, :, :-1] * 255  # RGB channels only

            # Add alpha channel based on mask: transparent where masked
            alpha_channel = np.where(mask, 0, 255).astype(np.uint8)
            img_colored_with_alpha = np.dstack(
                (img_colored, alpha_channel)).astype(np.uint8)

            # Add as image overlay to map
            ImageOverlay(
                image=img_colored_with_alpha,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True,
            ).add_to(map_obj)

    except Exception as e:
        st.error(f"❌ Error loading WBGT layer {raster_file}: {e}")


def add_pm25_layer(map_obj, raster_file, transparency_value=0.7):
    import numpy as np
    import matplotlib.colors as mcolors
    import rasterio
    from folium.raster_layers import ImageOverlay
    import os

    try:
        if not os.path.exists(raster_file):
            st.error(f"Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            img = src.read(1)

            # Mask zeros (set as transparent)
            mask_zero = (img == 0)
            img = np.where(mask_zero, np.nan, img)

            # Normalize but reversed: flip low and high
            vmin, vmax = 31.2252, 60.2498
            # Reverse normalization
            img_normalized = 1 - ((img - vmin) / (vmax - vmin))
            img_normalized = np.clip(img_normalized, 0, 1)

            # Reversed gradient: Red → Yellow → Blue (high → low)
            colors = ["red", "yellow", "blue"]
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "RedYellowBlue", colors)

            # Apply colormap
            img_colored = cmap(img_normalized)

            # Transparency: fully transparent where data is NaN or zero
            alpha_channel = np.where(np.isnan(img_normalized), 0, 1)
            img_colored[..., -1] = alpha_channel

            # Convert to 0-255 RGBA
            img_colored_255 = (img_colored * 255).astype(np.uint8)

            # Add overlay to map
            ImageOverlay(
                image=img_colored_255,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(f"Error adding PM2.5 layer: {e}")


def add_pm10_layer(map_obj, raster_file, transparency_value=0.7):

    try:
        if not os.path.exists(raster_file):
            st.error(f"Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            img = src.read(1)

            # Set zeros as transparent by masking them
            mask_zero = (img == 0)
            img = np.where(mask_zero, np.nan, img)

            # Normalize values between 82 and 122
            vmin, vmax = 82, 122
            img_normalized = (img - vmin) / (vmax - vmin)
            img_normalized = np.clip(img_normalized, 0, 1)

            # Define Blue → Yellow → Red colormap
            colors = ["blue", "yellow", "red"]
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "BlueYellowRed", colors)

            # Apply colormap
            img_colored = cmap(img_normalized)

            # Set alpha channel to 0 where data is nan (i.e., zero in original data)
            alpha_channel = np.where(np.isnan(img_normalized), 0, 1)
            img_colored[..., -1] = alpha_channel

            # Convert to 0-255 RGBA
            img_colored_255 = (img_colored * 255).astype(np.uint8)

            # Add the image overlay to the map
            ImageOverlay(
                image=img_colored_255,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(f"Error adding PM10 layer: {e}")


def add_no2_layer(map_obj, raster_file, transparency_value=0.7):

    try:
        if not os.path.exists(raster_file):
            st.error(f"Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            img = src.read(1)

            # Mask zeros to be transparent
            mask_zero = (img == 0)
            img = np.where(mask_zero, np.nan, img)

            # Normalize between 16 and 35
            vmin, vmax = 16, 35
            img_normalized = (img - vmin) / (vmax - vmin)
            img_normalized = np.clip(img_normalized, 0, 1)

            # Blue → Yellow → Red colormap (using matplotlib named colors)
            colors = ["blue", "yellow", "red"]
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "BlueYellowRed", colors)

            img_colored = cmap(img_normalized)

            # Alpha channel: 0 where data is NaN (transparent zeros)
            alpha_channel = np.where(np.isnan(img_normalized), 0, 1)
            img_colored[..., -1] = alpha_channel

            # Convert to 0-255 RGBA
            img_colored_255 = (img_colored * 255).astype(np.uint8)

            # Add overlay to the map
            ImageOverlay(
                image=img_colored_255,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(f"Error adding NO2 layer: {e}")


def add_o3_layer(map_obj, raster_file, transparency_value=0.7):

    try:
        if not os.path.exists(raster_file):
            st.error(f"Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            img = src.read(1)

            # Mask zeros to be transparent
            mask_zero = (img == 0)
            img = np.where(mask_zero, np.nan, img)

            # Normalize between 18 and 46
            vmin, vmax = 18, 46
            img_normalized = (img - vmin) / (vmax - vmin)
            img_normalized = np.clip(img_normalized, 0, 1)

            # Blue → Yellow → Red colormap
            colors = ["blue", "yellow", "red"]
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "BlueYellowRed", colors)

            img_colored = cmap(img_normalized)

            # Alpha channel: transparent where data is NaN (zero)
            alpha_channel = np.where(np.isnan(img_normalized), 0, 1)
            img_colored[..., -1] = alpha_channel

            # Convert to 255 RGBA
            img_colored_255 = (img_colored * 255).astype(np.uint8)

            # Add overlay to map
            ImageOverlay(
                image=img_colored_255,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(f"Error adding O3 layer: {e}")


def display_landuse_on_map(map_obj, shapefile_path):
    try:
        # Load the shapefile
        gdf = gpd.read_file(shapefile_path)

        # Check if 'LANDUSE_FI' column exists
        if 'LANDUSE_FI' not in gdf.columns:
            raise ValueError("Shapefile does not contain 'LANDUSE_FI' field.")

        # Apply color based on 'LANDUSE_FI'
        def color_function(feature):
            landuse_category = feature['properties']['LANDUSE_FI']
            # Get the color for the landuse category, defaulting to white if not found
            # Default to white if not found
            return color_map.get(landuse_category, "#FFFFFF")

        # Add GeoJSON to the map with the color mapping
        folium.GeoJson(
            gdf,
            style_function=lambda feature: {
                'fillColor': color_function(feature),
                'color': 'black',  # Border color
                'weight': 1,  # Border width
                'fillOpacity': 0.7
            }
        ).add_to(map_obj)

    except Exception as e:
        st.error(f"Error: {e}")


def add_landslide_susceptibility_layer(map_obj, raster_file, transparency_value=0.7):
    try:
        if not os.path.exists(raster_file):
            st.error(f"❌ Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            img = src.read(1).astype(np.float32)

            nodata = src.nodata
            if nodata is not None:
                img_masked = np.ma.masked_equal(img, nodata)
            else:
                img_masked = np.ma.masked_invalid(img)

            # Create transparency mask: transparent where value < 0.45 or nodata
            transparent_mask = (img_masked < 0.45)
            alpha = np.ones_like(img_masked, dtype=np.float32)
            alpha[transparent_mask] = 0
            alpha[img_masked.mask] = 0

            # Normalize values between 0.47 and 0.85 for color mapping
            vmin = 0.47
            vmax = 0.85

            # Clip image to the vmin-vmax range for normalization
            img_clipped = np.clip(img_masked.filled(vmin), vmin, vmax)
            norm = (img_clipped - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0, 1)

            # Define Blue → Yellow → Red colormap
            colors = ["blue", "yellow", "red"]
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
                "BlueYellowRed", colors)

            # Apply colormap
            rgba_img = cmap(norm)
            # Apply transparency mask as alpha channel
            rgba_img[..., -1] = alpha

            # Convert RGBA to uint8 0-255
            rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)

            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]

            # Add image overlay to map
            ImageOverlay(
                image=rgba_img_uint8,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(f"❌ Error loading Landslide Susceptibility layer: {e}")


def add_compound_slr_storm_layer(map_obj, raster_file, transparency_value=0.7):
    try:
        if not os.path.exists(raster_file):
            st.error(f"❌ Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            img = src.read(1).astype(np.float32)

            nodata = src.nodata
            if nodata is not None:
                img_masked = np.ma.masked_equal(img, nodata)
            else:
                img_masked = np.ma.masked_invalid(img)

            # Transparent mask for values <= 0 or NoData
            transparent_mask = (img_masked <= 0)
            alpha = np.ones_like(img_masked, dtype=np.float32)
            alpha[transparent_mask] = 0
            alpha[img_masked.mask] = 0

            # Normalize between 0 and 100 (clip values)
            vmin = 0
            vmax = 100
            img_clipped = np.clip(img_masked.filled(vmin), vmin, vmax)
            norm = (img_clipped - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0, 1)

            # Blues colormap (light blue to dark blue)
            cmap = plt.get_cmap('Blues')

            rgba_img = cmap(norm)
            rgba_img[..., -1] = alpha  # apply transparency mask

            rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)

            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]

            ImageOverlay(
                image=rgba_img_uint8,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(
            f"❌ Error loading Compound Sea Level Rise and Storm Surge layer: {e}")


def add_compound_slr_coast_layer(map_obj, raster_file, transparency_value=0.7):
    try:
        if not os.path.exists(raster_file):
            st.error(f"❌ Raster file not found: {raster_file}")
            return
        with rasterio.open(raster_file) as src:
            img = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                img_masked = np.ma.masked_equal(img, nodata)
            else:
                img_masked = np.ma.masked_invalid(img)
            transparent_mask = (img_masked <= 0)
            alpha = np.ones_like(img_masked, dtype=np.float32)
            alpha[transparent_mask] = 0
            alpha[img_masked.mask] = 0
            vmin = 0
            vmax = 100
            img_clipped = np.clip(img_masked.filled(vmin), vmin, vmax)
            norm = (img_clipped - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0, 1)
            cmap = plt.get_cmap('Blues')
            rgba_img = cmap(norm)
            rgba_img[..., -1] = alpha
            rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            ImageOverlay(image=rgba_img_uint8, bounds=bounds,
                         opacity=transparency_value, interactive=True).add_to(map_obj)

    except Exception as e:
        st.error(
            f"❌ Error loading Compound Sea Level Rise and Coastal Erosion layer: {e}")


def add_compound_heat_uhi_layer(map_obj, raster_file, transparency_value=0.7):

    if not os.path.exists(raster_file):
        st.error(f"Raster file not found: {raster_file}")
        return

    try:
        with rasterio.open(raster_file) as src:
            img = src.read(1).astype(np.float32)
            nodata = src.nodata

            mask_zero = (img == 0)
            img_masked = np.ma.masked_where(mask_zero, img)
            if nodata is not None:
                img_masked = np.ma.masked_equal(img_masked, nodata)

            vmin, vmax = 20, 35
            img_clipped = np.clip(img_masked.filled(vmin), vmin, vmax)
            img_norm = (img_clipped - vmin) / (vmax - vmin)
            img_norm = np.clip(img_norm, 0, 1)

            colors = ["blue", "yellow", "red"]
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "BlueYellowRed", colors)
            rgba_img = cmap(img_norm)

            alpha = np.where(img_masked.mask, 0, 1)
            rgba_img[..., -1] = alpha

            rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]

            ImageOverlay(image=rgba_img_uint8, bounds=bounds,
                         opacity=transparency_value, interactive=True).add_to(map_obj)

    except Exception as e:
        st.error(f"Error loading Compound Heat UHI layer: {e}")


def add_compound_heat_ozone_layer(map_obj, raster_file, transparency_value=0.7):

    try:
        if not os.path.exists(raster_file):
            st.error(f"Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            img = src.read(1).astype(np.float32)

            mask_zero = (img == 0)
            img_masked = np.ma.masked_where(mask_zero, img)

            vmin, vmax = 0, 100

            img_clipped = np.clip(img_masked.filled(vmin), vmin, vmax)
            img_normalized = (img_clipped - vmin) / (vmax - vmin)

            # Ozone color gradient: light pink to dark red
            colors = ["#FFCCCC", "#8B0000"]
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "RedGradient", colors)

            img_colored = cmap(img_normalized)

            alpha_channel = np.where(img_masked.mask, 0, 1)
            img_colored[..., -1] = alpha_channel

            img_colored_255 = (img_colored * 255).astype(np.uint8)

            ImageOverlay(
                image=img_colored_255,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(
            f"Error adding Compound Heat Stress and Ozone Formation layer: {e}")


def add_compound_landslide_flood_layer(map_obj, raster_file, transparency_value=0.7):

    try:
        if not os.path.exists(raster_file):
            st.error(f"Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            img = src.read(1).astype(np.float32)

            mask_zero = (img == 0)
            img_masked = np.ma.masked_where(mask_zero, img)

            vmin, vmax = 0, 8

            img_clipped = np.clip(img_masked.filled(vmin), vmin, vmax)
            img_normalized = (img_clipped - vmin) / (vmax - vmin)

            # Landslide-flood color gradient: Blues
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("Blues")

            img_colored = cmap(img_normalized)

            alpha_channel = np.where(img_masked.mask, 0, 1)
            img_colored[..., -1] = alpha_channel

            img_colored_255 = (img_colored * 255).astype(np.uint8)

            ImageOverlay(
                image=img_colored_255,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(f"Error adding Compound Landslide and Flood layer: {e}")


def add_compound_landslide_vegetation_layer(map_obj, raster_file, transparency_value=0.7):

    try:
        if not os.path.exists(raster_file):
            st.error(f"Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            img = src.read(1).astype(np.float32)

            mask_zero = (img == 0)
            img_masked = np.ma.masked_where(mask_zero, img)

            vmin, vmax = -7.77989, 5.4

            img_clipped = np.clip(img_masked.filled(vmin), vmin, vmax)
            img_normalized = (img_clipped - vmin) / (vmax - vmin)

            # Landslide-vegetation gradient: Red → Green
            colors = ["red", "green"]
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "RedGreen", colors)

            img_colored = cmap(img_normalized)

            alpha_channel = np.where(img_masked.mask, 0, 1)
            img_colored[..., -1] = alpha_channel

            img_colored_255 = (img_colored * 255).astype(np.uint8)

            ImageOverlay(
                image=img_colored_255,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(
            f"Error adding Compound Landslide and Vegetation Loss layer: {e}")


def add_compound_flood_wetland_layer(map_obj, raster_file, transparency_value=0.7):
    try:
        if not os.path.exists(raster_file):
            st.error(f"Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            img = src.read(1).astype(np.uint8)

            mask_zero = (img == 0)
            img_masked = np.ma.masked_where(mask_zero, img)

            rgba_img = np.zeros(
                (img_masked.shape[0], img_masked.shape[1], 4), dtype=np.uint8)
            blue = [0, 0, 255, 255]
            rgba_img[img_masked == 100] = blue

            ImageOverlay(
                image=rgba_img,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True,
            ).add_to(map_obj)

    except Exception as e:
        st.error(
            f"Error adding Compound Flood and Wetland Degradation layer: {e}")


def add_compound_flood_riverbank_layer(map_obj, raster_file, transparency_value=0.7):
    if not os.path.exists(raster_file):
        st.error(f"Raster file not found: {raster_file}")
        return

    try:
        with rasterio.open(raster_file) as src:
            bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            bounds_latlon = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

            nodata_value = src.nodata
            img = src.read(1)

            mask_transparent = np.zeros_like(img, dtype=bool)
            if nodata_value is not None:
                mask_transparent |= (img == nodata_value)
            mask_transparent |= (img == 0)
            mask_transparent |= np.isnan(img)

            img_masked = np.ma.masked_array(img, mask=mask_transparent)
            valid_data = img_masked.compressed()

            if valid_data.size == 0:
                st.warning(
                    "No valid data in Compound Flood and Riverbank Erosion layer.")
                return

            vmin, vmax = valid_data.min(), valid_data.max()
            if vmin == vmax:
                st.warning(
                    "No variation in Compound Flood and Riverbank Erosion data (min == max).")
                return

            img_norm = (img_masked - vmin) / (vmax - vmin)
            img_norm = np.clip(img_norm, 0, 1)

            colors = [(0.8, 0.9, 1.0), (0.0, 0.0, 0.5)]
            cmap = LinearSegmentedColormap.from_list("custom_blues", colors)

            rgba_img = cmap(img_norm)

            alpha_channel = np.where(mask_transparent, 0, 1)
            rgba_img[..., -1] = alpha_channel

            # Use the image as is — no flipping or rotation
            rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)

            ImageOverlay(
                image=rgba_img_uint8,
                bounds=bounds_latlon,
                opacity=transparency_value,
                interactive=True,
            ).add_to(map_obj)

    except Exception as e:
        st.error(
            f"Error adding Compound Flood and Riverbank Erosion layer: {e}")


def add_raster_layer(map_obj, raster_file, color_scheme='RdYlBu_r', transparency_value=0.7):
    try:
        if not os.path.exists(raster_file):
            st.error(f"❌ Raster file not found: {raster_file}")
            return

        with rasterio.open(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            img = src.read(1)

            # Use masked array for NoData or invalid values
            nodata = src.nodata
            if nodata is not None:
                img = np.ma.masked_equal(img, nodata)
            else:
                img = np.ma.masked_invalid(img)

            # Normalize only valid data (ignore masked)
            min_val = img.min()
            max_val = img.max()
            if min_val == max_val:
                st.warning(
                    f"Raster has no variation in data (min == max) in {raster_file}. Skipping.")
                return
            img_norm = (img - min_val) / (max_val - min_val)
            img_norm = np.clip(img_norm, 0, 1)

            # Apply colormap (RGBA floats 0-1)
            cmap = plt.get_cmap(color_scheme)
            # fill masked with 0 just for color mapping
            rgba_img = cmap(img_norm.filled(0))

            # Set alpha channel: transparent where masked or zero in original image
            alpha = (~img.mask & (img != 0)).astype(float)
            rgba_img[..., -1] = alpha

            # Convert to uint8 0-255
            rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)

            # Add overlay to map
            ImageOverlay(
                image=rgba_img_uint8,
                bounds=bounds,
                opacity=transparency_value,
                interactive=True
            ).add_to(map_obj)

    except Exception as e:
        st.error(f"❌ Error loading raster layer {raster_file}: {e}")


def add_raster_layer_categorical(map_obj, raster_file):
    try:
        with rasterio.open(raster_file) as src:
            bounds = [[src.bounds.bottom, src.bounds.left],
                      [src.bounds.top, src.bounds.right]]
            img = src.read(1)  # Read first band

            # Create RGBA image with shape (height, width, 4)
            rgba_img = np.zeros(
                (img.shape[0], img.shape[1], 4), dtype=np.uint8)

            # Set transparent where value == 0
            mask_0 = (img == 0)
            rgba_img[mask_0] = [0, 0, 0, 0]  # Fully transparent

            # Set red where value == 1
            mask_1 = (img == 1)
            rgba_img[mask_1] = [255, 0, 0, 255]  # Red, fully opaque

            # Set green where value == 2
            mask_2 = (img == 2)
            rgba_img[mask_2] = [0, 255, 0, 255]  # Green, fully opaque

            # Add image overlay to map
            ImageOverlay(image=rgba_img, bounds=bounds,
                         opacity=1).add_to(map_obj)

    except Exception as e:
        st.error(f"Error loading categorical raster layer {raster_file}: {e}")


# Loading vector layer with optional dashed lines
def load_vector(shapefile_path, map_obj, layer_name, color="#0000FF", opacity=1.0, dashed=False):
    """Loads a vector shapefile and applies custom styling with fully transparent fill and optional dashed lines."""
    try:
        gdf = gpd.read_file(shapefile_path)  # Load shapefile
        gdf.rename(columns={'name': 'NAME', 'tehsil': 'TEHSIL'}, inplace=True)
        # Simplify geometries for better performance
        gdf = gdf.simplify(tolerance=0.001)

        # Add GeoJSON layer to map with custom style
        folium.GeoJson(
            gdf,
            name=layer_name,
            style_function=lambda x: {
                'color': color,
                'weight': 2,
                # Dashed lines based on the 'dashed' argument
                'dashArray': '5, 5' if dashed else '0',
                'opacity': opacity,
                'fillOpacity': 0  # Transparent fill
            }
        ).add_to(map_obj)

    except Exception as e:
        st.error(f"❌ Error loading vector layer {layer_name}: {e}")


def sidebar():
    """Creates a sidebar with the necessary filter options."""

    # Select Extent
    extent = st.sidebar.radio(
        "Select Extent", ["MMR", "Urban Local Body", "Village"], key="sidebar_extent")

    region = None
    if extent == "Urban Local Body":
        # Load the ULB shapefile to fetch all ULB names dynamically
        # Ensure path matches your folder structure
        ulb_shapefile = "assets/ULB_Boundary.shp"
        if os.path.exists(ulb_shapefile):  # Correctly check for ULB shapefile
            try:
                gdf = gpd.read_file(ulb_shapefile)
                # Assuming 'NAME' is the field containing the ULB names
                ulb_names = gdf['NAME'].unique()
                region = st.sidebar.selectbox(
                    "Select ULB", ulb_names, key="sidebar_ulb")
            except Exception as e:
                st.error(f"⚠️ Error loading ULB shapefile: {e}")
        else:
            st.warning(f"⚠️ ULB shapefile not found at {ulb_shapefile}")

    elif extent == "Village":
        # Fetch Tehsils first since villages belong to Tehsils
        tehsil_shapefile = "assets/Tehsil_Boundary.shp"
        if os.path.exists(tehsil_shapefile):
            gdf = gpd.read_file(tehsil_shapefile)
            # Assuming 'TEHSIL' is the field containing the Tehsil names
            tehsil_names = gdf['TEHSIL'].unique()
            tehsil = st.sidebar.selectbox("Select Tehsil", list(
                tehsil_coordinates.keys()), key="sidebar_tehsil")

            # Handle dynamic fetching of villages
            try:
                village_gdf = gpd.read_file("assets/Village_Boundary.shp")
                # Assuming 'NAME' is the correct column
                villages_in_tehsil = village_gdf[village_gdf['TEHSIL'] == tehsil]['NAME'].unique(
                )
                region = st.sidebar.selectbox(
                    "Select Village", villages_in_tehsil, key="sidebar_village")
            except KeyError:
                st.error(
                    "The 'VILLAGE' column does not exist in the shapefile. Please check the column names.")
                region = None
        else:
            st.warning(f"⚠️ Tehsil shapefile not found at {tehsil_shapefile}")

    else:
        region = "Full View"

    # Select Base Layers
    base_layers = st.sidebar.multiselect(
        "Base Layers",
        ["MMR Boundary", "Tehsil Boundaries", "SPA Boundaries", "ULB Boundaries", "Village Boundaries",
         "Coastal Regulatory Zone", "Existing Land Use", "Slum Clusters"],
        key="sidebar_base_layers"
    )

    # Select Exposure and Vulnerability Layers
    exposure_layers = st.sidebar.multiselect(
        "Exposure and Vulnerability Layers",
        ["Exposed Area", "Socio-Economic Vulnerability",
            "Infrastructure Vulnerability"],
        key="sidebar_exposure_layers"
    )

    # Select Individual Hazard Layers
    hazard_layers = st.sidebar.multiselect(
        "Hazard Layers",
        ["Flood Frequency", "Flood Depth", "Thunderstorm Frequency", "Storm Surge", "Cyclone Density", "Drought Index",
         "Agricultural Drought", "Nighttime Heat Retention", "Wet Bulb Globe Temperature", "PM2.5", "PM10", "NO2",
         "O3", "Wildfire Susceptibility", "Deforestation and Wetland Health", "Coastal Erosion", "Riverbank Erosion",
         "Sea Level Rise Probability", "Landslide Susceptibility"],
        key="sidebar_hazard_layers"
    )

    # Select Compound Hazard Layers
    compound_layers = st.sidebar.multiselect(
        "Compound Layers",
        [
            "Compound Sea Level Rise and Storm Surge",
            "Compound Sea Level Rise and Coastal Erosion",
            "Compound High WBGT and Urban Heat Island",
            "Compound Heat Stress and Ozone Formation",
            "Compound Landslide and Flood",
            "Compound Landslide and Vegetation Loss",
            "Compound Flood and Wetland Degradation",
            "Compound Flood and Riverbank Erosion"
        ],
        key="sidebar_compound_layers"
    )

    return extent, region, base_layers, exposure_layers, hazard_layers, compound_layers


def show_map(center, zoom_start):
    """Creates a Folium map centered on Mumbai with the desired zoom level."""
    city_map = folium.Map(
        location=center, zoom_start=zoom_start, tiles="CartoDB Positron")
    return city_map


def load_coordinates_from_csv(file_path):
    """Loads village and tehsil coordinates from a CSV file."""
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Ensure that the expected columns exist in the DataFrame
        if not all(col in df.columns for col in ['NAME', 'TEHSIL', 'centroidY_decimal', 'centroidX_decimal']):
            st.error("❌ Missing expected columns in the CSV file.")
            return {}, {}

        # Create dictionaries for village and tehsil coordinates
        village_coordinates = {row['NAME']: [
            row['centroidY_decimal'], row['centroidX_decimal']] for _, row in df.iterrows()}
        tehsil_coordinates = {row['TEHSIL']: [
            row['centroidY_decimal'], row['centroidX_decimal']] for _, row in df.iterrows()}

        return village_coordinates, tehsil_coordinates

    except Exception as e:
        st.error(f"❌ Error loading CSV file: {e}")
        return {}, {}


def add_markers_from_coordinates(map_obj, coordinates, names):
    """Adds markers to the map with names as popups."""
    for name, coords in coordinates.items():
        lat, lon = coords
        folium.Marker(
            location=[lat, lon],
            popup=f"<strong>{name}</strong>",  # Display name as popup
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(map_obj)

# Function to get the extent and zoom level


def get_extent_coordinates(extent, region=None):
    """Return coordinates and zoom level based on extent selection."""

    # If the extent is for Mumbai Metropolitan Region (MMR)
    if extent == "MMR":
        return [19.0760, 72.8777], 10  # Mumbai metropolitan region coordinates

    # If the extent is for Urban Local Body (ULB)
    elif extent == "Urban Local Body" and region:
        # Use predefined ULB coordinates from the dictionary
        # Default to Mumbai if not found
        center = ulb_coordinates.get(region, [19.0760, 72.8777])
        zoom_level = 12  # Default zoom level for ULB
        return center, zoom_level

    # If the extent is for Village
    elif extent == "Village" and region:
        try:
            village_coordinates, _ = load_coordinates_from_csv("assets/v.csv")
            if region in village_coordinates:
                center = village_coordinates[region]
                zoom_level = 14  # Default zoom level for villages
                return center, zoom_level
            else:
                st.warning(
                    f"⚠️ Village '{region}' not found in the coordinates file.")
                return [19.0760, 72.8777], 10  # Default if no Village found
        except Exception as e:
            st.error(f"❌ Error loading village coordinates: {e}")
            return [19.0760, 72.8777], 10  # Default if error occurs

    # If the extent is for Tehsil and region is provided
    elif extent == "Tehsil" and region:
        # Fetch Tehsil coordinates from the predefined dictionary
        if region in tehsil_coordinates:
            center = tehsil_coordinates[region]
            zoom_level = 12  # Default zoom level for Tehsils
            return center, zoom_level
        else:
            st.warning(
                f"⚠️ Tehsil '{region}' not found in the predefined coordinates.")
            return [19.0760, 72.8777], 10  # Default if no Tehsil found

    # Default case if no valid region
    else:
        return [19.1152, 73.2257], 10  # Default coordinates if no valid region


def main():
    """Main function to display the dashboard and handle user interactions."""
    st.markdown("<h1 style='text-align: center;'>Compound Climate Risk Assessment: Mumbai Metropolitan Region</h1>", unsafe_allow_html=True)

    # Get user selections from the sidebar
    extent, region, base_layers, exposure_layers, hazard_layers, compound_layers = sidebar()

    # Get map center and zoom level based on extent and region
    center, zoom_level = get_extent_coordinates(extent, region)

    # Generate the map
    city_map = show_map(center, zoom_level)

    # Load base layers (vector layers)
    for base_layer in base_layers:
        if base_layer in layer_file_mapping:
            file_path = layer_file_mapping[base_layer]

            if base_layer == "Existing Land Use":
                # Display 'Existing Land Use' with color mapping based on LANDUSE_FI field
                display_landuse_on_map(city_map, file_path)
            else:
                # Default color if not found
                color = layer_colors.get(base_layer, "#0000FF")
                # Example dashed layers
                dashed = base_layer in ["MMR Boundary",
                                        "Tehsil Boundaries", "ULB Boundaries"]
                load_vector(file_path, city_map, base_layer,
                            color=color, dashed=dashed)

    # Load exposure and vulnerability layers after the base layers
    for exposure_layer in exposure_layers:
        if exposure_layer in layer_file_mapping:
            file_path = layer_file_mapping[exposure_layer]

            if exposure_layer == "Socio-Economic Vulnerability":
                add_socio_economic_vulnerability_layer(
                    city_map, file_path, transparency_value=0.7)

            elif exposure_layer == "Infrastructure Vulnerability":
                add_infrastructure_vulnerability_layer(
                    city_map, file_path, transparency_value=0.7)

            elif exposure_layer == "Exposed Area":
                load_vector(
                    file_path,
                    city_map,
                    exposure_layer,
                    color="#FF0000",   # Solid red
                    opacity=0.7,
                    dashed=False
                )
            elif file_path.endswith(".shp"):
                load_vector(file_path, city_map, exposure_layer)
            else:
                # fallback for other raster types
                exposure_config = exposure_vulnerability_colors.get(
                    exposure_layer, {'colormap': 'RdYlBu_r'})
                add_raster_layer(
                    city_map, file_path, color_scheme=exposure_config['colormap'], transparency_value=0.7)

       # Load hazard layers
    for hazard_layer in hazard_layers:
        if hazard_layer in layer_file_mapping:
            file_path = layer_file_mapping[hazard_layer]
            hazard_config = hazard_layer_colors.get(hazard_layer, {})
            color_scheme = hazard_config.get('colormap', 'RdYlBu_r')

            # For specific hazard layers, use the defined functions
            if hazard_layer == "Landslide Susceptibility":
                # Your new function call here:
                add_landslide_susceptibility_layer(
                    city_map, file_path, transparency_value=0.7)
            elif hazard_layer == "Flood Depth":
                add_flood_depth_layer(
                    city_map, file_path, transparency_value=0.7)

            elif hazard_layer == "O3":
                add_o3_layer(city_map, file_path, transparency_value=0.7)

            elif hazard_layer == "NO2":
                add_no2_layer(city_map, file_path, transparency_value=0.7)
            elif hazard_layer == "PM10":
                add_pm10_layer(city_map, file_path, transparency_value=0.7)

            elif hazard_layer == "PM2.5":
                add_pm25_layer(city_map, file_path, transparency_value=0.7)

            elif hazard_layer == "Wet Bulb Globe Temperature":
                add_wet_bulb_globe_temperature_layer(
                    city_map, file_path, transparency_value=0.7)

            elif hazard_layer == "Flood Frequency":
                # Apply the new function for the "Flood Frequency" layer
                add_flood_frequency_layer(
                    city_map, file_path, transparency_value=0.7)
            elif hazard_layer == "Deforestation and Wetland Health":
                # Apply the new function for the "DeforestationWetland" layer
                add_deforestation_wetland_layer(
                    city_map, file_path, transparency_value=0.7)
            elif hazard_layer in ['Coastal Erosion', 'Riverbank Erosion']:
                # For categorical layers like 'Coastal Erosion' or 'Riverbank Erosion', use the categorical function
                add_raster_layer_categorical(city_map, file_path)
            else:
                # For other hazard layers, use the regular raster layer function
                add_raster_layer(
                    city_map, file_path, color_scheme=color_scheme, transparency_value=0.7)

     # Load compound layers (specifically for compound layers)
    for compound_layer in compound_layers:
        if compound_layer in layer_file_mapping:
            file_path = layer_file_mapping[compound_layer]

            if compound_layer == "Compound High WBGT and Urban Heat Island":
                add_compound_heat_uhi_layer(
                    city_map, file_path, transparency_value=0.7)
            elif compound_layer == "Compound Heat Stress and Ozone Formation":
                add_compound_heat_ozone_layer(
                    city_map, file_path, transparency_value=0.7)
            elif compound_layer == "Compound Landslide and Flood":
                add_compound_landslide_flood_layer(
                    city_map, file_path, transparency_value=0.7)
            elif compound_layer == "Compound Landslide and Vegetation Loss":
                add_compound_landslide_vegetation_layer(
                    city_map, file_path, transparency_value=0.7)
            elif compound_layer == "Compound Flood and Wetland Degradation":
                add_compound_flood_wetland_layer(
                    city_map, file_path, transparency_value=0.7)
            elif compound_layer == "Compound Flood and Riverbank Erosion":
                add_compound_flood_riverbank_layer(
                    city_map, file_path, transparency_value=0.7)
            elif compound_layer == "Compound Sea Level Rise and Storm Surge":
                add_compound_slr_storm_layer(
                    city_map, file_path, transparency_value=0.7)
            elif compound_layer == "Compound Sea Level Rise and Coastal Erosion":
                add_compound_slr_coast_layer(
                    city_map, file_path, transparency_value=0.7)

    # Display map in Streamlit
    # make sure the path and filename are correct
    st_folium(city_map, width=1300, height=550)

    # Inject CSS to remove padding/margin and prevent vertical scroll
    st.markdown("""
        <style>
        .css-18e3th9 {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100vw !important;
        }
        html, body, .main {
            overflow-y: hidden;
            margin: 0;
            padding: 0;
            height: 100vh;
        }
        .full-screen-container {
            display: flex;
            flex-direction: column;
            height: 200px; /* fixed total height */
            width: 100vw;
            margin: 0;
            padding: 0;
        }
        .map-container {
            height: 550px;
            width: 100vw;
        }
        .legend-container {
            height: 200px;
            width: 100vw;
        }
        .full-width-image {
            width: 100vw !important;
            height: 40%;
            object-fit: contain;
            display: block;
            margin: 0;
            padding: 0;
        }
        </style>
    """, unsafe_allow_html=True)

    img_base64 = get_base64_image("LEGEND.jpg")
    st.markdown(
        f'<img src="data:image/jpg;base64,{img_base64}" class="full-width-image">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
