import geopandas as gpd

gdf = gpd.read_file("Exposure.shp")

# Simplify geometries with tolerance
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01, preserve_topology=True)

# Keep only essential columns
gdf = gdf[['geometry', 'NAME']]  # example

# Save simplified shapefile
gdf.to_file("Exposure_new.shp")
