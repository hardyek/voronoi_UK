import pandas as pd
import geopandas as gpd
import os
import sys
import time
from generator import PostcodeMapGenerator
from shapely.ops import unary_union
from shapely.validation import explain_validity

def preprocess_codepoint_open(data_path):
    """Preprocess the CodePoint Open dataset."""
    print(f"Loading CodePoint Open data from {data_path}")
    gdf = gpd.read_file(data_path)
    postcode_column = "postcode"

    print(f"Loaded {len(gdf)} postcodes from file")
    data = gdf.copy()

    # Convert to WGS84 (EPSG:4326) for latitude/longitude
    print("Converting to WGS84")
    data = data.to_crs(epsg=4326)
    
    # Extract coordinates
    data['longitude'] = data.geometry.x
    data['latitude'] = data.geometry.y

    # Ensure proper formatting with a space (e.g., "SW1A 2AA")
    print("Formatting postcodes")
    data[postcode_column] = data[postcode_column].str.strip()
    data[postcode_column] = data[postcode_column].str.replace(" ", "")
    
    # Then add space in the correct position
    data[postcode_column] = data[postcode_column].apply(
        lambda x: x[:-3] + ' ' + x[-3:] if isinstance(x, str) and len(x) > 3 else x
    )

    # Basic validation
    invalid_mask = data['longitude'].isna() | data['latitude'].isna()
    if invalid_mask.any():
        print(f"Found {invalid_mask.sum()} records with invalid coordinates")
        data = data[~invalid_mask]
    
    # UK bounds in WGS84
    uk_bounds_wgs84 = (-13.0, 49.5, 2.0, 61.0)  # (minx, miny, maxx, maxy)
    
    # Filter to only UK points
    uk_mask = (
        (data['longitude'] >= uk_bounds_wgs84[0]) & 
        (data['longitude'] <= uk_bounds_wgs84[2]) & 
        (data['latitude'] >= uk_bounds_wgs84[1]) & 
        (data['latitude'] <= uk_bounds_wgs84[3])
    )
    
    if not uk_mask.all():
        outside_uk = (~uk_mask).sum()
        print(f"Found {outside_uk} points outside UK bounds")
        data = data[uk_mask]
    
    print(f"Preprocessing complete. {len(data)} valid postcodes.")
    return data[[postcode_column, "latitude", "longitude"]]

def sample_data_if_needed(df, sample_size=None):
    """Sample the data if requested."""
    if sample_size is not None and len(df) > sample_size:
        print(f"Sampling {sample_size} records from {len(df)} total")
        return df.sample(sample_size, random_state=42)
    return df

def main():
    gpkg_file = "datasets/codepo_gb.gpkg"
    output_path = "postcode_maps"
    
    start_time = time.time()
    
    print(f"Output directory: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Preprocess data
    try:
        print("Preprocessing CodePoint Open data...")
        postcode_data = preprocess_codepoint_open(gpkg_file)

        # Load country boundaries and create GB outline
        country_path = "Supplementary_Country/country_region.shp"
        country_gdf = gpd.read_file(country_path)
        gb_outline = unary_union(country_gdf.geometry)
        
        # Initialize generator
        print("Initializing generator...")
        generator = PostcodeMapGenerator(output_path)
        
        # Load data
        print(f"Loading {len(postcode_data)} postcodes into generator...")
        generator.load_data_from_dataframe(postcode_data) 


        # Set GB outline for clipping
        generator.set_gb_outline(gb_outline, country_gdf.crs, simplify=True)
        
        # Generate maps
        print("Generating maps...")
        maps = generator.create_maps(levels=["XX", "XXNN", "XXNN-N"])
        
        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        print("\nGenerated maps:")
        for level, path in maps.items():
            print(f"  {level}: {path}")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()