import pandas as pd
import geopandas as gpd
import os 
import glob
from generator import PostcodeMapGenerator
import matplotlib.pyplot as plt

def preprocess_codepoint_open(data_path):
    gdf = gpd.read_file(data_path)
    postcode_column = "postcode"

    data = gdf.copy()

    # Convert to WGS84 (EPSG:4326) for latitude/longitude
    data = data.to_crs(epsg=4326)
    # Extract coordinates
    data['longitude'] = data.geometry.x
    data['latitude'] = data.geometry.y

    # Ensure proper formatting with a space (e.g., "SW1A 2AA")
    data[postcode_column] = data[postcode_column].str.strip()

    data[postcode_column] = data[postcode_column].str.replace(" ", "")
    # Then add space in the correct position
    data[postcode_column] = data[postcode_column].apply(
        lambda x: x[:-3] + ' ' + x[-3:] if isinstance(x, str) and len(x) > 3 else x
    )

    return data[
        [postcode_column, "latitude", "longitude"]
        ]

if __name__ == "__main__":
    gpkg_file = "datasets/codepo_gb.gpkg"
    output_path = "postcode_maps"
    image_output_path = "images"

    print("Preprocessing codepoint open data...")
    postcode_data = preprocess_codepoint_open(gpkg_file)
    
    print("Initialising generator class...")
    generator = PostcodeMapGenerator(output_path)

    print("Loading data into generator...")
    generator.load_data_from_dataframe(postcode_data)

    print("Generating maps..")
    generator.create_maps(levels=["XX","XXNN"])

    print("Maps created.")
    level = "XX"
    shapefile_path = os.path.join(output_path, f"postcode-{level}.shp")

    if os.path.exists(f"{shapefile_path}.shp"):
        gdf = gpd.read_file(shapefile_path)
        
        # Create a simple visualization
        fig, ax = plt.subplots(figsize=(10, 12))
        gdf.plot(
            column='color',
            categorical=True,
            linewidth=0.5,
            edgecolor='black',
            ax=ax
        )
        
        ax.set_title(f'UK Postcode Areas ({level})')
        ax.set_axis_off()
        
        # Save the figure
        plt.savefig(os.path.join(image_output_path, f"map_{level}.png"), dpi=300)
        print(f"Map visualization saved to {image_output_path}/map_{level}.png")
        
        # Show the plot
        plt.show()
