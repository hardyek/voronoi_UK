import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import re
import os
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

class Site:
    def __init__(self, x, y, tag=""):
        self.x = x
        self.y = y
        self.tag = tag

class PostcodeMapGenerator:
    def __init__(self, output_path="output"):
        
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        self.bng_crs = CRS.from_epsg(27700) # British National Grid
        self.wgs_crs = CRS.from_epsg(4326) # WGS84

        self.web_mercator_crs = CRS.from_epsg(3857) # Web Mercator

        self.transformer_wsg84_mercator = Transformer.from_crs( # WGS84 to Web Mercator Transformer
            self.wgs_crs, 
            self.web_mercator_crs, 
            always_xy=True
        )

    def load_data_from_file(self, fp = None):
        self.data = pd.read_csv(
            fp,
            sep=r"\s+",
            header=None,
            names=["latitude", "longitude", "postcode"],
            engine="python"
        )
        
        mercator_coords = [
                self.transformer_wsg84_mercator.transform(lon, lat) 
                for lon, lat in zip(self.data["longitude"], self.data["latitude"])
        ]
        self.data["x"] = [coord[0] for coord in mercator_coords]
        self.data["y"] = [coord[1] for coord in mercator_coords]

        return self.data
    
    def load_data_from_dataframe(self, df):
        self.data = df.copy()

        mercator_coords = [
            self.transformer_wsg84_mercator.transform(lon, lat) 
            for lon, lat in zip(df["longitude"], df["latitude"])
        ]
        self.data["x"] = [coord[0] for coord in mercator_coords]
        self.data["y"] = [coord[1] for coord in mercator_coords]

        return self.data
    
    def generate_voronoi(self):
        """Generate Voronoi diagram using scipy."""
        from scipy.spatial import Voronoi
        
        print("Generating Voronoi diagram...")
        points = np.column_stack((self.data["x"], self.data["y"]))
        self.vor = Voronoi(points)
        
        print(f"Generated {len(self.vor.vertices)} vertices and {len(self.vor.ridge_vertices)} ridges")
        return self.vor

    def _get_postcode_pattern(self, postcode, level):
        if level == "XX":
            match = re.match(r"^([A-Z][A-Z]?).*", postcode)
        elif level == "XXNN":
            match = re.match(r"^([A-Z][A-Z]?[0-9][A-Z0-9]?).*", postcode)
        elif level == "XXNN-N":
            match = re.match(r"^([A-Z][A-Z]?[0-9][A-Z0-9]? [0-9]).*", postcode)
        else:
            raise ValueError(f"Invalid level: {level}")
        
        return match.group(1) if match else ""
    
    def _finite_polygons(self):
        """Create finite polygons from Voronoi regions.
        
        Returns:
            List of (region_index, polygon) tuples
        """

        # Centroid as center point
        center = self.vor.points.mean(axis=0)
        # Calculate radius as 2x distance from the furthest point
        radius = 2 * np.max(np.linalg.norm(self.vor.points - center, axis=1))

        far_vertices = [] # Create set of infinite vertices far from the center
        print("Processing ridge vertices...")
        for i, (p1, p2) in tqdm(enumerate(self.vor.ridge_vertices), 
                          desc="Handling ridges",
                          total=len(self.vor.ridge_vertices)):
            if p1 == -1 or p2 == -1:
                # Get index of finite point
                p = max(p1, p2)
                # Get indices of points this ridge seperates
                p1_idx, p2_idx = self.vor.ridge_points[i]

                # Compute normal vector of the ridge
                points_mid = (self.vor.points[p1_idx] + self.vor.points[p2_idx]) / 2
                direction = self.vor.points[p2_idx] - self.vor.points[p1_idx]

                # Rotate 90 degrees and normalise
                normal = np.array([-direction[1], direction[0]])
                normal /= np.linalg.norm(normal)

                # Compute midpoint of the two points
                # Add the far point (midpoint + normal * radius)

                if np.dot(normal, center - points_mid) < 0:
                    normal = -1 * normal

                far_point = self.vor.vertices[p] + radius * normal
                far_vertices.append(far_point)
            else:
                far_vertices.append(None)

        # Create polygons for each Voronoi region
        regions_polys = []
        print("Creating region polygons...")

        print("Creating region lookup table...")
        region_to_point = {}
        for j, region_idx in tqdm(enumerate(self.vor.point_region), 
                          desc="Building lookup", 
                          total=len(self.vor.point_region)):
            region_to_point[region_idx] = j
        
        print("Processing regions...")
        for i, region in tqdm(enumerate(self.vor.regions), 
                      desc="Creating polygons", 
                      total=len(self.vor.regions)):
            if -1 in region or len(region) == 0:
                continue # Skip empty or infinite regions

            # Get the vertices of the region
            vertices = self.vor.vertices[region]

            # Create the polygon
            try:
                poly = Polygon(vertices)
                
                # Use the lookup dictionary instead of the inner loop
                point_idx = region_to_point.get(i)
                
                if point_idx is not None:
                    regions_polys.append((point_idx, poly))
            except Exception as e:
                # Sometimes invalid polygons can occur
                pass
            
        print(f"Created {len(regions_polys)} finite polygons")
        return regions_polys

    
    def create_postcode_polygons(self, level="XX"):
        """Create polygons for postcodes at the specified level.
        
        Args:
            level: Postcode level ("XX", "XXNN", or "XXNN-N")
            
        Returns:
            GeoDataFrame with the postcode polygons
        """
        # Generate Voronoi diagram if not already done
        if not hasattr(self, "vor"):
            print("Voronoi diagram not generated. Generating now...")
            self.generate_voronoi()

        #  Get finite polygons from Voronoi regions
        print("Generating finite polygons...")
        regions_polys = self._finite_polygons()

        postcode_polys = {}

        # Group polygons by postcode pattern
        print("Beginning to group polygons by postcode pattern...")
        for point_idx, poly in tqdm(regions_polys, desc=f"Processing {level}", unit="polygons"):
            postcode = self.data.iloc[point_idx]["postcode"]
            pattern = self._get_postcode_pattern(postcode, level)

            if pattern:
                if pattern in postcode_polys:
                    postcode_polys[pattern] = postcode_polys[pattern].union(poly)
                else:
                    postcode_polys[pattern] = poly

        gdf = gpd.GeoDataFrame(
            {
            "postcode": list(postcode_polys.keys()), 
            "geometry": list(postcode_polys.values())
            },
            crs=self.web_mercator_crs
        )

        return gdf
    
    def assign_colors(self, gdf):
        """Assign colors to postcodes to avoid adjacent regions having the same color.
        
        Args:
            gdf: GeoDataFrame with postcode polygons
            
        Returns:
            GeoDataFrame with color assignments
        """
        adjacency = {}

        for i, row_i in gdf.iterrows():
            for j, row_j in gdf.iterrows():
                if i != j and row_i.geometry.touches(row_j.geometry):
                    if row_i.postcode not in adjacency:
                        adjacency[row_i.postcode] = set()
                    adjacency[row_i.postcode].add(row_j.postcode)

        graph = sorted(
            adjacency.items(), 
            key=lambda x: len(x[1]) if x[1] else 0, 
            reverse=True
        )

        colors = {}
        for postcode, neighbors in graph:
            available_colors = list(range(6))  # 6 colors is sufficient
            for neighbor in neighbors:
                if neighbor in colors and colors[neighbor] in available_colors:
                    available_colors.remove(colors[neighbor])
            colors[postcode] = available_colors[0] if available_colors else 0

        color_labels = {i: chr(65 + i) for i in range(6)}

        gdf["color"] = gdf["postcode"].map(
            lambda x: color_labels[colors.get(x, 0)] if x in colors else "A"
        )

        return gdf
    
    def save_shapefile(self, gdf, level="XX"):
        output_path = os.path.join(self.output_path, f"postcode-{level}")

        gdf_bng = gdf.to_crs(self.bng_crs)
        gdf_bng.to_file(output_path)

        centroids = gdf_bng.copy()
        centroids["geometry"] = centroids["geometry"].centroid
        centroids_path = os.path.join(self.output_dir, f"postcode-{level}_cen")
        centroids.to_file(centroids_path)

        return output_path
    
    def create_maps(self, levels=None, coloured=False):
        """Create maps for all specified postcode levels.
        
        Args:
            levels: List of postcode levels to create maps for
                   Defaults to ["XX", "XXNN", "XXNN-N"]
                   
        Returns:
            Dictionary of {level: path} for saved shapefiles
        """
        if levels is None:
            levels = ["XX", "XXNN", "XXNN-N"]

        print("Creating maps for levels:", levels)
            
        result = {}
        
        for level in levels:
            # Create polygons for this level
            gdf = self.create_postcode_polygons(level)
            
            # Assign colors
            if coloured:
                gdf = self.assign_colors(gdf)
            
            # Save shapefile
            path = self.save_shapefile(gdf, level)
            result[level] = path

            print("Level", level, "created and saved to", path)
            
        return result
    
    def plot_chloropleth(self, data_gdf, value_column, title, output_path=None, figsize=(12,10), cmap="viridis", basemap=False):
        plot_gdf = data_gdf.to_crs(self.web_mercator_crs)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        plot_gdf.plot(
            column = value_column,
            cmap = cmap,
            legend = True,
            ax = ax,
            legend_kwds = {"label": value_column, "orientation": "horizontal"},
        )

        if basemap:
            try:
                import contextily as ctx
                ctx.add_basemap(ax, crs=plot_gdf.crs.to_string())
            except ImportError:
                print("Contextily is not installed. Basemap will not be added.")
            except Exception as e:
                print(f"Error adding basemap: {e}")

        ax.set_title(title, fontsize=15)
        ax.set_axis_off()

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        
        return fig