import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import re
import os
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from scipy.spatial import Voronoi, ConvexHull

class PostcodeMapGenerator:
    def __init__(self, output_path="output"):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        # Define coordinate reference systems
        self.bng_crs = CRS.from_epsg(27700)  # British National Grid
        self.wgs_crs = CRS.from_epsg(4326)   # WGS84
        self.web_mercator_crs = CRS.from_epsg(3857)  # Web Mercator

        # Create transformer for WGS84 to Web Mercator
        self.transformer_wsg84_mercator = Transformer.from_crs(
            self.wgs_crs, 
            self.web_mercator_crs, 
            always_xy=True
        )
        
        # Define UK bounds in Web Mercator (EPSG:3857)
        self.uk_bounds = (-1200000, 6000000, 200000, 8500000)  # (minx, miny, maxx, maxy)

    def load_data_from_file(self, fp=None):
        """Load data from a space-separated text file."""
        self.data = pd.read_csv(
            fp,
            sep=r"\s+",
            header=None,
            names=["latitude", "longitude", "postcode"],
            engine="python"
        )
        
        # Transform coordinates to Web Mercator
        mercator_coords = [
            self.transformer_wsg84_mercator.transform(lon, lat) 
            for lon, lat in zip(self.data["longitude"], self.data["latitude"])
        ]
        self.data["x"] = [coord[0] for coord in mercator_coords]
        self.data["y"] = [coord[1] for coord in mercator_coords]

        # Remove any rows with invalid coordinates
        self.data = self.data.dropna(subset=["x", "y"])
        
        # Filter to only include points within or near the UK
        buffer = 50000  # 50km buffer
        expanded_bounds = (
            self.uk_bounds[0] - buffer,
            self.uk_bounds[1] - buffer,
            self.uk_bounds[2] + buffer,
            self.uk_bounds[3] + buffer
        )
        
        mask = (
            (self.data["x"] >= expanded_bounds[0]) & 
            (self.data["x"] <= expanded_bounds[2]) & 
            (self.data["y"] >= expanded_bounds[1]) & 
            (self.data["y"] <= expanded_bounds[3])
        )
        self.data = self.data[mask].reset_index(drop=True)

        print(f"Loaded {len(self.data)} valid postcode points")
        return self.data
    
    def load_data_from_dataframe(self, df):
        """Load data from an existing DataFrame."""
        self.data = df.copy()

        # Transform coordinates to Web Mercator
        mercator_coords = [
            self.transformer_wsg84_mercator.transform(lon, lat) 
            for lon, lat in zip(df["longitude"], df["latitude"])
        ]
        self.data["x"] = [coord[0] for coord in mercator_coords]
        self.data["y"] = [coord[1] for coord in mercator_coords]

        # Remove any rows with invalid coordinates
        self.data = self.data.dropna(subset=["x", "y"])
        
        # Filter to only include points within or near the UK
        buffer = 50000  # 50km buffer
        expanded_bounds = (
            self.uk_bounds[0] - buffer,
            self.uk_bounds[1] - buffer,
            self.uk_bounds[2] + buffer,
            self.uk_bounds[3] + buffer
        )
        
        mask = (
            (self.data["x"] >= expanded_bounds[0]) & 
            (self.data["x"] <= expanded_bounds[2]) & 
            (self.data["y"] >= expanded_bounds[1]) & 
            (self.data["y"] <= expanded_bounds[3])
        )
        self.data = self.data[mask].reset_index(drop=True)

        print(f"Loaded {len(self.data)} valid postcode points")
        return self.data
    
    def set_gb_outline(self, outline_geometry, source_crs, simplify):
        """Set the Great Britain outline for clipping postcode polygons.
        
        Args:
            outline_geometry: The geometry representing the GB outline (Shapely geometry)
            source_crs: The coordinate reference system of the input geometry
        
        Returns:
            The processed GB outline geometry in Web Mercator projection
        """
        # Convert to Web Mercator projection (same as our Voronoi diagram)
        gb_outline_gdf = gpd.GeoDataFrame(geometry=[outline_geometry], crs=source_crs)
        gb_outline_gdf = gb_outline_gdf.to_crs(self.web_mercator_crs)
        
        # Extract the geometry
        self.gb_outline = gb_outline_gdf.geometry.iloc[0]
        
        # If it's a MultiPolygon, ensure we have proper geometry
        if self.gb_outline.geom_type == 'MultiPolygon':
            # Ensure valid geometry
            self.gb_outline = self.gb_outline.buffer(0)

        if simplify:
            self.gb_outline = self.gb_outline.simplify(20)
        
        print(f"GB outline set for clipping (area: {self.gb_outline.area:.2f} sq units)")
        return self.gb_outline
    
    def generate_voronoi(self):
        """Generate Voronoi diagram using scipy with improved boundary handling."""
        print("Generating Voronoi diagram...")
        
        # Get points in mercator projection
        points = np.column_stack((self.data["x"], self.data["y"]))
        
        # Calculate dimensions of UK bounds
        width = self.uk_bounds[2] - self.uk_bounds[0]
        height = self.uk_bounds[3] - self.uk_bounds[1]
        
        # Create a larger bounding box (3x the size of UK) to contain all Voronoi cells
        big_bounds = (
            self.uk_bounds[0] - width,
            self.uk_bounds[1] - height,
            self.uk_bounds[2] + width,
            self.uk_bounds[3] + height
        )
        
        # Add boundary points (more points for better results)
        num_boundary_points = 200
        boundary_points = []
        
        # Add points along each edge of the bounding box
        for i in range(num_boundary_points):
            # Bottom edge
            boundary_points.append([
                big_bounds[0] + i * (big_bounds[2] - big_bounds[0]) / (num_boundary_points - 1),
                big_bounds[1]
            ])
            # Top edge
            boundary_points.append([
                big_bounds[0] + i * (big_bounds[2] - big_bounds[0]) / (num_boundary_points - 1),
                big_bounds[3]
            ])
            # Left edge
            boundary_points.append([
                big_bounds[0],
                big_bounds[1] + i * (big_bounds[3] - big_bounds[1]) / (num_boundary_points - 1)
            ])
            # Right edge
            boundary_points.append([
                big_bounds[2],
                big_bounds[1] + i * (big_bounds[3] - big_bounds[1]) / (num_boundary_points - 1)
            ])
        
        # Combine actual points with boundary points
        all_points = np.vstack([points, boundary_points])
        
        # Generate Voronoi diagram
        self.vor = Voronoi(all_points)
        
        # Store the number of actual points (not boundary points)
        self.num_actual_points = len(points)
        
        print(f"Generated Voronoi diagram with {len(self.vor.vertices)} vertices and {len(self.vor.ridge_vertices)} ridges")
        return self.vor

    def _get_postcode_pattern(self, postcode, level):
        """Extract postcode pattern at the specified level."""
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
        """Create finite polygons from Voronoi regions with improved boundary handling."""
        
        # These are approximate bounds for the UK in EPSG:3857
        uk_bounds = self.uk_bounds  # (minx, miny, maxx, maxy)

        # Get the bounds of points with some padding
        points_bounds = (
            self.data["x"].min() - 50000, # 50km padding
            self.data["y"].min() - 50000,
            self.data["x"].max() + 50000,
            self.data["y"].max() + 50000
        )

        # Use the tighter of the two bounds
        bounds = (
            max(points_bounds[0], uk_bounds[0]),
            max(points_bounds[1], uk_bounds[1]),
            min(points_bounds[2], uk_bounds[2]),
            min(points_bounds[3], uk_bounds[3])
        )

        # Create a bounding box as a shapely polygon for efficient clipping
        bounding_poly = box(*bounds)

        # Convert Voronoi diagram regions to finite polygons
        region_polys = []

        print("Processing Voronoi regions...")
        for point_idx in tqdm(range(self.num_actual_points),
                            desc="Creating polygons",
                            total=self.num_actual_points):
            # Get the region for this point
            region_idx = self.vor.point_region[point_idx]
            region = self.vor.regions[region_idx]

            # Skip empty regions
            if len(region) == 0:
                continue

            # For regions with no infinite vertex (-1), simply use the vertices
            if -1 not in region:
                try: 
                    region_vertices = self.vor.vertices[region]
                    poly = Polygon(region_vertices)

                    if poly.is_valid and poly.area > 0:
                        # Clip the polygon to the bounding box
                        clipped_poly = poly.intersection(bounding_poly)
                        if not clipped_poly.is_empty and clipped_poly.area > 0:
                            region_polys.append((point_idx, clipped_poly))
                except Exception as e:
                    print(f"Error creating polygon for point {point_idx}: {e}")
                    continue
            else:
                # For regions with infinite vertex, we need to find all ridges
                # that involve this point and extract vertices
                vertices = []

                # Find all ridges for this point
                for ridge_idx, (p1, p2) in enumerate(self.vor.ridge_points):
                    if p1 == point_idx or p2 == point_idx:
                        # Get the vertices of this ridge
                        v1, v2 = self.vor.ridge_vertices[ridge_idx]
                        
                        # Add finite vertices to our collection
                        if v1 != -1:
                            vertices.append(self.vor.vertices[v1])
                        if v2 != -1:
                            vertices.append(self.vor.vertices[v2])
                
                # Only proceed if we have at least 3 vertices
                if len(vertices) >= 3:
                    try:
                        # Create a convex hull from these vertices
                        hull = ConvexHull(vertices)
                        hull_vertices = [vertices[i] for i in hull.vertices]
                        poly = Polygon(hull_vertices)
                        
                        # Clip the polygon to the bounding box
                        clipped_poly = poly.intersection(bounding_poly)
                        if not clipped_poly.is_empty and clipped_poly.area > 0:
                            region_polys.append((point_idx, clipped_poly))
                    except Exception as e:
                        print(f"Error creating hull for point {point_idx}: {e}")
                        continue

        print(f"Created {len(region_polys)} finite polygons")
        return region_polys
    
    def create_postcode_polygons(self, level="XX"):
        """Create polygons for postcodes at the specified level."""
        # Generate Voronoi diagram if not already done
        if not hasattr(self, "vor"):
            print("Voronoi diagram not generated. Generating now...")
            self.generate_voronoi()

        # Generate region polygons if not already done
        if not hasattr(self, "region_polys"):
            print("Region polygons not generated. Generating now...")
            self.region_polys = self._finite_polygons()

        # First group polygons by pattern without unionizing
        print("Grouping polygons by postcode pattern...")
        pattern_poly_lists = {}

        for point_idx, poly in tqdm(self.region_polys, desc=f"Grouping {level}", unit="polygons"):
            # Get the pattern for this point
            try:
                postcode = self.data.iloc[point_idx]["postcode"]
                pattern = self._get_postcode_pattern(postcode, level)
                
                # Only include valid patterns
                if pattern:
                    if pattern not in pattern_poly_lists:
                        pattern_poly_lists[pattern] = []
                    
                    # Clean the polygon before adding
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    
                    if poly.is_valid and poly.area > 0:
                        pattern_poly_lists[pattern].append(poly)
            except Exception as e:
                # print(f"Error processing polygon for point {point_idx}: {e}")
                continue

        # Now unionize the polygons for each pattern more efficiently
        print("Unionizing polygons for each pattern...")
        postcode_polys = {}

        for pattern, poly_list in tqdm(pattern_poly_lists.items(), desc="Creating unions", unit="patterns"):
            try:
                # Skip patterns with no polygons
                if len(poly_list) == 0:
                    continue
                    
                # Use unary_union which is faster than sequential unions
                union_poly = unary_union(poly_list)
                
                # Clean up any invalid geometries
                if not union_poly.is_valid:
                    union_poly = union_poly.buffer(0)
                    
                # Skip invalid or empty geometries
                if not union_poly.is_valid or union_poly.is_empty or union_poly.area == 0:
                    continue

                # Process MultiPolygons to remove small disconnected pieces
                if union_poly.geom_type == "MultiPolygon":
                    # Sort polygons by area
                    sorted_polys = sorted(union_poly.geoms, key=lambda p: p.area, reverse=True)
                    
                    # Calculate area threshold (5% of largest polygon)
                    largest_area = sorted_polys[0].area
                    area_threshold = largest_area * 0.05
                    
                    # Keep only significant polygons
                    significant_polys = [p for p in sorted_polys if p.area > area_threshold]
                    
                    # If we have significant polygons, create new union
                    if significant_polys:
                        union_poly = unary_union(significant_polys)
                    else:
                        # If all polygons are small, just keep the largest
                        union_poly = sorted_polys[0]
                
                # Add to results
                postcode_polys[pattern] = union_poly
                
            except Exception as e:
                # print(f"Error creating union for pattern {pattern}: {e}")
                continue
    
        print(f"Created {len(postcode_polys)} unique {level} patterns")

        if hasattr(self, 'gb_outline'):
            print("Clipping polygons to UK outline...")
            clipped_polys = {}
            for pattern, poly in tqdm(postcode_polys.items(), desc="Clipping to GB outline", unit="polygons"):
                clipped = poly.intersection(self.gb_outline)
                if not clipped.is_empty and clipped.area > 0:
                    clipped_polys[pattern] = clipped

            postcode_polys = clipped_polys
            print(f"Clipped to UK outline, {len(postcode_polys)} polygons remain")
        
        gdf = gpd.GeoDataFrame(
            {
            "postcode": list(postcode_polys.keys()), 
            "geometry": list(postcode_polys.values())
            },
            crs=self.web_mercator_crs
        )

        # Clean up any remaining invalid geometries
        valid_mask = gdf.geometry.is_valid & ~gdf.geometry.is_empty & (gdf.geometry.area > 0)
        gdf = gdf[valid_mask]
        
        # Final cleaning of any remaining invalid geometries
        gdf['geometry'] = gdf.geometry.buffer(0)

        print(f"Final GeoDataFrame contains {len(gdf)} postcode polygons")
        return gdf
    
    def save_shapefile(self, gdf, level="XX"):
        """Save the GeoDataFrame as a shapefile."""
        output_path = os.path.join(self.output_path, f"postcode-{level}")

        # Convert to British National Grid (EPSG:27700) for saving
        gdf_bng = gdf.to_crs(self.bng_crs)
        gdf_bng.to_file(output_path)

        # Create and save centroids file
        centroids = gdf_bng.copy()
        centroids["geometry"] = centroids["geometry"].centroid
        centroids_path = os.path.join(self.output_path, f"postcode-{level}_cen")
        centroids.to_file(centroids_path)

        return output_path
    
    def create_maps(self, levels=None):
        """Create maps for all specified postcode levels."""
        if levels is None:
            levels = ["XX", "XXNN", "XXNN-N"]

        print("Creating maps for levels:", levels)
            
        result = {}
        
        for level in levels:
            print(f"\nProcessing level: {level}")
            # Create polygons for this level
            gdf = self.create_postcode_polygons(level)
            
            # Save shapefile
            print(f"Saving level {level} shapefiles...")
            path = self.save_shapefile(gdf, level)
            result[level] = path

            print(f"Level {level} created and saved to {path}")
            
        return result