import sys

import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2, unit_type="deg"):
    """Calculate the Haversine distance between two geographic points."""
    R = 6371.0
    if unit_type == "deg":
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    elif unit_type == "arcmin":
        lat1, lon1, lat2, lon2 = np.radians([lat1 / 60, lon1 / 60, lat2 / 60, lon2 / 60])
    elif unit_type == "rad":
        pass
    else:
        print("Invalid unit_type. Use 'deg', 'arcmin', or 'rad'.")
        sys.exit(1)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def validate_coordinates(coord_str, is_latitude=True):
    """Validate that the coordinate is a valid decimal number (strictly decimal degrees) and within valid range."""
    try:
        coord_value = float(coord_str)
        if is_latitude:
            if coord_value < -90 or coord_value > 90:
                print(f"Latitude out of range: {coord_value}. Must be between -90 and 90.")
                raise ValueError()

        else:
            if coord_value < -180 or coord_value > 180:
                print(f"Longitude out of range: {coord_value}. Must be between -180 and 180.")
                raise ValueError()

        return coord_value
    except ValueError:
        print(f"Invalid coordinate format: '{coord_str}'. Coordinates must be strictly decimal degrees and within valid range.")
        sys.exit(1)

def read_and_clean_data(csv_path, column_names):
    """Read CSV files and clean data (drop NaN and Inf values, parse and validate coordinates)."""
    
    lat_column, lon_column = column_names[csv_path]
    df = pd.read_csv(csv_path, usecols=[lat_column, lon_column])
    
    try:
        df[lat_column] = df[lat_column].apply(lambda x: validate_coordinates(x, is_latitude=True))
        df[lon_column] = df[lon_column].apply(lambda x: validate_coordinates(x, is_latitude=False))
    except ValueError as e:
        raise ValueError(f"Error parsing coordinates in file {csv_path}: {str(e)}")
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[lat_column, lon_column])
        
    lat_arr = df.loc[:, lat_column].to_numpy()
    lon_arr = df.loc[:, lon_column].to_numpy()
    
    valid_mask_1 = ~np.isnan(lat_arr) & ~np.isnan(lon_arr) & ~np.isinf(lat_arr) & ~np.isinf(lon_arr)
    lat_arr = lat_arr[valid_mask_1]
    lon_arr = lon_arr[valid_mask_1]
    
    return lat_arr, lon_arr

def find_closest_point(lat_arr, lon_arr, point):
    """Find the closest point to the given point."""
    lat1, lon1 = point
    min_dist = float('inf')
    closest_lat, closest_lon = None, None
    
    for lat2, lon2 in zip(lat_arr, lon_arr):
        dist = haversine(lat1, lon1, lat2, lon2)
        if dist < min_dist:
            min_dist = dist
            closest_lat, closest_lon = lat2, lon2
            
    return float(closest_lat), float(closest_lon)

def match_points(csv_paths, column_names, point=None):
    """Match each point in the first array to the closest one in the second array (CSV)."""
    if point is not None:
        lat_arr, lon_arr = read_and_clean_data(csv_paths, column_names)
        return find_closest_point(lat_arr, lon_arr, point)
    else:
        lat_arr_1, lon_arr_1 = read_and_clean_data(csv_paths[0], column_names)
        lat_arr_2, lon_arr_2 = read_and_clean_data(csv_paths[1], column_names)
        closest_points = []
        for lat1, lon1 in zip(lat_arr_1, lon_arr_1):
            closest_lat, closest_lon = find_closest_point(lat_arr_2, lon_arr_2, (lat1, lon1))
            closest_points.append((closest_lat, closest_lon))
        
        return closest_points

if __name__ == "__main__":
    # example usage
    point = (1, 1)
    csv_paths = ['airports.csv', 'countries.csv']
    
    column_names = {
        'airports.csv': ('latitude', 'longitude'),
        'countries.csv': ('lat', 'lng')
    }
    
    print(match_points(csv_paths, column_names))
    print(match_points(csv_paths[0], column_names, point))
