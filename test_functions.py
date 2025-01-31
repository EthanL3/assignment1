import pytest
import numpy as np
import pandas as pd
from io import StringIO
import tracemalloc
from main import haversine, match_points, validate_coordinates, read_and_clean_data, find_closest_point


def test_haversine():
    lat1, lon1 = 40.7128, -74.0060
    lat2, lon2 = 34.0522, -118.2437
    
    expected_distance = 3936
    
    # Degrees
    distance_deg = haversine(lat1, lon1, lat2, lon2, unit_type="deg")
    assert np.isclose(distance_deg, expected_distance, atol=10), f"Failed for degrees: {distance_deg}"

    # Arcminutes
    distance_arcmin = haversine(lat1 * 60, lon1 * 60, lat2 * 60, lon2 * 60, unit_type="arcmin")
    assert np.isclose(distance_arcmin, expected_distance, atol=10), f"Failed for arcminutes: {distance_arcmin}"

    # Radians
    distance_rad = haversine(np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2), unit_type="rad")
    assert np.isclose(distance_rad, expected_distance, atol=10), f"Failed for radians: {distance_rad}"

def test_validate_coordinates():
    assert validate_coordinates("40.7128", is_latitude=True) == 40.7128
    assert validate_coordinates("-74.0060", is_latitude=False) == -74.0060
    assert validate_coordinates("34.0522", is_latitude=True) == 34.0522
    assert validate_coordinates("-118.2437", is_latitude=False) == -118.2437
    
    with pytest.raises(SystemExit):
        validate_coordinates("100.0", is_latitude=True)
    with pytest.raises(SystemExit):
        validate_coordinates("-200.0", is_latitude=False)
    with pytest.raises(SystemExit):
        validate_coordinates("abc", is_latitude=True)
    with pytest.raises(SystemExit):
        validate_coordinates("40.7128 degrees N", is_latitude=True)
    with pytest.raises(SystemExit):
        validate_coordinates("48.8566 degrees", is_latitude=True)
    with pytest.raises(SystemExit):
        validate_coordinates("74.0060° W", is_latitude=False)

def test_read_and_clean_data():
    column_names = {
        "tests/test1.csv": ("latitude_column", "longitude_column"),
        "tests/test2.csv": ('lat', 'lon')
    }

    lat_arr, lon_arr = read_and_clean_data("tests/test1.csv", column_names)
    expected_lat = [12.3, -12.4, 14.5, -68.223]
    expected_lon = [-47.2, -120.3, 58.21, -48.382]
    assert np.array_equal(lat_arr, expected_lat), f"Expected {expected_lat}, but got {lat_arr}"
    assert np.array_equal(lon_arr, expected_lon), f"Expected {expected_lon}, but got {lon_arr}"

    lat_arr_1, lon_arr_1 = read_and_clean_data("tests/test1.csv", column_names)
    lat_arr_2, lon_arr_2 = read_and_clean_data("tests/test2.csv", column_names)

    expected_lat_1 = [12.3, -12.4, 14.5, -68.223]
    expected_lon_1 = [-47.2, -120.3, 58.21, -48.382]
    expected_lat_2 = [14.5, 51.5]
    expected_lon_2 = [-118.25, -0.1]

    assert np.array_equal(lat_arr_1, expected_lat_1, f"Expected {expected_lat_1}, but got {lat_arr_1}")
    assert np.array_equal(lon_arr_1, expected_lon_1, f"Expected {expected_lon_1}, but got {lon_arr_1}")
    assert np.array_equal(lat_arr_2, expected_lat_2, f"Expected {expected_lat_2}, but got {lat_arr_2}")
    assert np.array_equal(lon_arr_2, expected_lon_2, f"Expected {expected_lon_2}, but got {lon_arr_2}")

def test_find_closest_point():
    point = (13.5, 38.59)

    lat_arr = [12.3, -12.4, 14.5, -68.223]
    lon_arr = [-47.2, -120.3, 58.21, -48.382]
    closest_lat, closest_lon = find_closest_point(lat_arr, lon_arr, point)
    assert closest_lat == 14.5, f"Expected {14.5}, but got {closest_lat}"
    assert closest_lon == 58.21, f"Expected {58.21}, but got {closest_lon}"

def test_match_points_1():
    column_names = {
        'tests/Boston.csv': ('latitude', 'longitude')
    }
    point = (42.350724, -71.102878)
    closest_lat, closest_lon = match_points("tests/Boston.csv", column_names, point)
    assert closest_lat == 42.34988626501212, f"Expected {42.34988626501212}, but got {closest_lat}"
    assert closest_lon == -71.10304421603614, f"Expected {-71.10304421603614}, but got {closest_lon}"

def test_match_points_2():
    csv_paths = ['tests/Cities.csv', 'tests/world_airports.csv']
    column_names = {
        'tests/Cities.csv': ('lat', 'lon'),
        'tests/world_airports.csv': ('latitude', 'longitude')
    }
    with pytest.raises(SystemExit):  # Expecting the program to exit due to invalid data
        match_points(csv_paths, column_names)

def test_match_points_3():
    csv_paths = ['tests/Major_Cities_GPS.csv', 'tests/world_airports.csv']
    column_names = {
        'tests/Major_Cities_GPS.csv': ('Latitude', 'Longitude'),
        'tests/world_airports.csv': ('latitude', 'longitude')
    }
    with pytest.raises(SystemExit):  # Expecting the program to exit due to invalid data
        match_points(csv_paths, column_names)

def test_match_points_4():
    column_names = {
        'tests/test3.csv': ('lat', 'lon')
    }
    point = (1, 1)
    with pytest.raises(SystemExit):
        match_points("tests/test3.csv", column_names, point)
