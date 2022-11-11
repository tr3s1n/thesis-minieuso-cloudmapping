import logging
import warnings

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings('ignore')

import os
import sys
import time

import pickle
import yaml
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import geopandas as gpd
import modapsclient
from planet import api as planetApi
from planet.api import downloader, filters
from pyproj import Transformer
from sentinelhub import SentinelHubCatalog, SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, \
    MimeType, bbox_to_dimensions

import ROOT as XROOT

import astropy.time
import skyfield.api
import skyfield.toposlib 
import skyfield.framelib
import skyfield.units

import matplotlib as mpl
import matplotlib.pyplot as plt

from gps_and_attitude.vectors import *
from gps_and_attitude.reader import  AttSvResolveMissing, AttSvReader
from gps_and_attitude.coordinates import interpolate_yrp_at_time_more_seconds, \
    rotate2, ellipsoid_intersect, xyz2geo, EllipsoidIntersectFailed
from gps_and_attitude.tle_searcher import TleSearcherByTimestamp
from gps_and_attitude.calc_skyfield.functions import calc_pixel_edges_on_surface, get_pixel_grid_corner_mapping
from shapely.geometry import box, Polygon, Point

# Read API keys from config.yaml file and initialize api clients

CONFIG_FILE = 'config.yaml'
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

PLANET_API_KEY = config['planet']['planet_api_key']

planetClient = planetApi.ClientV1(api_key=PLANET_API_KEY)
modisClient = modapsclient.ModapsClient()


# Setup input files and parameters

file_tle="extra_files/ISS_TLE.txt"
pixeldirection_pathname = 'extra_files/pixeldirection_v2.npy'
timestamps_df = pd.read_csv('extra_files/files_timestamps_best_label_diff_df.csv')
att_eulr_filename_format = "%04d/ISS_ATT_EULR_LVLH-%4d-%d-%d-24H.root"
tle_max_difference_days = 3
gps_time_offset = 19
planet_api_item_types = ['Sentinel2L1C', 'PSScene4Band', 'Landsat8L1G']


def get_pixels_edges_on_surface_arr(timestamp):
    datetime_at_timestamp = datetime.utcfromtimestamp(timestamp)
    
    ts = skyfield.api.load.timescale()
    ephemeris = skyfield.api.load("extra_files/de421.bsp")
    t = ts.utc(datetime_at_timestamp.year, datetime_at_timestamp.month, datetime_at_timestamp.day, 
           datetime_at_timestamp.hour, datetime_at_timestamp.minute, datetime_at_timestamp.second)
    xxt = astropy.time.Time(t.tt, format="jd", scale="tt")
    xxt.format = "unix"
    
    tle_searcher = TleSearcherByTimestamp(
        file_tle,
        only_lower=True, cache_satellite=True,
        max_difference_seconds=3600 * 24 * tle_max_difference_days
    )
    
    attitude_reader_for_interpolation = reader = AttSvReader(
        reuse_opened_files=True,
        searched_entries_cache=None,
        resolve_missing=AttSvResolveMissing.use_low_and_high_for_interpolation,
        num_search_iterations_threshold=100,
        low_high_max_distance_seconds=-1,
        att_eulr_format=att_eulr_filename_format,
        sv_gtod_format=None,  # gps position is not needed here
        gps_time_offset=gps_time_offset
    )
    
    satellite_for_pos2 = tle_searcher.find_tle(xxt.value)
    geocentric = geocentric_for_pos2 = satellite_for_pos2.at(t)
    wgs84_subpoint = skyfield.toposlib.wgs84.subpoint(geocentric)
    geoAtTime_km = wgs84_subpoint.itrs_xyz.km
    gei_at_time_pos, gei_at_time_vel = \
    geocentric.frame_xyz_and_velocity(skyfield.framelib.true_equator_and_equinox_of_date)
    sidereal_rad = 2*np.pi * t.gast / 24
    velGeoAtTime_km_per_s = rotate_z(
        gei_at_time_vel.km_per_s, 
        -sidereal_rad
    )
    
    try:
        (low_entry, high_entry), entry_found_exactly = reader.read_data(timestamp, time_offsets=(0,1))

        all_entries_found = np.all(entry_found_exactly)

        yawAtTime_deg = interpolate_yrp_at_time_more_seconds(
            timestamp + reader.gps_time_offset,
            low_entry.att_eulr.gps_unix, high_entry.att_eulr.gps_unix,
            low_entry.att_eulr.yaw, high_entry.att_eulr.yaw)

        pitchAtTime_deg = interpolate_yrp_at_time_more_seconds(
            timestamp + reader.gps_time_offset,
            low_entry.att_eulr.gps_unix, high_entry.att_eulr.gps_unix,
            low_entry.att_eulr.pitch, high_entry.att_eulr.pitch)

        rollAtTime_deg = interpolate_yrp_at_time_more_seconds(
            timestamp + reader.gps_time_offset,
            low_entry.att_eulr.gps_unix, high_entry.att_eulr.gps_unix,
            low_entry.att_eulr.roll, high_entry.att_eulr.roll)


    except Exception as e:
        # should catch if reader is None and also any reading error
        if fallback_attitude_to_0:
            yawAtTime_deg = 0
            pitchAtTime_deg = 0
            rollAtTime_deg = 0
            all_entries_found = False

            out_dict['unixtime_gps_low'] = 0
            out_dict['unixtime_gps_high'] = 0

        else:
            raise e

    pitchAtTime_rad = np.deg2rad(pitchAtTime_deg)
    rollAtTime_rad = np.deg2rad(rollAtTime_deg)
    yawAtTime_rad = np.deg2rad(yawAtTime_deg)
    
    a = skyfield.toposlib.wgs84.radius.km
    b = skyfield.toposlib.wgs84.radius.km * (1 - 1/skyfield.toposlib.wgs84.inverse_flattening)

    try:
        pixels_edges_on_surface_arr = calc_pixel_edges_on_surface(
                geoAtTime_km,
                velGeoAtTime_km_per_s,
                pitchAtTime_rad,
                rollAtTime_rad,
                yawAtTime_rad,
                pixeldirection_pathname,
                a2=a*a,
                b2=b*b
        )

    except EllipsoidIntersectFailed as e:
        print(
            '({}) Cannot calculate ellipsoid intersect for pixels: {}'.format(timestamp, str(e)))
        

    return pixels_edges_on_surface_arr


def get_minieuso_bbox_from_pixels_edges_on_surface_arr(pixels_edges_on_surface_arr):
    lon_deg_list = []
    lat_deg_list = []

    for point in pixels_edges_on_surface_arr[:,:,0].reshape((-1, 3)):
        geographic_position = geographic_position_of_distance(skyfield.toposlib.wgs84, skyfield.units.Distance(km=point))

        lon = geographic_position.longitude.degrees
        lat = geographic_position.latitude.degrees

        lon_deg_list.append(lon)
        lat_deg_list.append(lat)
        
    minieuso_bbox = [min(lon_deg_list), min(lat_deg_list), max(lon_deg_list), max(lat_deg_list)]
    return minieuso_bbox


def get_minieuso_polygon_from_pixels_edges_on_surface_arr(pixels_edges_on_surface_arr):
    # left top, right top, right bottom, left bottom
    fov_edges_list = [pixels_edges_on_surface_arr[0][0][3], pixels_edges_on_surface_arr[0][-1][4], pixels_edges_on_surface_arr[-1][-1][1], pixels_edges_on_surface_arr[-1][0][2]]
    fov_edges_lats = []
    fov_edges_lons = []

    for edge in fov_edges_list:
        edge_geographic_position = geographic_position_of_distance(
            skyfield.toposlib.wgs84, 
            skyfield.units.Distance(km=edge))
        fov_edges_lons.append(edge_geographic_position.longitude.degrees)
        fov_edges_lats.append(edge_geographic_position.latitude.degrees)

    # We need to append the first edge coords to the lists again
    edge_geographic_position = geographic_position_of_distance(
        skyfield.toposlib.wgs84, 
        skyfield.units.Distance(km=fov_edges_list[0]))
    fov_edges_lons.append(edge_geographic_position.longitude.degrees)
    fov_edges_lats.append(edge_geographic_position.latitude.degrees)

    fov_polygon_geom = Polygon(zip(fov_edges_lons, fov_edges_lats))
    crs = {'init': 'epsg:4326'}
    fov_polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[fov_polygon_geom])

    return fov_polygon


def itrs_compute_latitude(geoid, itrs_position):
    if itrs_position.center != 399:
        raise ValueError(
            'you can only calculate a geographic position from a'
            ' position which is geocentric (center=399), but this'
            ' position has a center of {0}'.format(itrs_position.center)
        )
    return itrs_distance_compute_latitude(geoid, itrs_position.itrs_xyz.au)


def itrs_distance_compute_latitude(geoid, itrs_position_distance):
    xyz_au = itrs_position_distance.au
    x, y, z = xyz_au
    a = geoid.radius.au
    e2 = geoid._e2
    R = np.sqrt(x*x + y*y)
    lat = np.arctan2(z, R)
    for iteration in 0,1,2:
        sin_lat = np.sin(lat)
        e2_sin_lat = e2 * sin_lat
        aC = a / np.sqrt(1.0 - e2_sin_lat * sin_lat)
        lat = np.arctan2(z + aC * e2_sin_lat, R)
    return xyz_au, x, y, aC, R, lat


def geographic_position_of(geoid, itrs_position):   
    return geographic_position_of_distance(geoid, itrs_position.itrs_xyz.au)


def geographic_position_of_distance(geoid, itrs_position_distance):
    xyz_au, x, y, aC, R, lat= itrs_distance_compute_latitude(geoid, itrs_position_distance)
    
    lon = (np.arctan2(y, x) - np.pi) % (np.pi * 2) - np.pi
    height_au = R / np.cos(lat) - aC
    return skyfield.toposlib.GeographicPosition(
        latitude=skyfield.units.Angle(radians=lat),
        longitude=skyfield.units.Angle(radians=lon),
        elevation=skyfield.units.Distance(height_au),
        itrs_xyz=skyfield.units.Distance(xyz_au),
        model=geoid,
    )


def planet_api_build_search_request(aoi_geom, start_date, stop_date, item_types):
    query = filters.and_filter(
        filters.geom_filter(aoi_geom),
        filters.date_range('acquired', gt=start_date),
        filters.date_range('acquired', lt=stop_date)
    )
    return filters.build_search_request(query, item_types)


def planet_api_search_data(request, client, limit=10):
    result = client.quick_search(request)
    return result.items_iter(limit=limit)

def get_intersections(step=30, delta=1800, number_of_sessions=44):
    intersections = []
    intersections_timestamps = []
    intersections_all_sources = []
    final_search_res_list = []
    Sentinel2L1C_items_num, PSScene4Band_items_num, Landsat8L1G_items_num, MYD35_L2_items_num, MOD35_L2_items_num = 0, 0, 0, 0, 0
    min_ts_diff = delta
    closest_intersection = None
    
    # For each session
    for i in range(1, number_of_sessions+1):
        
        # In the given CSV file, sessions 40 and 42 are missing so we have to skip those iterations 
        if i == 40 or i == 42:
            continue
        
        session_name = f'S{i:02d}'
        session_df = timestamps_df[timestamps_df.session_num == session_name]
        session_ts_low_values_list = [i for i in session_df['timestamp_low'].values.tolist() if i != -1]
        session_ts_high_values_list = [i for i in session_df['timestamp_high'].values.tolist() if i != -1]
        intervals = list(zip(session_ts_low_values_list, session_ts_high_values_list))

        # Get the first and the last timestamp of the session
        start_ts = session_ts_low_values_list[0]
        end_ts = session_ts_high_values_list[-1]

        print(f'---------{session_name}---------\n\n')
        
        # For each timestamp in range of the session lenght with given step in seconds 
        for timestamp in range(start_ts, end_ts, step):
            planet_api_search_requests = []

            # Check if timestamp is in any interval
            in_interval = False
            for interval in intervals:
                if interval[0] <= timestamp <= interval[1]:
                    in_interval = True

            # If it is in interval, find the bounding box and search for cloud data
            if in_interval:
                MYD35_L2_items = []
                MOD35_L2_items = []
                
                # Format inputs for APIs
                datetime_at_timestamp_start = datetime.utcfromtimestamp(timestamp - delta)
                datetime_at_timestamp_end = datetime.utcfromtimestamp(timestamp + delta)
                minieuso_bbox = get_minieuso_bbox_from_pixels_edges_on_surface_arr(get_pixels_edges_on_surface_arr(timestamp))
                bbox_s2cloudless = BBox([minieuso_bbox[0], minieuso_bbox[1], minieuso_bbox[2], minieuso_bbox[3]], crs=CRS.WGS84)
                bbox_planet = bbox_s2cloudless.geojson

                # Build planet API search requests
                for item_type in planet_api_item_types:
                    request = planet_api_build_search_request(bbox_planet, datetime_at_timestamp_start, datetime_at_timestamp_end, [item_type])
                    planet_api_search_requests.append(request)

                # Retries workaround for occasional Internal Server Error 
                for attempt in range(3):
                    try:
                        # Search for Sentinel, Landsat and PlanetScope data trough planet api client
                        Sentinel2L1C_items = list(planet_api_search_data(planet_api_search_requests[0], planetClient))
                        PSScene4Band_items = list(planet_api_search_data(planet_api_search_requests[1], planetClient))
                        Landsat8L1G_items = list(planet_api_search_data(planet_api_search_requests[2], planetClient))
                        
                        # Search for MYD35_L2 data trough modapsclient
                        modis_search_res = modisClient.searchForFiles('MYD35_L2', datetime_at_timestamp_start, datetime_at_timestamp_end, minieuso_bbox[2], minieuso_bbox[0], minieuso_bbox[3], minieuso_bbox[1], collection=61)
                        if len(modis_search_res) > 0:
                            if modis_search_res[0] != 'No results':
                                for item in modis_search_res:
                                    item_properties = modisClient.getFileProperties(item)[0]
                                    MYD35_L2_items.append(item_properties)
                    
                        # Sleep between modaps searches
                        time.sleep(2)
                    
                        # Search for MOD35_L2 data trough modapsclient
                        modis_search_res2 = modisClient.searchForFiles('MOD35_L2', datetime_at_timestamp_start, datetime_at_timestamp_end, minieuso_bbox[2], minieuso_bbox[0], minieuso_bbox[3], minieuso_bbox[1], collection=61)
                        if len(modis_search_res2) > 0:
                            if modis_search_res2[0] != 'No results':
                                for item in modis_search_res2:
                                    item_properties = modisClient.getFileProperties(item)[0]
                                    MOD35_L2_items.append(item_properties)

                        time.sleep(2)
                    except Exception as e:
                        print(f'Caught an exception: {e}')
                        print('Trying again in 10 seconds.')
                        time.sleep(1)
                    else:
                        break
                
                # Update final counters
                Sentinel2L1C_items_num += len(Sentinel2L1C_items)
                PSScene4Band_items_num += len(PSScene4Band_items)
                Landsat8L1G_items_num += len(Sentinel2L1C_items)
                MYD35_L2_items_num += len(MYD35_L2_items)
                MOD35_L2_items_num += len(MOD35_L2_items)
                
                # Check if we found intersection with all data sources at the same time
                if len(Sentinel2L1C_items) > 0 and len(PSScene4Band_items) > 0 and len(Sentinel2L1C_items) > 0 and (len(MYD35_L2_items) > 0 or len(MOD35_L2_items) > 0):
                    intersections_all_sources.append((timestamp, [len(Sentinel2L1C_items), len(PSScene4Band_items), len(Landsat8L1G_items), len(MYD35_L2_items), len(MOD35_L2_items)]))
                        
                # Group items from planet client search into one list
                planet_intersections = Sentinel2L1C_items + PSScene4Band_items + Landsat8L1G_items
                        
                # If there is any intersection found, add the current timestamp to the list intersections_timestamps 
                if len(planet_intersections) > 0 or len(MYD35_L2_items) > 0 or len(MOD35_L2_items) > 0:
                    intersections_timestamps.append(timestamp)
                
                # Get List of Search results from both clients
                ts_search_res = [len(Sentinel2L1C_items), len(PSScene4Band_items), len(Landsat8L1G_items), len(MYD35_L2_items), len(MOD35_L2_items)]
                final_search_res_list.append(ts_search_res)
                
                # Deal with Planet client search results 
                if len(planet_intersections) > 0:
                    for item in planet_intersections:
                        item_acquired_dt = datetime.strptime(item['properties']['acquired'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(microsecond=0)
                        item_acquired_ts = int(item_acquired_dt.replace(tzinfo=timezone.utc).timestamp())
                        ts_diff = abs(item_acquired_ts - timestamp)
                        info_dict = {'bbox': minieuso_bbox, 'search_result_list': ts_search_res, 'delta': delta, 'item_type': item['properties']['item_type'], 'timestamp': timestamp, 'ts_diff': ts_diff}

                        if ts_diff < min_ts_diff:
                            min_ts_diff = ts_diff
                            closest_intersection = [item, info_dict]

                        intersections.append([item, info_dict])

                # Deal with Modis search results
                if len(MYD35_L2_items) > 0:
                    MYD03_items = []
                    myd03_search_res = modisClient.searchForFiles('MYD03', datetime_at_timestamp_start, datetime_at_timestamp_end, minieuso_bbox[2], minieuso_bbox[0], minieuso_bbox[3], minieuso_bbox[1], collection=61)
                    if len(myd03_search_res) > 0:
                        if myd03_search_res[0] != 'No results':
                            for item in myd03_search_res:
                                myd03_item_properties = modisClient.getFileProperties(item)[0]
                                MYD03_items.append(myd03_item_properties)
                                 
                    # for item in MYD35_L2_items:
                    for i in range(len(MYD35_L2_items)):   
                        item_acquired_dt = datetime.strptime(MYD35_L2_items[i]['startTime'], '%Y-%m-%d %H:%M:%S.%f').replace(microsecond=0)
                        item_acquired_ts = int(item_acquired_dt.replace(tzinfo=timezone.utc).timestamp())
                        ts_diff = abs(item_acquired_ts - timestamp)
                        info_dict = {'bbox': minieuso_bbox, 'search_result_list': ts_search_res, 'delta': delta, 'item_type': 'MYD35_L2', 'timestamp': timestamp, 'ts_diff': ts_diff, 'geoloc_fileId': MYD03_items[i]['fileId']}

                        if ts_diff < min_ts_diff:
                            min_ts_diff = ts_diff
                            closest_intersection = [MYD35_L2_items[i], info_dict]

                        intersections.append([MYD35_L2_items[i], info_dict])
                    time.sleep(1)
                        
                if len(MOD35_L2_items) > 0:
                    MOD03_items = []
                    mod03_search_res = modisClient.searchForFiles('MOD03', datetime_at_timestamp_start, datetime_at_timestamp_end, minieuso_bbox[2], minieuso_bbox[0], minieuso_bbox[3], minieuso_bbox[1], collection=61)
                    if len(mod03_search_res) > 0:
                        if mod03_search_res[0] != 'No results':
                            for item in mod03_search_res:
                                mod03_item_properties = modisClient.getFileProperties(item)[0]
                                MOD03_items.append(mod03_item_properties)
                                 
                    # for item in MOD35_L2_items:
                    for i in range(len(MOD35_L2_items)):   
                        item_acquired_dt = datetime.strptime(MOD35_L2_items[i]['startTime'], '%Y-%m-%d %H:%M:%S.%f').replace(microsecond=0)
                        item_acquired_ts = int(item_acquired_dt.replace(tzinfo=timezone.utc).timestamp())
                        ts_diff = abs(item_acquired_ts - timestamp)
                        info_dict = {'bbox': minieuso_bbox, 'search_result_list': ts_search_res, 'delta': delta, 'item_type': 'MOD35_L2', 'timestamp': timestamp, 'ts_diff': ts_diff, 'geoloc_fileId': MOD03_items[i]['fileId']}

                        if ts_diff < min_ts_diff:
                            min_ts_diff = ts_diff
                            closest_intersection = [MOD35_L2_items[i], info_dict]

                        intersections.append([MOD35_L2_items[i], info_dict])
                    time.sleep(1)
                        
                        
                print(minieuso_bbox)
                print(datetime_at_timestamp_start, datetime_at_timestamp_end)
                print(ts_search_res)
                print('\n')
                
                # Sleep after each timestamp just to make sure that everything will run smoothly
                time.sleep(1)
                
        # Also sleep after each session                
        time.sleep(5)
    
    if len(intersections_timestamps) > 0: 
        print(f'{len(intersections)} intersections with cloud data sources have been found in {len(intersections_timestamps)} timestamps.')
        print(f'Number of Sentinel2L1C items found: {Sentinel2L1C_items_num}')
        print(f'Number of PSScene4Band items found: {PSScene4Band_items_num}')
        print(f'Number of Landsat8L1G items found: {Landsat8L1G_items_num}')
        print(f'Number of MYD35_L2 items found: {MYD35_L2_items_num}')
        print(f'Number of MOD35_L2 items found: {MOD35_L2_items_num}')
        print('\n')
        print(f'The closest intersection found was {min_ts_diff} seconds from the timestamp: ')
        print(closest_intersection)
        print('\n')
        if len(intersections_all_sources) > 0:
            print(f'There were {len(intersections_all_sources)} intersections with all cloud data sources at the same time found in these timestamps (timestamp, number of items in each source): ')
            print(intersections_all_sources)
    else:
        print('No intersections with cloud data sources have been found with given parameters, try expanding the search.')
        
    return intersections, final_search_res_list

def main():
    intersections, search_res = get_intersections(step=30, delta=300, number_of_sessions=44)
    with open('intersections_300.pkl', 'wb') as f:
        pickle.dump(intersections, f)
    with open('intersections_search_results.pkl', 'wb') as f:
        pickle.dump(search_res, f)

if __name__ == '__main__':
    main()