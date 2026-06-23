import numpy as np
import pandas as pd

# Approximate coordinates for the 78 districts in Dubai
DISTRICT_COORDINATES = {
    'Al Barari': (25.0994, 55.3178),
    'Al Barsha': (25.1068, 55.2003),
    'Al Furjan': (25.0233, 55.1469),
    'Al Jaddaf': (25.2158, 55.3267),
    'Al Quoz': (25.1378, 55.2447),
    'Al Qusais': (25.2747, 55.3944),
    'Al Satwa': (25.2206, 55.2736),
    'Al Sufouh': (25.1092, 55.1764),
    'Al Warsan': (25.1614, 55.4192),
    'Al Wasl': (25.2053, 55.2558),
    'Arjan': (25.0569, 55.2372),
    'Barsha Heights (Tecom)': (25.0994, 55.1800),
    'Bluewaters Island': (25.0792, 55.1228),
    'Bur Dubai': (25.2486, 55.3031),
    'Business Bay': (25.1833, 55.2727),
    'City of Arabia': (25.0906, 55.3361),
    'Culture Village (Jaddaf Waterfront)': (25.2242, 55.3347),
    'DAMAC Hills': (25.0142, 55.2472),
    'DAMAC Hills 2 (Akoya by DAMAC)': (24.9608, 55.3850),
    'DIFC': (25.2136, 55.2811),
    'Deira': (25.2678, 55.3122),
    'Discovery Gardens': (25.0272, 55.1428),
    'Downtown Dubai': (25.1972, 55.2744),
    'Dubai': (25.2048, 55.2708), # Centroid fallback
    'Dubai Creek Harbour': (25.1994, 55.3522),
    'Dubai Festival City': (25.2222, 55.3556),
    'Dubai Harbour': (25.0886, 55.1372),
    'Dubai Hills Estate': (25.0983, 55.2694),
    'Dubai Industrial City': (24.8767, 55.0858),
    'Dubai Investment Park (DIP)': (24.9819, 55.1619),
    'Dubai Marina': (25.0794, 55.1368),
    'Dubai Maritime City': (25.2683, 55.2750),
    'Dubai Media City': (25.0967, 55.1506),
    'Dubai Production City (IMPZ)': (25.0294, 55.1822),
    'Dubai Residence Complex': (25.0964, 55.3944),
    'Dubai Science Park': (25.0536, 55.2533),
    'Dubai Silicon Oasis (DSO)': (25.1278, 55.3803),
    'Dubai South': (24.8872, 55.1547),
    'Dubai Sports City': (25.0400, 55.2194),
    'Dubai Studio City': (25.0442, 55.2458),
    'Dubai Waterfront': (24.8808, 54.9450),
    'Dubailand': (25.0833, 55.3500),
    'Expo City': (24.9608, 55.1539),
    'Green Community': (24.9781, 55.1628),
    'International City': (25.1644, 55.4022),
    'Jebel Ali': (24.9856, 55.0844),
    'Jumeirah': (25.2122, 55.2494),
    'Jumeirah Beach Residence (JBR)': (25.0772, 55.1322),
    'Jumeirah Golf Estates': (25.0217, 55.1978),
    'Jumeirah Heights': (25.0683, 55.1558),
    'Jumeirah Lake Towers (JLT)': (25.0753, 55.1433),
    'Jumeirah Park': (25.0483, 55.1547),
    'Jumeirah Village Circle (JVC)': (25.0566, 55.2078),
    'Jumeirah Village Triangle (JVT)': (25.0469, 55.1903),
    'Living Legends': (25.0928, 55.3581),
    'Liwan': (25.1147, 55.3967),
    'Majan': (25.0967, 55.3678),
    'Meydan City': (25.1611, 55.3111),
    'Mirdif': (25.2189, 55.4217),
    'Mohammed Bin Rashid City': (25.1533, 55.3117),
    'Motor City': (25.0456, 55.2344),
    'Mudon': (25.0242, 55.2536),
    'Muhaisnah': (25.2678, 55.4089),
    'Palm Jumeirah': (25.1124, 55.1390),
    'Pearl Jumeirah': (25.2464, 55.2561),
    'Ras Al Khor': (25.1839, 55.3636),
    'Remraam': (24.9825, 55.2169),
    'Sobha Hartland': (25.1817, 55.3217),
    'The Greens': (25.0894, 55.1747),
    'The Hills': (25.0778, 55.1736),
    'The Views': (25.0833, 55.1764),
    'The World Islands': (25.2342, 55.1658),
    'Town Square': (24.9922, 55.2867),
    'Umm Suqeim': (25.1417, 55.2042),
    'Unknown': (25.2048, 55.2708),
    'Wasl Gate': (25.0239, 55.1275),
    'World Trade Centre': (25.2269, 55.2894),
    "Za'abeel": (25.2194, 55.3017)
}

# Key Landmark Coordinates
BURJ_KHALIFA = (25.1972, 55.2744)
DXB_AIRPORT = (25.2532, 55.3657)
JUMEIRAH_BEACH = (25.1610, 55.2206)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance in kilometers between two points
    on the earth (specified in decimal degrees).
    Supports scalar and numpy array operations.
    """
    R = 6371.0  # Earth radius in kilometers
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def add_geospatial_features(df):
    """
    Appends geospatial distances (to Burj Khalifa, DXB Airport, and Jumeirah Beach)
    to the input DataFrame.
    """
    df = df.copy()
    
    # Map district names to coordinates
    mapped_coords = df["district"].map(DISTRICT_COORDINATES)
    
    # If any district is unmapped, fallback to the Dubai centroid (tuple)
    default_coord = DISTRICT_COORDINATES['Unknown']
    coords = [c if isinstance(c, tuple) else default_coord for c in mapped_coords]
    
    lats = np.array([c[0] for c in coords])
    lons = np.array([c[1] for c in coords])
    
    # Calculate distances in kilometers
    df["dist_to_burj"] = haversine_distance(lats, lons, BURJ_KHALIFA[0], BURJ_KHALIFA[1])
    df["dist_to_airport"] = haversine_distance(lats, lons, DXB_AIRPORT[0], DXB_AIRPORT[1])
    df["dist_to_beach"] = haversine_distance(lats, lons, JUMEIRAH_BEACH[0], JUMEIRAH_BEACH[1])
    
    return df
