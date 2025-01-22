import math

def match_points(list1, list2):
    """
    Returns a list of indices, where each index i represents the index of the closest
    point from list 2 to the point at index i from list 1. list1 and list2 are assumed to
    contain at least 1 element each. The length of the list returned is the same as the
    length of list1.
    """
    index_list = []
    for p1 in list1:
        min_distance = 41000
        min_index = -1
        for i in range(len(list2)):
            if calc_distance(p1, list2[i]) < min_distance:
                min_distance = calc_distance(p1, list2[i])
                min_index = i
        index_list.append(i)
    return index_list

def calc_distance(point1, point2):
    """
    Given two points (tuples), where the first element of each tuple is a longitude coordinate,
    and the second is a latitude coordinate, this function returns the distance between the two 
    using the haversine formula.
    """
    earthRadius = 6371
    distanceLat = math.radians(point2[1] - point1[1])
    distanceLon = math.radians(point2[0] - point1[0])
    lat1 = math.radians(point1[1])
    lat2 = math.radians(point2[1])
    a = math.sin(distanceLat/2) * math.sin(distanceLat/2) + math.sin(distanceLon/2) * math.sin(distanceLon/2) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earthRadius * c
