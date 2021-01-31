from math import acos,cos,sin,pi,atan2

def great_circle_distance(loc1, loc2):
    """
    Returns the approximate distance between (lat1, lon1) and (lat2, lon2) in
    miles, taking into account the Earth's curvature (but assuming a spherical
    earth).

    Latitude and longitudes given in degrees.  Thanks to Berthold Horn for this
    implementation.
    """
    lat1, lon1 = loc1
    lat2, lon2 = loc2
    phi1 = lat1*pi/180.
    theta1 = lon1*pi/180.
    phi2 = lat2*pi/180.
    theta2 = lon2*pi/180.
    cospsi = sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2)*cos(theta2-theta1)
    sinpsi = ((sin(theta1)*cos(phi1)*sin(phi2) - sin(theta2)*cos(phi2)*sin(phi1))**2 +\
              (cos(theta2)*cos(phi2)*sin(phi1) - cos(theta1)*cos(phi1)*sin(phi2))**2 +\
              (cos(phi1)*cos(phi2)*sin(theta2-theta1))**2)**0.5
    return atan2(sinpsi,cospsi) * 3958