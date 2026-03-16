import numpy as np
import healpy as hp

from astropy import units as u
from astropy.coordinates import (
    SkyCoord, AltAz, EarthLocation, FK5, Galactic,
    solar_system_ephemeris, get_body,
    get_body_barycentric_posvel
)
from astropy.time import Time
from astropy.utils import iers
from astropy.utils.iers import conf
conf.auto_max_age = None
# Use robust JPL ephemerides for planets
solar_system_ephemeris.set('de432s')

# ---- Calibrators & site ----
CalibratorList = {
    'TauA': [(5 + 34./60. + 31.94/3600.)*15, 22 + 0/60. + 52.2/3600.],
    'CasA': [(23 + 23./60. + 26.93/3600.)*15, 58 + 49/60. + 7.68/3600.],
    'CygA': [(19 + 59/60. + 28.356/3600.)*15, 40 + 44/60. + 2.097/3600.],
    'jupiter': None,
    'sun': None,
    'saturn': None,
    'moon': None,
}

comap_longitude = -(118 + 16./60. + 56./3600.)  # deg (west negative)
comap_latitude  =   37 + 14./60. + 2./3600.     # deg

# ---- helpers ----

def _as_time(mjd):
    # Accept scalar or array
    return Time(np.atleast_1d(mjd), format='mjd', scale='utc')

def _site(lon_deg, lat_deg):
    return EarthLocation(lon=lon_deg*u.deg, lat=lat_deg*u.deg, height=0*u.m)

def _ensure_array(x):
    return np.atleast_1d(np.array(x, dtype=float))

# ---- sexagesimal ----

def sex2deg(dms, hours=False):
    d, m, s = dms.split(':')
    sign = -1 if d.strip().startswith('-') else 1
    val = abs(float(d)) + float(m)/60. + float(s)/3600.
    val *= sign
    return val*15.0 if hours else val

def deg2sex(x, hours=False):
    x = float(x)
    if hours:
        x /= 15.0
    sign = -1 if x < 0 else 1
    x = abs(x)
    d = int(np.floor(x))
    m = int(np.floor((x - d)*60.0))
    s = (x - d - m/60.0)*3600.0
    d *= sign
    return f'{d:02d}:{m:02d}:{s:.2f}'

# ---- 3D rotations (keep your original API) ----

def RotatePhi(skyVec, objRa):
    out = np.zeros_like(skyVec)
    ang = np.deg2rad(objRa)
    c, s = np.cos(ang), np.sin(ang)
    out[:,0] =  skyVec[:,0]*c + skyVec[:,1]*s
    out[:,1] = -skyVec[:,0]*s + skyVec[:,1]*c
    out[:,2] =  skyVec[:,2]
    return out

def RotateTheta(skyVec, objDec):
    out = np.zeros_like(skyVec)
    ang = np.deg2rad(objDec)
    c, s = np.cos(ang), np.sin(ang)
    out[:,0] =  skyVec[:,0]*c + skyVec[:,2]*s
    out[:,1] =  skyVec[:,1]
    out[:,2] = -skyVec[:,0]*s + skyVec[:,2]*c
    return out

def RotateR(skyVec, objPang):
    out = np.zeros_like(skyVec)
    ang = np.deg2rad(objPang)
    c, s = np.cos(ang), np.sin(ang)
    out[:,0] =  skyVec[:,0]
    out[:,1] =  skyVec[:,1]*c + skyVec[:,2]*s
    out[:,2] = -skyVec[:,1]*s + skyVec[:,2]*c
    return out

def Rotate(ra, dec, r0, d0, p0):
    skyVec = hp.ang2vec(np.deg2rad(90.-np.array(dec)), np.deg2rad(np.array(ra)))
    outVec = RotatePhi(skyVec, r0)
    outVec = RotateTheta(outVec, d0)
    outVec = RotateR(outVec, p0)
    _theta, _phi = hp.vec2ang(outVec)
    _dec = np.rad2deg(0.5*np.pi - _theta)
    _ra  = np.rad2deg(_phi)
    _ra[_ra > 180] -= 360
    return _ra, _dec

def UnRotate(ra, dec, r0, d0, p0):
    skyVec = hp.ang2vec(np.deg2rad(90.-np.array(dec)), np.deg2rad(np.array(ra)))
    outVec = RotateR(skyVec, p0)
    outVec = RotateTheta(outVec, d0)
    outVec = RotatePhi(outVec, r0)
    _theta, _phi = hp.vec2ang(outVec)
    _dec = np.rad2deg(0.5*np.pi - _theta)
    _ra  = np.rad2deg(_phi)
    _ra[_ra > 360] -= 360
    _ra[_ra < 0]   += 360
    return _ra, _dec

def AngularSeperation(phi0,theta0,phi1,theta1, degrees=True):
    # (phi,theta) ~ (lon,lat)
    c = np.deg2rad(1.0) if degrees else 1.0
    mid = (np.sin(theta0*c)*np.sin(theta1*c) +
           np.cos(theta0*c)*np.cos(theta1*c)*np.cos((phi1-phi0)*c))
    return np.arccos(np.clip(mid, -1, 1)) / c

# ---- Equatorial <-> Horizon ----

def h2e(az, el, mjd, lon, lat, degrees=True):
    """Horizon (az,el) -> Equatorial (ra,dec), approximate (UTC, IERS default)."""
    az  = _ensure_array(az)
    el  = _ensure_array(el)
    t   = _as_time(mjd)
    loc = _site(lon, lat)

    frame = AltAz(az=az*(u.deg if degrees else u.rad),
                  alt=el*(u.deg if degrees else u.rad),
                  obstime=t, location=loc)
    sc = SkyCoord(frame)
    icrs = sc.icrs
    ra  = icrs.ra.to(u.deg if degrees else u.rad).value
    dec = icrs.dec.to(u.deg if degrees else u.rad).value

    # For compatibility with your old e2h(..., return_lha=True) pattern, h2e never returns LHA.
    return ra, dec

def h2e_full(az, el, mjd, lon, lat, degrees=True, sample_rate=50):
    """Full apparent transform using Astropy (already accounts for nut/prec/aberr.)."""

    ra,dec= h2e(az[::sample_rate], el[::sample_rate], mjd[::sample_rate], lon, lat, degrees=degrees)
    ra = np.interp(mjd,mjd[::sample_rate],ra)
    dec = np.interp(mjd,mjd[::sample_rate],dec) 
    return ra, dec 

def e2h(ra, dec, mjd, lon, lat, degrees=True, return_lha=False):
    """Equatorial -> Horizon; optionally return Local Hour Angle (deg or rad)."""
    ra  = _ensure_array(ra)
    dec = _ensure_array(dec)
    t   = _as_time(mjd)
    loc = _site(lon, lat)

    sc = SkyCoord(ra=ra*(u.deg if degrees else u.rad),
                  dec=dec*(u.deg if degrees else u.rad),
                  frame='icrs')
    altaz = sc.transform_to(AltAz(obstime=t, location=loc))
    az = altaz.az.to(u.deg if degrees else u.rad).value
    el = altaz.alt.to(u.deg if degrees else u.rad).value

    if return_lha:
        lst = t.sidereal_time('apparent', longitude=loc.lon)  # Angle
        lst_val = lst.to(u.deg if degrees else u.rad).value
        lha = (lst_val - ra)  # same units as ra
        # wrap to (-180,180] or (-pi,pi]
        if degrees:
            lha = (lha + 180.0) % 360.0 - 180.0
        else:
            lha = (lha + np.pi) % (2*np.pi) - np.pi
        return az, el, lha
    else:
        return az, el

def e2h_full(ra, dec, mjd, lon, lat, degrees=True, return_lha=False):
    """Full apparent; same as e2h in Astropy (AltAz includes all terms if IERS present)."""
    return e2h(ra, dec, mjd, lon, lat, degrees=degrees, return_lha=return_lha)

# ---- Precession/Nutation wrappers ----

def precess(ra, dec, mjd, degrees=True):
    """
    Precess coordinates from equinox=epoch(mjd) to FK5 J2000 (ICRS-like).
    """
    t = _as_time(mjd)
    ra0  = _ensure_array(ra)
    dec0 = _ensure_array(dec)
    eq_src = FK5(equinox=t)
    eq_dst = FK5(equinox=Time('J2000'))

    sc = SkyCoord(ra=ra0*(u.deg if degrees else u.rad),
                  dec=dec0*(u.deg if degrees else u.rad),
                  frame=eq_src)
    sc2 = sc.transform_to(eq_dst)
    return sc2.ra.to(u.deg if degrees else u.rad).value, sc2.dec.to(u.deg if degrees else u.rad).value

def prenut(ra, dec, mjd, degrees=True):
    """
    Historically SLA's 'prenut' applied precession+nutation of date->J2000.
    Astropy FK5 transform includes those effects via ERFA given equinox.
    """
    return precess(ra, dec, mjd, degrees=degrees)

def precess2year(ra, dec, mjd, degrees=True):
    """
    Your original called precess_year(ra,dec,mjd).
    Here we interpret as: coordinates at epoch(mjd) -> FK5(J2000).
    Same as precess().
    """
    return precess(ra, dec, mjd, degrees=degrees)

# ---- Parallactic angle ----
def pa(ra, dec, mjd, lon, lat, degrees=True):
    """
    Calculate parallactic angle directly (no Astropy dependency).

    args:
    ra  - arraylike, right ascension
    dec - arraylike, declination
    mjd - arraylike, modified julian date
    lon - float, observer longitude (deg, east positive)
    lat - float, observer latitude (deg)
    """
    ra  = np.atleast_1d(ra).astype(float)
    dec = np.atleast_1d(dec).astype(float)
    mjd = np.atleast_1d(mjd).astype(float)

    # degrees or radians
    if degrees:
        c = np.pi / 180.0
    else:
        c = 1.0

    # convert to radians
    ra_rad  = ra * c
    dec_rad = dec * c
    lon_rad = lon * c
    lat_rad = lat * c

    # Compute Local Sidereal Time at observer longitude
    LST = Time(mjd, format='mjd',location=(lon*u.deg,lat*u.deg)).sidereal_time('apparent').value/24.*360*np.pi/180.

    # Hour angle
    H = LST - ra_rad
    H = np.mod(H + np.pi, 2*np.pi) - np.pi  # wrap to [-pi, pi]

    # Parallactic angle formula
    sin_p = np.sin(H) * np.cos(lat_rad)
    cos_p = np.sin(lat_rad)*np.cos(dec_rad) - np.cos(lat_rad)*np.sin(dec_rad)*np.cos(H)
    p = np.arctan2(sin_p, cos_p)

    if degrees:
        return np.degrees(p)
    else:
        return p

# ---- Galactic transforms ----

def e2g(ra, dec, degrees=True):
    sc = SkyCoord(ra=_ensure_array(ra)*(u.deg if degrees else u.rad),
                  dec=_ensure_array(dec)*(u.deg if degrees else u.rad),
                  frame='icrs')
    g = sc.transform_to(Galactic())
    gl = g.l.to(u.deg if degrees else u.rad).value
    gb = g.b.to(u.deg if degrees else u.rad).value
    return gl, gb

def g2e(gl, gb, degrees=True):
    sc = SkyCoord(l=_ensure_array(gl)*(u.deg if degrees else u.rad),
                  b=_ensure_array(gb)*(u.deg if degrees else u.rad),
                  frame=Galactic())
    e = sc.icrs
    ra  = e.ra.to(u.deg if degrees else u.rad).value
    dec = e.dec.to(u.deg if degrees else u.rad).value
    return ra, dec

# ---- Planets ----

# Mean radii (km) for angular sizes (rough; tweak as needed)
_PLANET_RADIUS_KM = {
    1: 2439.7,   # Mercury
    2: 6051.8,   # Venus
    3: 1737.4,   # Moon
    4: 3389.5,   # Mars
    5: 69911.0,  # Jupiter (equatorial)
    6: 58232.0,  # Saturn (equatorial, no rings here)
    7: 25362.0,  # Uranus (equatorial)
    8: 24622.0,  # Neptune (equatorial)
    9: 1188.3,   # Pluto
    0: 695700.0, # Sun
}

def rdplan(mjd, planet, lon, lat, degrees=True):
    """
    Approx topocentric apparent (ra, dec) and angular size.
    SLA-style: planet index: 1..9 as in your doc; else Sun (0).
    Returns (ra, dec, ang_diameter) in radians if degrees=False, else degrees.
    """
    t = _as_time(mjd)
    loc = _site(lon, lat)

    # Map index to names astropy understands
    idx_to_name = {
        0: 'sun', 1: 'mercury', 2: 'venus', 3: 'moon', 4: 'mars',
        5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune', 9: 'pluto'
    }
    name = idx_to_name.get(planet, 'sun')

    # Topocentric apparent RA/Dec via AltAz back to ICRS
    body = get_body(name, t, loc)  # GCRS at observer -> transforms fine
    icrs = body.icrs
    ra  = icrs.ra
    dec = icrs.dec

    # Distance from observer for angular size
    # Transform to AltAz to get distance (body.distance is to observer in many cases)
    # For get_body, 'distance' is often observer distance already:
    dist = getattr(body, 'distance', None)
    if dist is None:
        # Fallback: compute from barycentric positions (rough)
        pos, _vel = get_body_barycentric_posvel(name, t)
        # distance from Earth barycenter — not topocentric; acceptable approx
        epos, _evel = get_body_barycentric_posvel('earth', t)
        dist = (pos - epos).norm().to(u.km)

    R = _PLANET_RADIUS_KM.get(planet, _PLANET_RADIUS_KM[0]) * u.km
    ang_rad = 2.0 * np.arctan2(R, dist)  # small-angle ok, but this is general

    if degrees:
        return ra.to(u.deg).value, dec.to(u.deg).value, ang_rad.to(u.deg).value
    else:
        return ra.to(u.rad).value, dec.to(u.rad).value, ang_rad.value

def planet(mjd, planet):
    """
    Approx heliocentric position and velocity of a planet.
    Return array shape (6, N): [x,y,z, vx,vy,vz] in AU, AU/day.
    """
    t = _as_time(mjd)
    idx_to_name = {
        1: 'mercury', 2: 'venus', 3: 'moon', 4: 'mars',
        5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune', 9: 'pluto'
    }
    if planet not in idx_to_name:
        raise ValueError("planet index must be 1..9 for planet()")
    name = idx_to_name[planet]
    pos, vel = get_body_barycentric_posvel(name, t)  # GCRS barycentric
    # to numpy arrays
    p = np.vstack([pos.x.to(u.AU).value,
                   pos.y.to(u.AU).value,
                   pos.z.to(u.AU).value])
    v = np.vstack([vel.x.to(u.AU/u.day).value,
                   vel.y.to(u.AU/u.day).value,
                   vel.z.to(u.AU/u.day).value])
    return np.vstack([p, v])

def getPlanetPosition(source, lon, lat, mjdtod, allpos=False):
    """
    Get ra, dec and Earth-source distance (mean), SLA-like wrapper.
    """
    s = source.lower()
    if 'jupiter' in s:
        pid = 5
    elif 'saturn' in s:
        pid = 6
    elif 'moon' in s:
        pid = 3
    else:
        pid = 0  # sun

    ra, dec, ang = rdplan(mjdtod, pid, lon, lat, degrees=False)  # radians
    # Distance via barycentric vectors
    pos, _ = get_body_barycentric_posvel('jupiter' if pid==5 else
                                         'saturn' if pid==6 else
                                         'moon'    if pid==3 else
                                         'sun',
                                         _as_time(mjdtod))
    epos, _ = get_body_barycentric_posvel('earth', _as_time(mjdtod))
    rdist = (pos - epos).norm().to(u.AU).value  # Earth-object distance in AU

    if allpos:
        return np.array(ra), np.array(dec), np.array(rdist)
    else:
        return float(np.mean(ra)*180/np.pi), float(np.mean(dec)*180/np.pi), float(np.mean(rdist))

def sourcePosition(src, mjd, lon, lat):
    """
    Return (az, el, r0, d0) where r0,d0 are RA/Dec (deg) of the source over time.
    For fixed calibrators, do full apparent transform; for moving bodies, sample/coarsen.
    """
    t = _as_time(mjd)
    loc = _site(lon, lat)

    if CalibratorList.get(src) is None:
        # Moving body
        step_sec = 300.0  # 5 minutes
        cadence = np.abs((t[1]-t[0]).to_value('sec')) if t.size > 1 else step_sec
        stride = max(1, int(step_sec / max(cadence, 1.0)))
        ra_sub, dec_sub, _ = rdplan(mjd[::stride], 5 if 'jupiter' in src.lower()
                                               else 6 if 'saturn' in src.lower()
                                               else 3 if 'moon' in src.lower()
                                               else 0, lon, lat, degrees=True)
        ra  = np.interp(mjd, mjd[::stride], ra_sub)
        dec = np.interp(mjd, mjd[::stride], dec_sub)
    else:
        ra0, dec0 = CalibratorList[src]
        ra  = np.full_like(mjd, ra0, dtype=float)
        dec = np.full_like(mjd, dec0, dtype=float)

    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    altaz = sc.transform_to(AltAz(obstime=t, location=loc))
    az = altaz.az.to(u.deg).value
    el = altaz.alt.to(u.deg).value
    return az, el, ra, dec
