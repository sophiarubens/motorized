from itertools import combinations

### Hydra is not installable yet :( so point to it here ###
import sys
sys.path.insert(0, "/Users/sophiarubens/Hydra")

import os
import pickle

import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import numpy as np
from hydra import sparse_beam
from matplotlib.colors import LogNorm, SymLogNorm
from astropy import units as u


from skyfield.api import load, wgs84, Star
from skyfield.api import N, W
from skyfield.positionlib import Apparent
from skyfield.units import Angle,Distance

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
    }
)

from pyuvdata import UVBeam

"""# Set some global variables

Ignore warning that pops if ZOOM_RHO is True and using large nmax
"""

### system-specific paths ###
# beamfile = "/path/to/beam.fits" # simulated beamfile
# outdir = "/path/to/outdir" # where to save outputs
simdir= "/Users/sophiarubens/Downloads/research/code/transit_simulations/"
beamfile= simdir+"beam.fits"
outdir= simdir
### system-specific paths ###
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Whether just looking at just the first ten degrees
ZOOM_RHO = True
# Bring the basis functions in if zooming
rho_const = np.sqrt(1-np.cos(np.pi * 3 / 45)) if ZOOM_RHO else np.sqrt(1-np.cos(np.pi * 23 / 45))
za_range = (0, 10) if ZOOM_RHO else (0, 90)
# Maximum number of radial modes on the sparse_beam object
nmax = 80
# Maximum azimuthal mode number to use on the sparse_beam object
mmax = 179
# Test frequency channel to work with on simulated beam
test_freq = 120
dpiuse=500

unpert_sb = sparse_beam.sparse_beam(beamfile, nmax=nmax,
                                    mmodes=np.arange(-mmax, mmax + 1),
                                    za_range=za_range,
                                    Nfeeds=None,
                                    alpha=rho_const,
                                    convert_to_power=True,
                                    sqrt=False,
                                    load=False)
# Make an azimuth-zenith angle grid for plotting purposes
Az, Za = np.meshgrid(unpert_sb.axis1_array, unpert_sb.axis2_array)

# CHORD parameters
lat = (49 + 19/60 + 15/3600)
lon = (119 + 37/60 + 15/3600)

# source parameters
planets = load('de421.bsp')
earth = planets['earth']
CasA_RA_hrs=(23, 23, 26.0)
CasA_dec_deg=(58, 48, 41)
CasA_dec_deg_decimal= 58 + 48/60 + 41/3600
CasA = Star(ra_hours=CasA_RA_hrs, dec_degrees=CasA_dec_deg)
CygA = Star(ra_hours=(19, 59, 28.3566), dec_degrees=(40, 44, 2.096))
# TauA = Star(ra_hours=(), dec_degrees=()) https://research.csiro.au/racs/home/gallery/a-sources/
# PicA = Star(ra_hours=(), dec_degrees=())
# VirA = Star(ra_hours=(), dec_degrees=())
Zen = Star(ra_hours=(19, 59, 28.3566), dec_degrees=(lat))
NCP = Star(ra_hours=(0), dec_degrees=(90))

# Easier to read in a separate UVBeam object from the same file so that I don't have to deal with overridden interp method
UVB = UVBeam.from_file(beamfile, za_range=za_range)
UVB.peak_normalize()
UVB.efield_to_power(calc_cross_pols=False)

ts = load.timescale()

"""# Make a little library"""

def truncated_cmap(cmap_name, vmin=0.1, vmax=0.9, n=256):
    """
    Truncates a colormap.

    Parameters:
        cmap_name (str):
            Name of the colormap.
        vmin (float):
            Minimum of colormap.
        vmax (float):
            Maximum of colormap.
        n (int):
            Number of discrete color states in the map.
    Returns:
        The colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    new_colors = cmap(np.linspace(vmin, vmax, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc_{vmin}_{vmax}",
        new_colors
    )

def altaz_to_xy(alt_deg: np.ndarray, az_deg: np.ndarray, rot_deg=0, rref=90.0):
    """
    Map (alt, az) to unit-disk (x, y) with rref mapped to r=1 using
    an azimuthal equidistant style mapping:
        r = (90 - alt) / 90
        x = r * sin(az)
        y = r * cos(az)
    az convention: 0=N, 90=E.

    NOTE: This convention is different UVBeam's as well as matplotlib's sense of polar plotting.
    This is just done for compatibility with certain plotting code. Convention is transformed
    when feeding az/za to UVBeam/sparse_beam interp methods, as well as matplotlib polar plots.

    Parameters:
        alt_deg (float):
            Altitude in degrees.
        az_deg (float):
            Azimuth in degrees.
        rot_deg (float):
            Rotation to apply to azimuth, in degrees. This number gets added to az_deg.
        rref (float):
            Reference zenith angle for r=1.
    Returns:
        x, y (float):
            Cartesian coordinates of inputs.
    """
    r = (90.0 - alt_deg) / rref
    az = np.deg2rad(az_deg+rot_deg)
    x = r * np.sin(az)
    y = r * np.cos(az)
    return x, y

def xy_to_ij(x: np.ndarray, y: np.ndarray, n: int = 181):
    """
    Map x,y in [-1,1] to integer pixel indices i,j in [0,n-1]
    for an n×n image whose extent is [-1,1]×[-1,1].

    Parameters:
        x, y (float):
            Cartesian coordinates of points on a disc representing the FoV (see altaz_to_xy)
        n (int):
            Size of the image on on side.
    Returns:
        i, j: Indices of (x,y) into image pixels.
    """
    scale = (n - 1) / 2.0
    j = np.rint((x + 1.0) * scale).astype(int)  # x -> column
    i = np.rint((y + 1.0) * scale).astype(int)  # y -> row
    j = np.clip(j, 0, n - 1)
    i = np.clip(i, 0, n - 1)
    return i, j

def get_points_within_view(az_deg, alt_deg, fov=10.):
    """
    Determine which points from a track are in the field of view based on their az/alt.

    NOTE: Argument ordering is opposite of alt_az_to_xy (sorry!).

    Parameters:
        az_deg (float):
            Azimuth in degrees.
        alt_deg (float):
            Altitude in degrees.
        fov (float):
            Field of view, in degrees.
    Returns:
        az_within_view, za_within_view (float):
            Subarrays of inputs that are in view.
            Will return None's if none were in view, rather than empty arrays.
    """

    # Find part of track within view
    za_deg = 90. - alt_deg
    # plt.figure()
    # plt.plot(za_deg)
    # plt.savefig("za_deg_{:6.3f}.png".format(za_deg[0]))
    # plt.close()
    track_within_view = za_deg <= fov


    if np.any(track_within_view):
        az_within_view = np.deg2rad(az_deg[track_within_view])
        za_within_view = np.deg2rad(za_deg[track_within_view])
    else:
        az_within_view = None
        za_within_view = None
    return az_within_view, za_within_view

def plot_fullsky_tracks(sky, x_ncp, y_ncp, name="sky_tracks"):
    """
    Plot tracks from draw_tracks horizon-to-horizon

    Parameters:
        sky (array):
            Boolean array representing az/el grid that is True where tracks passed and False otherwise.
        x_ncp, y_ncp (float):
            x,y coordinates on sky of North Celestial Pole (see altaz_to_xy).
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(
        sky,
        origin="lower",
        extent=[-1, 1, -1, 1],
        interpolation="none",
        cmap=truncated_cmap(cmr.pride, vmin=0.3, vmax=0.8, n=256)
    )

    # Horizon circle (r=1)
    th = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(th), np.sin(th), ls=":", lw=2.1, color="whitesmoke")

    ax.scatter(x_ncp, y_ncp, s=32, marker=r"$\ast$", color="whitesmoke")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Cardinal labels (az=0 N, 90 E)
    ax.text(0, 0.92, "N", ha="center", va="bottom", color="whitesmoke")
    ax.text(0.92, 0, "E", ha="left", va="center", color="whitesmoke")
    ax.text(0, -0.92, "S", ha="center", va="top", color="whitesmoke")
    ax.text(-0.92, 0, "W", ha="right", va="center", color="whitesmoke")

    #ax.set_title("Cyg A visited pixels on 181×181 pixel-first sky disk")

    plt.savefig(name,dpi=dpiuse)

    return

def plot_omniscient_sky(az_scatter, za_scatter, src,
                        freqs=None):
    fig=plt.figure(figsize=(10,6),layout="constrained")
    for i,az_track in enumerate(az_scatter):
        za_track=za_scatter[i]
        az_track_deg=np.rad2deg(az_track)
        dec_track_deg=np.rad2deg(np.pi/2-za_track)
        plt.plot(az_track_deg,dec_track_deg, 
                 c="mediumturquoise") # these are probably in radians and probably need to be preprocessed and I probably can't iterate over them like this. pseudocode or whatever.
    src_dec=src.dec.degrees
    # print("az_track[0]=",az_track[0])
    # print("az_track=",az_track)
    N_timesteps=len(az_track) # use the most recent track just as a matter of convenience
    generic_RAs=np.linspace(0,2*np.pi,N_timesteps)
    src_dec_vec=src_dec*np.ones(N_timesteps)
    # print("len(generic_RAs),len(src_dec_vec)=",len(generic_RAs),len(src_dec_vec))
    # print("az_track.shape=",az_track.shape) # saving this for after the last known failure point because I think this is too list-y to have a shape
    plt.plot(generic_RAs,src_dec_vec,
             c="mediumpurple",ls="dashdot")
    # plt.xlim(0,180)
    # plt.ylim(0,90)
    plt.xlabel("RA (deg)")
    plt.ylabel("dec (deg)")
    return fig

def get_alt_az_above_horizon(alt_deg, az_deg):
    """
    Filter az/alt arrays to only include points at or above the horizon.

    Parameters:
        alt_deg (array):
            Altitude in degrees.
        az_deg (array):
            Azimuth in degrees.
    Returns:
        alt_deg, az_deg (array):
            Subarrays of inputs containing only points with altitude >= 0.
    """

    above_horizon = alt_deg >= 0.0
    alt_deg = alt_deg[above_horizon]
    az_deg = az_deg[above_horizon]

    return alt_deg, az_deg

def get_alt_az_deg(apparent):
    """
    Extract altitude and azimuth in degrees from a Skyfield apparent position,
    discarding points below the horizon.

    Parameters:
        apparent (skyfield.positionlib.Apparent):
            Apparent position of a source as seen from an observer.
    Returns:
        alt_deg (array):
            Altitude in degrees, above-horizon points only.
        az_deg (array):
            Azimuth in degrees, above-horizon points only.
    """

    alt, az, _ = apparent.altaz()
    alt_deg = alt.degrees
    az_deg  = az.degrees

    alt_deg, az_deg = get_alt_az_above_horizon(alt_deg, az_deg)

    return alt_deg, az_deg

def get_DRAO_apparent(el, t, mode="rot", 
                      nod_params=None # nod parameters amp,freq
                      ):
    """
    Construct a DRAO observer position and compute the apparent position of a
    source at a sequence of times.

    Parameters:
        el (float):
            Latitude offset in degrees added to the DRAO latitude. In 'rot' mode
            this is unused (keep 0); in 'el' mode it shifts the effective elevation
            of the telescope to place Cas A near boresight.
        t (skyfield.timelib.Time):
            Array of times at which to evaluate the apparent position.
        mode (str):
            'rot' to simulate rotations, 'el' to simulate elevation shifts, 'nod' to simulate the motorized experiment.
            Default 'rot'.
    Returns:
        DRAO (skyfield.toposlib.GeographicPosition):
            The observer position on the Earth's surface.
        apparent (skyfield.positionlib.Apparent):
            Apparent position of the source as seen from DRAO at each time in t.
    """
    DRAO = wgs84.latlon((lat+el)*N, lon*W, 545.671) # offset lat by el to simulate el change
    if mode=="rot":
        src=Zen
    elif mode=="el" or mode=="nod":
        src=CasA
    else:
        raise ValueError("unknown obs mode. try rot, el, or nod")

    apparent = (earth + DRAO).at(t).observe(src).apparent()
    
    if mode=="nod":
        nod_amp, nod_freq = nod_params
        astropyified_time0 = t.tt * u.d
        sinusoid_arg_rad = (nod_freq * astropyified_time0).decompose() * u.rad
        sinusoid_profile=nod_amp * np.sin(sinusoid_arg_rad) # actually sinusoidal with the expected amplitude -> no unit conversion silent failure
        nod = Angle(degrees=sinusoid_profile) # still time-indexed (didn't get si)
        ra, dec, dist = apparent.radec()
        dec = Angle(degrees=dec.degrees + nod.degrees)
        ra_rad = ra.radians *u.rad
        dec_rad = dec.radians *u.rad
        dist_au = dist.au
        x = dist_au * np.cos(dec_rad) * np.cos(ra_rad)
        y = dist_au * np.cos(dec_rad) * np.sin(ra_rad)
        z = dist_au * np.sin(dec_rad)
        xyz_au = np.array([x.value, y.value, z.value])
        newpos=Distance(au=xyz_au)
        apparent.position = apparent.xyz = newpos # have to update the position *and* xyz attributed to correctly override the cached property
    
    return DRAO, apparent

def get_xy_ncp(DRAO, t):
    """
    Compute the cartesian coordinates of the North Celestial Pole as seen from DRAO.
    Used to mark the NCP on full-sky track plots.

    Parameters:
        DRAO (skyfield.toposlib.GeographicPosition):
            The observer position on the Earth's surface.
        t (skyfield.timelib.Time):
            Array of times; the NCP position is evaluated at t[0].
    Returns:
        x_ncp, y_ncp (array):
            Disk coordinates of the NCP (see altaz_to_xy).
    """
    app_ncp = (earth + DRAO).at(t[0]).observe(NCP).apparent() # FIXME: hardcode t[0]???
    alt_ncp_deg, az_ncp_deg = get_alt_az_deg(app_ncp)

    x_ncp, y_ncp = altaz_to_xy(
        np.array([alt_ncp_deg]),
        np.array([az_ncp_deg]),
    )

    return x_ncp, y_ncp

def get_coord_wrapper(t, el=0, mode="rot", nod_params=None, plot=True):
    """
    Convenience wrapper that computes the apparent alt/az of a source and,
    optionally, the disk coordinates of the NCP.

    Parameters:
        t (skyfield.timelib.Time):
            Array of times at which to evaluate positions.
        el (float):
            Latitude offset in degrees passed to get_DRAO_apparent to simulate an elevation shift. Default 0.
        mode (str):
            'rot' or 'el', passed to get_DRAO_apparent. Default 'rot'.
        plot (bool):
            If True, also compute and return cartesian coordinates of NCP for plotting.
            If False, x_ncp and y_ncp are returned as None. Default True.
    Returns:
        alt_deg (array):
            Source altitude in degrees, above-horizon points only.
        az_deg (array):
            Source azimuth in degrees, above-horizon points only.
        x_ncp, y_ncp (array or None):
            Disk coordinates of the NCP, or None if plot is False.
    """
    DRAO, apparent = get_DRAO_apparent(el, t, mode=mode, nod_params=nod_params)
    alt_deg, az_deg = get_alt_az_deg(apparent)
    if plot:
        x_ncp, y_ncp = get_xy_ncp(DRAO, t)
    else:
        x_ncp = None
        y_ncp = None
    return alt_deg, az_deg, x_ncp, y_ncp

def view_cutoff_and_xy(alt_deg, az_deg, rot_deg=0, plot=True):
    """
    Apply a rotation to a track, identify points within the field of view,
    and optionally compute full-sky cartesian coordinates for plotting.

    Parameters:
        alt_deg (array):
            Altitude in degrees.
        az_deg (array):
            Azimuth in degrees.
        rot_deg (float):
            Rotation to apply to the azimuth in degrees, passed to both
            get_points_within_view and altaz_to_xy. Default 0.
        plot (bool):
            If True, compute and return disk coordinates x, y for the full track.
            If False, x and y are returned as None. Default True.
    Returns:
        az_within_view (array or None):
            Azimuths of points within the FoV in radians, or None if none are in view.
        za_within_view (array or None):
            Zenith angles of points within the FoV in radians, or None if none are in view.
        x, y (array or None):
            Disk coordinates of the full track (see altaz_to_xy), or None if plot is False.
    """
    az_within_view, za_within_view = get_points_within_view(az_deg + rot_deg, alt_deg)
    if plot:
        x, y = altaz_to_xy(alt_deg, az_deg, rot_deg=rot_deg) # apply rotation in this function
    else:
        x = None
        y = None

    return az_within_view, za_within_view, x, y

def draw_tracks(tres=0.5, rots=np.arange(0, -90, -90), n=181, src=CasA,
                plot=True, mode="rot", nod_params=None, name="sky_tracks",
                el_delta=0.5):
    """
    Draw some tracks.

    Args:
        tres (float):
            Time resolution in _minutes_.
        rots (float_array):
            Rotations to perform
        n (int):
            Size of grid to draw on
        plot (bool):
            Whether to draw horizon-to-horizon plot of the tracks
        mode (str):
            'rot' (for rotations) or 'el' for elevations
        el_delta (float):
            How finely to space the elevation shifts, in degrees.
    Returns:
        az_scatter (array):
            Azimuths traced within FoV.
        za_scatter (array):
            Zenith angles (interpreted as boresight angles) traced within FoV.
    """
    assert mode in ['rot', 'el', 'nod'], f"{mode} is an invalid mode. Must be 'rot', 'el', or 'nod'."

    t = ts.utc(2026, 7, 10, 0, np.arange(0, 1440, tres)) # discretized into !minutes!

    if mode == "rot":
        alt_deg, az_deg, x_ncp, y_ncp = get_coord_wrapper(t, plot=plot)
        iterator = rots
        bad_inds = None
    else:
        # If the telescope is pointed at zenith_el CasA passes through the boresight
        zenith_el = src.dec.degrees - lat
        # Try to avoid some edge effects
        els = np.arange(zenith_el-10 + el_delta, zenith_el + 10, el_delta)
        iterator = els

    sky = np.zeros((n, n), dtype=bool)

    az_scatter = []
    za_scatter = []

    for iter_ind, iter_val in enumerate(iterator):
        # Map track to x,y and then to pixels
        if mode == "rot":
            az_within_view, za_within_view, x, y = view_cutoff_and_xy(alt_deg, az_deg, rot_deg=iter_val,
                                                                      plot=plot)
        else:
            alt_deg, az_deg, x_ncp, y_ncp = get_coord_wrapper(t, el=iter_val, mode=mode, nod_params=nod_params, plot=plot)
 
            az_within_view, za_within_view, x, y = view_cutoff_and_xy(alt_deg, az_deg, rot_deg=0,
                                                                      plot=plot)        

        if plot:
            i, j = xy_to_ij(x, y, n=n)
            sky[i, j] = True

        if az_within_view is not None:
            if mode == "rot": # FIXME: make these both compatible with append
                az_scatter.extend((np.pi/2 - az_within_view) % (2 * np.pi)) # Change to E to N convention
                za_scatter.extend(za_within_view)
            else:
                az_scatter.append((np.pi/2 - az_within_view) % (2 * np.pi)) # Change to E to N convention
                za_scatter.append(za_within_view)
        else:
            print(f"Failed track at iterator index: {iter_ind}")


    if mode == "rot":
        az_scatter = np.array(az_scatter)
        za_scatter = np.array(za_scatter)

    if plot:
        plot_fullsky_tracks(sky, x_ncp, y_ncp, name=name)

    return az_scatter, za_scatter

def plot_tracks_within_fov(az_scatter, za_scatter):
    """
    Plot the az/za track points that fall within the field of view on a polar axis.
    Both the original polarization orientation and a 90-degree-rotated version
    (representing the perpendicular polarization) are shown.

    Parameters:
        az_scatter (array):
            Azimuths of points within the FoV, in radians. East-of-North convention.
        za_scatter (array):
            Zenith angles of points within the FoV, in radians.
    Returns:
        fig (matplotlib.figure.Figure):
            The figure containing the polar scatter plot.
    """

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                           figsize=(6, 6))

    ax.scatter(az_scatter, np.rad2deg(za_scatter), s=1, marker=".", label="From X Pols", c="C0")
    ax.scatter(az_scatter + np.pi/2, np.rad2deg(za_scatter), s=1, marker=".", label="From Y Pols", c="C1")
    # ax.plot(az_scatter, np.rad2deg(za_scatter), c="C0")
    # ax.plot(az_scatter + np.pi/2, np.rad2deg(za_scatter), c="C1")
    ax.grid(False, axis="x")
    ax.legend(loc="upper right")
    fig.tight_layout()

    return fig

def get_perp_slices(az_interp, za_interp):
    """
    Augment an az/za sample set with a perpendicular copy, offset by 90 degrees in azimuth.
    This doubles the number of sample points by appending a rotated duplicate of the track,
    useful for simultaneously sampling both polarization orientations of the beam.

    Parameters:
        az_interp (array):
            Azimuths of interpolation points, in radians.
        za_interp (array):
            Zenith angles of interpolation points, in radians.
    Returns:
        new_az_interp (array):
            Concatenation of original and 90-degree-rotated azimuths, in radians.
        new_za_interp (array):
            Concatenation of original zenith angles with themselves (unchanged by rotation).
    """

    new_az_interp = np.concatenate([az_interp, az_interp + np.pi/2])
    new_za_interp = np.concatenate([za_interp, za_interp])

    return new_az_interp, new_za_interp

def prep_for_interp(az_interp, za_interp, get_beam=True, unpert_sb=unpert_sb, use_perp_slices=True):
    """
    Prepare interpolation inputs by constructing the Fourier-Bessel basis matrices and,
    optionally, sampling the reference beam at the given az/za coordinates.

    Optionally augments the sample points with a perpendicular polarization slice
    (see get_perp_slices). Any points where the interpolated beam is non-finite
    (e.g. below the horizon or outside the beam model support) are masked out.

    Parameters:
        az_interp (array):
            Azimuths of interpolation points, in radians.
        za_interp (array):
            Zenith angles of interpolation points, in radians.
        get_beam (bool):
            If True, evaluate the UVBeam object at the sample points and return
            the log beam amplitude. Default True.
        unpert_sb (sparse_beam):
            Unperturbed sparse beam object used to construct the design matrices.
        use_perp_slices (bool):
            If True, augment az/za with a 90-degree-rotated copy before processing.
            Default True.
    Returns:
        new_az_interp (array):
            Final azimuths used for interpolation (after optional augmentation and masking), in radians.
        new_za_interp (array):
            Final zenith angles used for interpolation (after optional augmentation and masking), in radians.
        bess_matr (array):
            Bessel component of the Fourier-Bessel design matrix evaluated at the sample points.
        trig_matr (array):
            Fourier component of the Fourier-Bessel design matrix evaluated at the sample points.
        interped_beam (array or None):
            Log beam amplitudes at the sample points, or None if get_beam is False.
        finite (array or None):
            Boolean mask of which sample points had finite beam values, or None if get_beam is False.
    """
    if use_perp_slices:
        new_az_interp, new_za_interp = get_perp_slices(az_interp, za_interp)
    else:
        new_az_interp = np.copy(az_interp)
        new_za_interp = np.copy(za_interp)
    bess_matr, trig_matr = unpert_sb.get_dmatr_interp(new_az_interp, new_za_interp)
    if get_beam:
        interped_beam = np.log(UVB.interp(az_array=new_az_interp % (2 * np.pi),
                                          za_array=new_za_interp,
                                          freq_array=UVB.freq_array[test_freq:test_freq + 1])[0][0, 0, 0])
        finite = np.isfinite(interped_beam)


        if not np.all(finite):
            print(f"Fraction of acceptable entries: {np.sum(finite) / finite.size})")
            interped_beam = interped_beam[finite]
            new_az_interp = new_az_interp[finite]
            new_za_interp = new_za_interp[finite]


            bess_matr = bess_matr[finite]
            trig_matr = trig_matr[finite]
    else:
        interped_beam = None
        finite = None

    return new_az_interp, new_za_interp, bess_matr, trig_matr, interped_beam, finite

def plot_sliced_beam(az_interp, za_interp, interped_beam):
    """
    Plot beam amplitude at the sampled az/za points on a polar axis, colored by amplitude.

    Parameters:
        az_interp (array):
            Azimuths of sample points, in radians.
        za_interp (array):
            Zenith angles of sample points, in radians.
        interped_beam (array):
            Log beam amplitudes at the sample points (as returned by prep_for_interp).
            Exponentiated before display.
    Returns:
        fig (matplotlib.figure.Figure):
            The figure containing the polar scatter plot.
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                            figsize=(6, 6))

    im = ax.scatter(az_interp, np.rad2deg(za_interp), c=np.exp(interped_beam), cmap="magma",
                    norm=LogNorm(vmin=1.9e-4))
    fig.colorbar(im, label="Beam Amplitude")

    return fig

def fourier_to_cos_sin(Tmatr, dm):
    """
    Convert basis functions from complex exponential to cosine/sine representation.
    Helps keep the fits real-valued.

    Parameters:
        Tmatr (array):
            Input Fourier basis functions in complex exponential representation.
        dm (int):
            Max azimuthal mode number to use for the fit.
    Returns:
        new_Tmatr (array):
            Cosine/Sine representation of input Tmatr, up to mode number dm.
    """

    new_Tmatr = np.copy(Tmatr)
    new_Tmatr[:, :dm] = np.sqrt(2) * Tmatr[:, :dm].real
    new_Tmatr[:, dm] = Tmatr[:, dm].real
    new_Tmatr[:, dm + 1:] = np.sqrt(2) * Tmatr[:, dm + 1:].imag
    return new_Tmatr

def get_Dmatr(bess_matr, trig_matr, dm=12, Nrad=50, recon=False, Ninterp=0):
    """
    Construct full design matrix based on a restricted number of azimuthal and radial modes.
    Optionally return the 'reconstruction' version, which evaluates the beam at the original Az, Za
    grid from the beam file.

    Parameters:
        bess_matr (array):
            Bessel component of the Fourier-Bessel design matrix evaluated at the sample points.
        trig_matr (array):
            Fourier component of the Fourier-Bessel design matrix evaluated at the sample points.
        dm (int):
            Max azimuthal mode number to use for the fit.
        Nrad (int):
            Number of radial modes to use for the fit.
        recon (bool):
            Whether to form the reconstruction version or not.
    Returns:
        Dmatr (array):
            The design matrix mapping from FB coefficients to points on the sky.
    """

    Nparam = 2 * dm * Nrad

    Tmatr = trig_matr[:, mmax - dm:mmax + dm]
    Tmatr = fourier_to_cos_sin(Tmatr, dm).astype(float)

    Bmatr = bess_matr[:, :Nrad]
    if recon:
        Dmatr = Bmatr[:, None, :, None] * Tmatr[None, :, None, :]
        Dmatr = Dmatr.reshape(Dmatr.shape[:2] + (Nparam,))
    else:
        Dmatr = Bmatr[:, :, None] * Tmatr[:, None]
        Dmatr = Dmatr.reshape(Ninterp, Nparam)
    return Dmatr

def weighted_least_squares(Dmatr, weights, y):
    """
    Compute solution, x, to weighted least squares problem Dmatr @ x = y.

    Parameters:
        Dmatr (array):
            Design matrix in question.
        weights (array):
            Weights at sample points.
        y (array):
            The right-hand-side vector in the least-squares equation (the interpolated beam).
    Returns:
        coeffs (array):
            The coefficients for the columns of Dmatr solving the problem.
    """
    LHS = (Dmatr.T * weights) @ Dmatr
    RHS = Dmatr.T @ (weights * y)

    coeffs = np.linalg.solve(LHS, RHS)

    return coeffs

def plot_beam_comparison(test_beam, recon, norm=LogNorm(vmin=1e-5, vmax=1)):
    """
    Plot a beam reconstruction side by side with the test beam

    Parameters:
        recon (array):
            Reconstructed beam evaluated at the Az, Za grid (i.e. non-interpolated points).
        norm (matplotlib.colors.Normalize):
            A Normalize instance (or subclass) for setting the colorbar on the plot.
    Returns:
        fig (matplotlib.Figure):
            The figure that was drawn on.
    """


    fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                           figsize=(13, 6.5), ncols=2)
    im = ax[0].pcolormesh(Az[:11], Za[:11] * 180 / np.pi, np.exp(test_beam.real[:11]) , norm=norm, cmap="magma")
    im = ax[1].pcolormesh(Az[:11], Za[:11] * 180 / np.pi, np.exp(recon.real)[:11] , norm=norm, cmap="magma")
    ax[0].set_title("Simulated Beam")
    ax[1].set_title("Reconstructed Beam")
    fig.colorbar(im, ax=ax.ravel().tolist(), label="Beam Amplitude")


    for ax_ob in ax:
        ax_ob.grid(False)
    return fig

def get_symmetric_inds(inds,az_scatter):
    """
    Take some indices mapping to one half of the FoV and get the symmetric ones, as well as the centre index.

    Parameters:
        inds (array):
            The indices for which to get the partner indices.
    Returns:
        symmetric_inds (array):
            Concatenation of inds, with its partner set, and the centre index.
    """
    centre_ind = (len(az_scatter) - 1) // 2
    assert all(inds < centre_ind)
    symmetric_inds = np.concatenate([inds, -(inds + 1)])
    symmetric_inds = np.append(centre_ind, symmetric_inds)

    return symmetric_inds

def get_symmetric_az_za_interp(az_scatter, za_scatter, inds, north_south_stripes=False):
    """

    Make a function for pulling symmetric track sets, and plot an example

    Take some indices into a list of tracks and get the symmetric pair.
    Will probably fail silently if some indices are greater than half the length of the list.
    Should not already have the rotated complement appended (will do so here).

    Parameters:
        az_scatter (list):
            List where each element is an array of azimuths for a particular track.
        za_scatter (list):
            List where each element is an array of zenith angles for a particular track.
        inds (array_like):
            Indices into the az_scatter and za_scatter lists (i.e. which tracks on one half of the sky to use).
        centre_ind (int):
            The index of the centre track.
    Returns:
        az_interp, za_interp (array):
            Azimuths and zenith angles to interpolate to (including the perpendicular complement).
    """

    symmetric_inds = get_symmetric_inds(inds,az_scatter)
    az_interp = [az_scatter[ind] for ind in symmetric_inds]
    za_interp = [za_scatter[ind] for ind in symmetric_inds]

    az_interp = np.hstack(az_interp)
    za_interp = np.hstack(za_interp)

    # Get the vertical stripes
    if north_south_stripes:
        az_interp, za_interp = get_perp_slices(az_interp, za_interp)

    return az_interp, za_interp

def get_Dmatr_from_trees(az_scatter, za_scatter, Dmatr_trees, noise_sigma_trees, 
                         inds, get_az_za=False, get_noise_sigma=False):
    """
    Grab the appropriate rows of the design matrices from the lists of matrices computed above.
    Optionally get the corresponding az/za and noise standard deviation.

    Parameters:
        inds (array):
            Indices on one half of field of view (will get the symmetric ones at runtime).
        get_az_za (bool):
            Return the az and za for the rows in question.
        get_noise_sigma (bool):
            Return the noise standard deviation for the az/za in question.
    Returns:
        Dmatr (array):
            The design matrix evaluated at all rows, including the perpendicular slices from sampling.
        az (array or None):
            Azimuth for rows in question.
        za (array or None):
            Zenith angle for rows in question.
        noise_sigma (array or None):
            Noise standard deviation for rows in question.
    """
    Dmatr_tree_x,Dmatr_tree_y = Dmatr_trees
    noise_sigma_tree_x,noise_sigma_tree_y = noise_sigma_trees

    symmetric_inds = get_symmetric_inds(inds,az_scatter)
    Dmatr_x = [Dmatr_tree_x[ind] for ind in symmetric_inds] # horizontal slices
    Dmatr_y = [Dmatr_tree_y[ind] for ind in symmetric_inds] # vertical slices

    Dmatr = np.concatenate(Dmatr_x + Dmatr_y, axis=0).astype(float)

    if get_az_za:
        az_x = [az_scatter[k] for k in symmetric_inds]
        za_x = [za_scatter[k] for k in symmetric_inds]

        az_y = [az_scatter[k] + np.pi/2 for k in symmetric_inds]
        za_y = [za_scatter[k] for k in symmetric_inds]

        az = np.concatenate(az_x + az_y, axis=0)
        za = np.concatenate(za_x + za_y, axis=0)
    else:
        az = None
        za = None

    if get_noise_sigma:
        noise_sigma_x = [noise_sigma_tree_x[ind] for ind in symmetric_inds]
        noise_sigma_y = [noise_sigma_tree_y[ind] for ind in symmetric_inds]
        noise_sigma = np.concatenate(noise_sigma_x + noise_sigma_y, axis=0).astype(float)
    else:
        noise_sigma = None

    return Dmatr, az, za, noise_sigma

def fisher_loop(az_scatter, za_scatter, Dmatr_trees, noise_sigma_trees, 
                use_noise_weighting=True):
    """
    Brute force compute Fisher information (specifically, log-det of Fisher matrix) for all symmetric
    5-index combinations (11-night experiments).

    Parameters:
        use_noise_weighting (bool):
            Whether to do inverse variance weighting  (i.e. actually compute a Fisher matrix and
            not just a Gram matrix).
    Returns:
        fishers (dict):
            Dictionary where the keys are 5-index tuples and values are log-determinants of the Fisher matrices.
    """

    # multi-night experiment will have az_scatter structure like (N_nights, N_els, N_times)
    centre_ind = (len(az_scatter) - 1) // 2

    fishers = {}
    combinations_for_fisher_loop=combinations(range(centre_ind), 5)
    k=0
    for inds in list(combinations_for_fisher_loop):
        Dmatr, _, _, noise_sigma = get_Dmatr_from_trees(az_scatter, za_scatter, Dmatr_trees, noise_sigma_trees,
                                                           np.array(inds), get_noise_sigma=use_noise_weighting)
        if use_noise_weighting:
            Dmatr /= noise_sigma[:, None]
        
        fishers[inds] = np.linalg.slogdet(Dmatr.T @ Dmatr)[1]
        k+=1 # counting manually because this is just a check and the loop is not meant to be index-based
    return fishers