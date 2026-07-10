from transit_beam import *

"""### Draw some tracks with rotated feeds for 15 degree slice separation and 90 degree slice separation.

Keeping the az and za of slices in some dictionaries, keyed by slice separation, for further use.
"""

az_scatter_dict = {}
za_scatter_dict = {}
for slice_sep in [15, 90]:
    az_scatter, za_scatter = draw_tracks(rots=np.arange(0,-90,-slice_sep)) # E to N convention
    az_scatter_dict[slice_sep] = az_scatter
    za_scatter_dict[slice_sep] = za_scatter

    fig = plot_tracks_within_fov(az_scatter, za_scatter)
    fig.savefig(f"{outdir}/rot_tracks_{slice_sep}.png",dpi=dpiuse)

for slice_sep in [15, 90]:

    az_interp, za_interp, bess_matr, trig_matr, interped_beam, _ = prep_for_interp(az_scatter_dict[slice_sep],
                                                                                   za_scatter_dict[slice_sep])



    fig = plot_sliced_beam(az_interp, za_interp, interped_beam)
    fig.savefig(f"{outdir}/sliced_beam_{slice_sep}.png",dpi=dpiuse)

"""### Do some experimentation to see a good number of radial modes to use.

Real data will need a different type of test that makes for an interesting stats problem about overfitting. WARNING: may have developed some bugs since I ran it last.
"""

RADIAL_DETERMINATION_EXPERIMENT = False
if RADIAL_DETERMINATION_EXPERIMENT:
    Nrads=10
    test_beam=None # need to fix if I want to reconstruct Mike's experiment
    rms_vals = []
    az_interp, za_interp = get_perp_slices(az_scatter, za_scatter)
    for Nrad in Nrads:


        az_interp, za_interp, bess_matr, trig_matr, interped_beam, _ = prep_for_interp(az_interp, za_interp)
        Dmatr = get_Dmatr(bess_matr, trig_matr, Nrad=Nrad)
        Dmatr_recon = get_Dmatr(unpert_sb.bess_matr, unpert_sb.trig_matr, recon=True, Nrad=Nrad)

        noise_sigma = 1e-4/np.exp(interped_beam)
        Ninv = 1/noise_sigma**2
        coeffs = weighted_least_squares(Dmatr, Ninv, interped_beam)
        #coeffs = np.linalg.lstsq(Dmatr, interped_beam)[0]
        errors = np.exp((Dmatr @ coeffs)) - np.exp(interped_beam)

        recon_beam = Dmatr_recon @ coeffs

        recon_errors = np.exp(recon_beam) - np.exp(test_beam)
        rms = np.sqrt(np.mean(recon_errors**2))
        rms_vals.append(rms)

    plt.plot(Nrads, rms_vals)
    plt.yscale("log")
    plt.ylim([1e-4, 1e-1])
    print(Nrads[np.argmin(rms_vals)])

"""### Fit a beam to an interpolated simulation with a made up, not-totally-insane noise model."""

az_interp, za_interp, bess_matr, trig_matr, interped_beam, _ = prep_for_interp(az_scatter_dict[15],
                                                                               za_scatter_dict[15])
Ninterp = len(az_interp)
# Did some experimentation to determine this number -- will be an interesting stats problem here for the real data
Nrad = 25
Dmatr = get_Dmatr(bess_matr, trig_matr, Nrad=Nrad, Ninterp=Ninterp)
Dmatr_recon = get_Dmatr(unpert_sb.bess_matr, unpert_sb.trig_matr, recon=True, Nrad=Nrad, Ninterp=Ninterp)

# Setting noise according to rough expected fractional error based on beam amplitude
# Real data will want more careful noise modeling
noise_sigma = 1e-4/np.exp(interped_beam)
Ninv = 1/noise_sigma**2

# Inverse-variance weighted least-squares fit (i.e. maximum likelihood)
coeffs = weighted_least_squares(Dmatr, Ninv, interped_beam)

errors = np.exp((Dmatr @ coeffs)) - np.exp(interped_beam)
recon_beam = Dmatr_recon @ coeffs
print(f"RMS errors: {np.sqrt(np.mean(errors**2))}")

"""### Plot residuals and beam at sampled points"""

fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                       figsize=(6.5, 6.5))
im1 = ax.scatter(az_interp, za_interp, c=errors,
                 cmap="coolwarm", norm=SymLogNorm(linthresh=1e-4, vmin=-1e-1, vmax=1e-1))

fig.colorbar(im1, ax=ax, label="Residual")
plt.savefig("residuals_and_beam.png",dpi=dpiuse)

"""### Plot reconstructed beam compared to simulation"""

# Evulate the input beam in the FoV at the test frequency
test_beam = np.log(unpert_sb.data_array[0,0,test_freq, :za_range[-1] + 1])

fig = plot_beam_comparison(test_beam, recon_beam)
plt.savefig("beam_comparison.png",dpi=dpiuse)


recon_errors = np.exp(recon_beam) - np.exp(test_beam)
print(f"RMS errors: {np.sqrt(np.mean(recon_errors**2))}")

"""### Plot a slice at a given slice angle. Optionally add some noise to get a feel for what that might look like"""

test_ang = 4
add_noise = False
test_to_plot = test_beam + np.random.normal(scale=1e-4/np.exp(test_beam)) if add_noise else test_beam


plt.plot(np.exp(recon_beam[test_ang, :]), label="Reconstruction")
plt.plot(np.exp(test_to_plot[test_ang, :]), label="Measured Beam")
plt.legend()
plt.savefig("slice_at_one_angle.png",dpi=dpiuse)

"""### Show the residuals of the reconstructed beam"""

fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                       figsize=(6., 6.))
im = ax.pcolormesh(Az, np.rad2deg(Za), recon_errors.real, cmap="coolwarm",
                   norm=SymLogNorm(linthresh=1e-3, vmax=1e-1, vmin=-1e-1))
fig.colorbar(im, label="Reconstruction Error")
fig.savefig(f"{outdir}/recon_errors_az_slicing.png",dpi=dpiuse)

"""### Instead of rotating the feed, get multiple elevation slices. Here I'll draw 21 of them (half degree separation).

az_scatter and za_scatter here are globals that get used in functions later
"""

az_scatter, za_scatter = draw_tracks(el_delta=0.5, mode="el")


fig = plot_tracks_within_fov(np.concatenate(az_scatter), np.concatenate(za_scatter))
fig.savefig(f"{outdir}/el_tracks.png",dpi=dpiuse)

az_interp, za_interp = get_symmetric_az_za_interp(az_scatter, za_scatter, np.array([0, 4, 8, 12, 16]))
fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                       figsize=(6.5, 6.5))
ax.scatter(az_interp, np.rad2deg(za_interp), s=1)
fig.savefig(f"{outdir}/even_grid.png",dpi=dpiuse)

"""### Cache a bunch of design matrices to make the Fisher computation easier."""


fisher_path = f"{outdir}/fishers.yaml"
RECALC = False
if os.path.exists(fisher_path) and not RECALC:
    with open(fisher_path, "rb") as fisher_file:
        fishers = pickle.load(fisher_file)
else:
    fishers = fisher_loop(use_noise_weighting=True)
    with open(fisher_path, "wb") as fisher_file:
        pickle.dump(fishers, fisher_file)

"""### Find the index set with the most Fisher information, plot the slices."""

the_max = -np.inf
for inds, logdet in fishers.items():
    if logdet > the_max:
        the_max = logdet
        max_inds = inds

print(max_inds, logdet)

# Mike's version
az_interp, za_interp = get_symmetric_az_za_interp(az_scatter, za_scatter, np.array(max_inds), north_south_stripes=True)
fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                       figsize=(6.5, 6.5))
ax.scatter(az_interp, np.rad2deg(za_interp), s=1)
fig.savefig(f"{outdir}/Mike_Aman_optimal_pattern.png",dpi=dpiuse)

"""### See how it does"""
assert 1==0, "still refining "

best_Dmatr, az, za, _, sb_for_els = get_Dmatr_from_trees(np.array(max_inds), get_az_za=True)
az_interp, za_interp, _, _, interped_beam, finite = prep_for_interp(az, za, get_beam=True, unpert_sb=sb_for_els,
                                                                    use_perp_slices=False)

noise_sigma = 1e-4/np.exp(interped_beam)
Ninv = 1/noise_sigma**2
coeffs = weighted_least_squares(best_Dmatr, Ninv, interped_beam)

fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                       figsize=(6.5, 6.5))
im = ax.scatter(az_interp, za_interp, c=best_Dmatr @ coeffs)
plt.colorbar(im)
plt.savefig("best_design_matrix.png",dpi=dpiuse)

"""### Residuals at sample points."""

errors = np.exp(best_Dmatr @ coeffs) - np.exp(interped_beam)

fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                       figsize=(6.5, 6.5))
im = ax.scatter(az_interp, np.rad2deg(za_interp), c=errors, norm=SymLogNorm(linthresh=1e-3, vmin=-1e-1, vmax=1e-1),
                cmap="coolwarm")
plt.colorbar(im)
plt.savefig("residuals_at_sample_points.png",dpi=dpiuse)

bess_matr_recon = sb_for_els.bess_matr[:, :Nrad]

dm=12 # hard-coded max mode number to use
trig_matr_recon = fourier_to_cos_sin(sb_for_els.trig_matr, dm)

Dmatr_recon = bess_matr_recon[:, None, :, None] * trig_matr_recon[None, :, None]
Nparam = 2 * dm * Nrad
Dmatr_recon = Dmatr_recon.reshape(11, 360, Nparam)
el_sweep_recon = Dmatr_recon @ coeffs

"""### Side-by-side plot"""

fig = plot_beam_comparison(el_sweep_recon)
fig.savefig(f"{outdir}/recon_el_transits.png",dpi=dpiuse)

"""###"""

plt.plot(np.exp(el_sweep_recon[4, :]), label="Reconstruction")
plt.plot(np.exp(test_beam[4, :]), label="Measured Beam")
plt.legend()
plt.savefig("test_beam.png",dpi=dpiuse)

"""### 2d Residuals"""

errors = np.exp(el_sweep_recon) - np.exp(test_beam)

fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                       figsize=(6., 6.))
im = ax.pcolormesh(Az, np.rad2deg(Za), errors.real, cmap="coolwarm",
                   norm=SymLogNorm(linthresh=1e-3, vmax=1e-1, vmin=-1e-1))
fig.colorbar(im, label="Reconstruction Error")
fig.savefig(f"{outdir}/recon_errors_els.png",dpi=dpiuse)