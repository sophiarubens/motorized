from transit_beam import *

# same between all cases: CasA, 24 hrs of obs
cases=[ ["Mike_Aman", "el", True],  # 90º feed rotation <-> measure using both pols
        ["Evan", "nod", False],     # small-amplitude slow nod; realistic transits
        ["Sophia", "nod", False] ]  # large-amplitude slow nod; realistic transits

for case in cases:
    name, mode, north_south_stripes = case

    # >>> >>> >>> preposterously many tracks -> can think of as a not-in-the-programming-sense library to draw from after computing the Fisher-optimal sampling pattern
    az_scatter, za_scatter = draw_tracks(el_delta=0.5, mode=mode) # compute the tracks
    fig = plot_tracks_within_fov(np.concatenate(az_scatter),      # plot the tracks (all)
                                np.concatenate(za_scatter))
    fig.savefig(f"{outdir}/el_tracks_"+name+".png",dpi=dpiuse)

    # >>> >>> >>> pick optimal tracks
    # cache design matrices for easier Fisher computation
    fisher_path = f"{outdir}/fishers_"+name+".yaml"
    RECALC = False
    if os.path.exists(fisher_path) and not RECALC:      # import if possible
        with open(fisher_path, "rb") as fisher_file:
            fishers = pickle.load(fisher_file)
    else:
        fishers = fisher_loop(az_scatter, use_noise_weighting=True) # otherwise, recompute
        with open(fisher_path, "wb") as fisher_file:
            pickle.dump(fishers, fisher_file)

    # >>> >>> >>> determine indices of elevation tracks corresponding to beam slices that contribute the most to the Fisher information 
    the_max = -np.inf
    for inds, logdet in fishers.items():
        if logdet > the_max:
            the_max = logdet
            max_inds = inds

    # >>> >>> >>> plot interpolated versions of Fisher-optimal tracks
    az_interp, za_interp = get_symmetric_az_za_interp(az_scatter, za_scatter, np.array(max_inds), north_south_stripes=north_south_stripes)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                        figsize=(6.5, 6.5))
    ax.scatter(az_interp, np.rad2deg(za_interp), s=1)
    fig.savefig(f"{outdir}/"+case+"_optimal_tracks.png",dpi=dpiuse)