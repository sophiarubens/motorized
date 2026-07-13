from transit_beam import *

slow_nod_rate=np.pi/300 /u.min # 1º/ 10 s
nod_params_Evan =   [2,  slow_nod_rate]
nod_params_Sophia = [30, slow_nod_rate]

# same between all cases: CasA, 24 hrs of obs
experimental_setups=[ ["Mike_Aman", "el",  True,  None              ],    # 90º feed rotation <-> measure using both pols
                      ["Evan",      "nod", False, nod_params_Evan   ],    # small-amplitude slow nod; realistic transits
                      ["Sophia",    "nod", False, nod_params_Sophia ]  ]  # large-amplitude slow nod; realistic transits

for experimental_setup in experimental_setups:
    name, mode, north_south_stripes, nod_params = experimental_setup
    # mode="rot" # overwrite for array shape compatibility testing benchmarks

    # >>> >>> >>> preposterously many tracks -> can think of as a not-in-the-programming-sense library to draw from after computing the Fisher-optimal sampling pattern
    az_scatter, za_scatter = draw_tracks(el_delta=0.5, rots=np.atleast_1d((0.,)), mode=mode, nod_params=nod_params) # compute the tracks
    print("len(az_scatter) =",len(az_scatter))
    concatenated_az_scatter=np.concatenate(az_scatter)
    concatenated_za_scatter=np.concatenate(za_scatter)
    fig =plot_tracks_within_fov(concatenated_az_scatter,      # plot the tracks (all)
                                concatenated_za_scatter)
    fig.savefig(f"{outdir}/el_tracks_"+name+".png",dpi=dpiuse)
    print("plotted el tracks")

    # >>> >>> >>> cache a bunch of design matrices to speed Fisher computation
    sb_for_els = sparse_beam.sparse_beam(beamfile, nmax=50, 
                                        mmodes=np.arange(-12, 12),
                                        za_range=za_range,
                                        Nfeeds=None,
                                        alpha=rho_const,
                                        convert_to_power=True,
                                        sqrt=False,
                                        load=False)

    dm = 12
    Nrad = 25
    Nparam = 2 * dm * Nrad
    use_noise_weighting = True


    Dmatr_tree_x = []
    Dmatr_tree_y = []

    if use_noise_weighting:
        noise_sigma_tree_x = []
        noise_sigma_tree_y = []

    for k in range(len(az_scatter)):
        _, _, bess_matr_x, trig_matr_x, interped_beam_x, _ = prep_for_interp(az_scatter[k], 
                                                                            za_scatter[k], 
                                                                            unpert_sb=sb_for_els, 
                                                                            get_beam=use_noise_weighting,
                                                                            use_perp_slices=False)
        _, _, bess_matr_y, trig_matr_y, interped_beam_y, _ = prep_for_interp(az_scatter[k] + np.pi/2, 
                                                                            za_scatter[k], 
                                                                            unpert_sb=sb_for_els, 
                                                                            get_beam=use_noise_weighting,
                                                                            use_perp_slices=False)
        
        Tmatr_x = fourier_to_cos_sin(trig_matr_x, dm)
        Tmatr_y = fourier_to_cos_sin(trig_matr_y, dm)
        
        
        Ninterp_x = len(interped_beam_x) if use_noise_weighting else len(az_scatter[k])
        Ninterp_y = len(interped_beam_y) if use_noise_weighting else len(za_scatter[k])
        
        Dmatr_x = bess_matr_x[:, :Nrad, None] * Tmatr_x[:, None]
        Dmatr_x = Dmatr_x.reshape(Ninterp_x, Nparam)
        
        Dmatr_y = bess_matr_y[:, :Nrad, None] * Tmatr_y[:, None]
        Dmatr_y = Dmatr_y.reshape(Ninterp_y, Nparam)
        
        Dmatr_tree_x.append(Dmatr_x)
        Dmatr_tree_y.append(Dmatr_y)
        
        if use_noise_weighting:
            noise_sigma_tree_x.append(1e-4 * np.exp(-interped_beam_x))
            noise_sigma_tree_y.append(1e-4 * np.exp(-interped_beam_y))
    
    Dmatr_trees=[Dmatr_tree_x,Dmatr_tree_y]
    noise_sigma_trees=[noise_sigma_tree_x,noise_sigma_tree_y]
    print("finished caching design matrices")

    # >>> >>> >>> pick optimal tracks
    # make sure the Fisher matrices based on the cached design matrices are easily accessible (import or re-compute)
    fisher_path = f"{outdir}fishers_"+name+".yaml"
    RECALC = True
    # if os.path.exists(fisher_path) and not RECALC:      # import if possible
    if not RECALC:
        with open(fisher_path, "rb") as fisher_file:
            fishers = pickle.load(fisher_file)
        print("imported Fisher matrices")
    else:
        fishers = fisher_loop(az_scatter, za_scatter, Dmatr_trees, noise_sigma_trees, # az_scatter, za_scatter, Dmatr_trees, noise_sigma_trees, 
                              use_noise_weighting=True) # otherwise, recompute
        if not fishers:
            raise ValueError("Did not manage to compute any Fisher matrices = empty dictionary")
        with open(fisher_path, "wb") as fisher_file:
            pickle.dump(fishers, fisher_file)
        print("computed Fisher matrices")

    # >>> >>> >>> determine indices of elevation tracks corresponding to beam slices that contribute the most to the Fisher information 
    the_max = -np.inf
    print("len(fishers.items())=",len(fishers.items()))
    for inds, logdet in fishers.items():
        if logdet > the_max:
            the_max = logdet
            max_inds = inds
    print("determined indices of most useful elevation tracks")

    # >>> >>> >>> plot interpolated versions of Fisher-optimal tracks
    az_interp, za_interp = get_symmetric_az_za_interp(az_scatter, za_scatter, np.array(max_inds), north_south_stripes=north_south_stripes)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                        figsize=(6.5, 6.5))
    ax.scatter(az_interp, np.rad2deg(za_interp), s=1)
    fig.savefig(f"{outdir}/"+name+"_optimal_tracks.png",dpi=dpiuse)