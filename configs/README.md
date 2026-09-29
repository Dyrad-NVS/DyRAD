# Configuration layout

A run config is a stack of yaml files merged in order (later files win), declared with `base:`:

    recipes/radial.yaml              the sensor and training recipe every run starts from
    recipes/{boreas,synthetic}.yaml  the keys that differ for that dataset (layered after radial.yaml)
    sequences/<dataset>_<seq>.yaml   the sequence: data paths, seq_name, frame window, sensor overrides
    variants/<variant>.yaml          the method variant: dyrad, dyrad_static, abl_* (paper Sec. 4.4)

The recipes, the variants and `benchmarks/offpath_real_*.json` are written by hand. The sequence
stubs are written by hand or by the prepare step (`python -m dyrad.preprocessing.radial prepare`
writes `sequences/radial_<ID>.yaml` for a new window), and
`benchmarks/synthetic_norm_ranges.json` is measured by `python -m dyrad.synthetic.prepare`.
Everything under `radial/`, `boreas/`, `synthetic/` and `offpath_real/` is generated from the
three tiers:

    python -m dyrad.generate_configs            # rewrite
    python -m dyrad.generate_configs --check    # verify the committed files

Each generated file holds the `base:` stack, the init-cloud path and the result directory; the
`offpath_real/` files also carry the data paths, window and `test_every: 0` of their off-path view.
`offpath_real/<dataset>_<seq>_<variant>_{M0,M1}.yaml` are the two fits of the real-data off-path
protocol, and `offpath_real/<dataset>_<seq>_score_T0.yaml` is the variant-independent config
`offpath_real score` uses to read the real T0 measurements. Edit a stub, recipe or variant and
regenerate; never edit a generated file.

`benchmarks/offpath_real_<dataset>.json` defines the real-data off-path protocol (sequences, window
length, lateral shift, init-cloud parameters, variants). `benchmarks/synthetic_norm_ranges.json`
holds the per-scene RA / RD / AD normalization ranges of the synthetic benchmark used by the
scorer; `synthetic.prepare` rewrites a scene's entry every time it prepares that scene.

Init-cloud file names encode the window and holdout the cloud was built for:

    radar_cloud_f<start>_<end>_<split>_<voxel>m[_free[_psf<k>]].npy

`f<start>_<end>` is the frame window (end inclusive), `<split>` is `no_every<K>` when every K-th
frame was left out or `full` when none was (`test_every: 0`, the off-path fits and the synthetic
benchmark), `<voxel>m` the dedup voxel, and `_free` marks the RA-partitioned cloud of the full
method (object footprints removed; `_psf<k>` when they were dilated by k PSF half-widths);
DyRAD-static uses the unpartitioned cloud. The trainer checks the cloud's `.partition.json` sidecar against the config.
