"""Real-data off-path evaluation (paper Sec. 4.2): fit M0 on the recorded window, render it
along a laterally shifted trajectory, fit M1 to those renders, render M1 back at the original
poses and score it against the real measurements. Entry point: `python -m dyrad.offpath_real`.

    spec.py    the benchmark definition (configs/benchmarks/offpath_real_<dataset>.json) and its paths
    view.py    the view writer (tensors, poses, ego velocity, labels, normalization range)
    labels.py  re-projection of the annotations into a shifted view
    build.py   view definitions and M0 renders
    evaluate.py, score.py   M1 at the original poses, scored with the shared scorer
"""
