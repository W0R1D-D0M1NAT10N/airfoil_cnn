# Figure provenance

Source: `Flight_Vehicle_Design_JEI_Final.pdf`, the 18-page research manuscript by Alex Yu and Fei Liu supplied for this repository presentation. See the [source notes](../research/README.md). The source filename does not establish publication status, and the full PDF is not distributed here.

The six `manuscript-*.png` files were extracted from the PDF's embedded images, without redrawing data or altering labels.

| Asset | Original location | Notes |
| --- | --- | --- |
| [Airfoils and CNN](manuscript-fig1-airfoils-cnn.png) | Figure 1, p. 13 | Diagram labels and caption describe different head sizes; neither matches the GPU script in full |
| [Airfoil comparison curves](manuscript-fig2-airfoil-comparisons.png) | Figure 2, p. 14 | Selected SimFlow/XFOIL/ML curves; ML calibration status not specified |
| [Radius functions](manuscript-fig3-radius-functions.png) | Figure 3, p. 15 | Manuscript geometry illustration; complete generation scripts are not present |
| [Pods and PointNet](manuscript-fig4-pods-pointnet.png) | Figure 4, p. 16 | Shows dropout and normalized AoA, absent from the checked-in 3D model/data path |
| [CFD lift and drag](manuscript-fig5-cfd-lift-drag.png) | Figure 5, p. 16 | Drag scale differs from the committed CSV; see [results notes](../RESULTS.md#simulation-and-data-provenance) |
| [PointNet comparison curves](manuscript-fig6-pointnet-comparisons.png) | Figure 6, p. 17 | Eight-run means/error bars reported by the manuscript; underlying prediction tables unavailable |
| [Pod geometry strip](pod-geometries.png) | Crop of Figure 4(a) | Pixel crop `(20, 65, 1220, 205)` from the extracted 1280 × 720 image; labels and all four shapes retained |
| [Model architectures](model-architectures.svg) | New documentation diagram | Describes `airfoilNN_nvidia.py` and `3DModel/model.py`, not a measured output or the saved CNN checkpoint |

All manuscript plots are historical figures, not newly reproduced experiments. The original legends, colors, and annotations are retained; use the legends rather than assuming the prose's color descriptions are correct. In particular, Figure 6 displays calibrated predictions in green, although the main text describes them as orange.

The source PDF was inspected locally. Its Appendix points to `W0R1D-D0M1NAT10N/Machine_Learning`; this documentation audits the user-selected `airfoil_cnn` snapshot instead. Figure extraction does not validate a plot's underlying numerical data. No generated or illustrative replacement data were used.
