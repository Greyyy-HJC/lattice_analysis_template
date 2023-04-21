# lattice_analysis_template
This is a template repo for lattice analysis, especially under LaMET

## Folder structure
```
data folder: [keep at local only] raw data files, such as the data from lattice QCD simulations, usually in .h5 format

cache folder: cache files, such as the data after pre-processing

output folder: [keep at local only] output files, including the plots and dump files
    dump folder
    plots folder
    else folder

log folder: log files and the plots, such as ground state fit result and the plots

scripts folder: scripts for specific project, not for general template

liblattice folder: the liblattice library, which is general for most lattice analysis projects, subfolders are organized by the analysis steps
    pre_process: read and pre-process the raw data, like dispersion relation, bootstrap, select out the bad configs, etc.
    gs_fit: ground state fit
    coor_to_mom: extrapolation in the coordinate space and FT
    matching: matching
    physics_limit: large momentum extrapolation, continuum limit, physical pion mass extrapolation, etc.
    general: other general functions, such as constants, plot settings, etc.
```