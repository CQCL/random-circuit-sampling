# Random circuit sampling on H2
==============================

Project Organization
------------
    ├── amplitudes          <- Ideal amplitudes of the measured bit strings in /results, computed by 
                            statevector simulation.
    ├── analysis            <- Contains functions required to parse results and produce plots contained in 
                              arXiv:2406.02501.
    ├── circuits            <- OpenQASM files for circuits executed on H2. Circuits at N=56 and various depths are contained in
                               /N56_depths. Circuits at depth-12 for various N are contained in /N_scan_depth12. Circuits at
                               N=40 and various depths are contained in /N40_verification. Folders are labeled according to
                               whether corresponding circuits are part of an RCS experiment (XEB), mirror-benchmarking experiment
                               (MB), or transport 1Q RB experiment (Transport_1QRB). Circuit instances are labeled by qubit 
                               number N, depth d, and instance number r. This labeling convention also applies to all files
                               contained in /amplitudes and /results.
    ├── results             <- Measured bit strings and ideal bit strings (if applicable) for all circuits 
                              executed on H2. Endianness of bit strings matches pytket convention.
    └── system_benchmarking <- Results and analysis for component benchmarks.

------------
<div align="center"> &copy; 2024 by Quantinuum. All Rights Reserved. </div>