
set dsets iris breast-cancer-wisconsin column3C ecoli ionosphere 


for dset in $dsets
    python sensitivity_analysis.py real -sigma 0.01 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.02 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.03 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.04 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.05 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.06 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.07 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.09 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.10 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.11 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.15 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.25 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.35 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.5 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.6 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.7 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.8 -u $dset UCI
    python sensitivity_analysis.py real -sigma 0.9 -u $dset UCI
end
