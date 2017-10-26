
#set dsets iris breast-cancer-wisconsin column3C ecoli ionosphere 
set dsets breast-cancer-wisconsin

# Small hack to bypass system comma for decimal values

for dset in $dsets
    for sigma in (seq 0 0.05 2 | sed 's/,/\./')
        python sensitivity_analysis.py real -sigma $sigma -u $dset UCI
    end
end
