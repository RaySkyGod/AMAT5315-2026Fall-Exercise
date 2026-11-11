## Periodic-square peak study (TRG)

| L | T_peak χ=16, h=1.25e-3 | T_peak χ=32, h=1.25e-3 | T_peak χ=32, h=2.5e-3 | c_V max χ=16 | c_V max χ=32 |
|---|---|---|---|---|---|
| 64 | 2.283844 | 2.281602 | 2.281598 | 2.279454 | 2.248992 |
| 256 | 2.272537 | 2.272260 | 2.271900 | 2.929941 | 2.902792 |
| 1024 | 2.270813 | 2.270012 | 2.270709 | 3.188754 | 3.096011 |

Peak stability under rank and step changes (target < 1%):

| L | |ΔT|/T rank χ16→32 | |ΔT|/T step h→2h (χ=32) |
|---|---|---|
| 64 | 0.09826% | 0.00016% |
| 256 | 0.01218% | 0.01586% |
| 1024 | 0.03525% | 0.03067% |

T_peak(L) = T_c + a/L fits at χ = 32 (Onsager T_c = 2.269185):

| fit | sizes | T_c | a | |T_c - Onsager|/T_c |
|---|---|---|---|---|
| all_sizes | 64 256 1024 | 2.269205 | 0.7929 | 0.0009% |
| largest_two | 256 1024 | 2.269263 | 0.7672 | 0.0034% |

Timings per run (solver seconds): L64 chi16: mean 0.319s max 0.338s (n=9), L64 chi32: mean 19.699s max 20.653s (n=9), L256 chi16: mean 0.721s max 0.804s (n=9), L256 chi32: mean 29.012s max 30.211s (n=9), L1024 chi16: mean 0.822s max 0.906s (n=9), L1024 chi32: mean 42.332s max 50.365s (n=9)
