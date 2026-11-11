## Open-square peak study (boundary MPS)

| L | T_peak χ=16, h=1.25e-3 | T_peak χ=32, h=1.25e-3 | T_peak χ=32, h=2.5e-3 | c_V max χ=16 | c_V max χ=32 |
|---|---|---|---|---|---|
| 16 | 2.093105 | 2.093105 | 2.093075 | 1.115438 | 1.115438 |
| 32 | 2.179676 | 2.179676 | 2.179655 | 1.387531 | 1.387531 |
| 64 | 2.224219 | 2.224220 | 2.224194 | 1.683047 | 1.683047 |

Peak stability under rank and step changes (target < 1%):

| L | |ΔT|/T rank χ16→32 | |ΔT|/T step h→2h (χ=32) |
|---|---|---|
| 16 | 0.00000% | 0.00145% |
| 32 | 0.00000% | 0.00097% |
| 64 | 0.00004% | 0.00117% |

T_peak(L) = T_c + a/L fits at χ = 32 (Onsager T_c = 2.269185):

| fit | sizes | T_c | a | |T_c - Onsager|/T_c |
|---|---|---|---|---|
| all_sizes | 16 32 64 | 2.267506 | -2.7933 | 0.0740% |
| largest_two | 32 64 | 2.268765 | -2.8508 | 0.0185% |

Timings per run (solver seconds): L16 chi16: mean 0.061s max 0.067s (n=9), L16 chi32: mean 0.160s max 0.186s (n=9), L32 chi16: mean 0.363s max 0.436s (n=9), L32 chi32: mean 1.297s max 1.391s (n=9), L64 chi16: mean 1.684s max 1.783s (n=9), L64 chi32: mean 6.378s max 6.866s (n=9)
