# Ring Attention Benchmark Results
Run on 2x A100 GPUs

| Number Count | Input Tokens | GPUs | Ring (ms) | Regular (ms) | Speedup | Match |
|--------------|--------------|------|-----------|--------------|---------|-------|
| 1000 | 2998 | 2 | 1604.94 ms | 253.94 ms | .15x | FAILED |
| 100 | 298 | 2 | 235.72 ms | 39.60 ms | .16x | FAILED |
| 500 | 1498 | 2 | 483.22 ms | 124.65 ms | .25x | FAILED |
| 1000 | 2998 | 2 | 1232.58 ms | 249.39 ms | .20x | FAILED |
| 1500 | N/A | 2 | ERROR | ERROR | N/A | FAILED |
| 1500 | 4998 | 2 | 2890.67 ms | 413.11 ms | .14x | FAILED |
| 2000 | 6998 | 2 | 5895.96 ms | 663.60 ms | .11x | FAILED |
| 2000 | 6998 | 2 | 5570.09 ms | 603.96 ms | .10x | FAILED |
