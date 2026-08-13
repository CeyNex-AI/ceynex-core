---
description: Run the shared rolling-origin backtest harness and write metadata.json
argument-hint: <sector> <item> (e.g. agriculture cinnamon)
---

Run the rolling-origin backtest for sector `$1`, item `$2`.

```bash
make backtest SECTOR=$1 ITEM=$2
```

This uses `eval/backtest.py` — the **one** shared harness. Do not write a second
one: a single implementation is what makes M1's, M2's and M3's numbers
comparable in `docs/EVALUATION.md`.

After it runs, check and report:

1. **Expanding window, not a random split.** A random `train_test_split` on a
   time series leaks the future and produces a beautiful, meaningless MAPE. The
   harness enforces this; confirm the fold boundaries in the output are
   monotonically increasing.
2. **MAPE, RMSE, and interval coverage** are all present in the written
   `models/$1/$2/<target>/<version>/metadata.json`, along with train window,
   features, model type, `benchmark_ref`, `trained_at`, and git SHA.
3. **Coverage sanity.** Do roughly 80% of held-out actuals fall inside the 80%
   prediction intervals? If not, say so in the report rather than quietly
   shipping the number.
4. **If the result beats the published benchmark, be suspicious before pleased.**
   Check for overlapping train/test periods, a feature that encodes the target,
   and FX data dated after the forecast origin. Report what you checked.

Published benchmarks: cinnamon → Marasinghe et al. / Liyanage & Silva (2025);
tea → Mampitiya et al. (2025). Apparel has **no** published Sri Lankan
benchmark — compare against seasonal naive and state the absence explicitly.
