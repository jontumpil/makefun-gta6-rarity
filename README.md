# makefun-gta6-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jontumpil/makefun-gta6-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jontumpil/makefun-gta6-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A count-consistent refresh forecaster for GTA VI's trophy earn rates on Exophase. The house's
reads repeat the page between Exophase refreshes, so the program first compresses a case's
history to its refresh states. In state space it extrapolates counts, not rates: players
tracked and each trophy's earners (rate × players) grow log-linearly against
ln(1 + hours since the first state), fitted by recency-weighted least squares with per-trophy
slopes shrunk toward the total-earners slope. The per-trophy earners are rescaled so they sum
to the extrapolated total, then divided by the extrapolated player count — so a rate falls
exactly when its earners grow slower than the player base (dilution) and rises when they grow
faster (progression). Quantiles sit on the logit scale around that median, with a spread that
widens with the horizon, the fit's residual volatility, and refresh uncertainty. One file,
Python 3.13 standard library only, fully deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=w7fy1fnp&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
