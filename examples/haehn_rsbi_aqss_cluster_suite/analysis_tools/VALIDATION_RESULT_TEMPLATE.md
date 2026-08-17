# Haehn/RSBI AQSS Validation Result

## Case Identity

- Case name:
- Run directory:
- Date:
- MFC branch/build:
- Case file:
- Mechanism:
- Resolution / cells_per_d0:
- Grid:
- `reaction_substeps`:
- `mpp_lim`:
- Notes:

## Solver Stability Diagnostics

- Completed / failed:
- Last reached `t_step`:
- First abort:
- `AQSS_PRE_REPAIR` count:
- `AQSS_PRE_ABORT` count:
- `AQSS_POST_ABORT` count:
- Worst `minY`:
- Worst pre-closure error:
- Worst post-closure error:
- NaN / solver instability evidence:

## Chemistry Metrics

- First nonzero heat step:
- Max `heat_abs_sum`:
- Max `heat_pos_sum`:
- Max `heat_neg_sum` magnitude:
- Evidence of ignition / no ignition:
- Product/radical output availability:
- Interpretation: interface-stability only or chemistry-active:

## Comparison Table Against Previous Haehn Runs

| Run | Resolution | Chemistry path | Limiter / repair | Outcome | First issue | Heat activity | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inert Haehn RSBI | | reactions off | | | | | |
| reacting Haehn limiter ON | | explicit/AQSS | limiter ON | | | | |
| coarse limiter ON | | | limiter ON | | | | |
| coarse limiter OFF | cpd30 | fixed AQSS | `mpp_lim=F` | | | | |
| explicit subcycles N=5 | | explicit | | | | | |
| explicit subcycles N=10 | | explicit | | | | | |
| AQSS cpd60_t130_nsub10 | cpd60 | fixed AQSS | pre-repair only | | | | |
| AQSS cpd120_t650_nsub10 | cpd120 | fixed AQSS | pre-repair only | | | | |

## Decision Logic

- If `AQSS_PRE_ABORT` disappears at higher resolution and heat remains zero through t130, classify the coarse failure as an interface/species-advection resolution artifact.
- If `AQSS_PRE_ABORT` persists with similar magnitude at cpd60/cpd120, investigate species transport at the composition interface before changing AQSS.
- If `AQSS_POST_ABORT` appears, treat it as an AQSS chemistry or AQSS delta-writeback failure, not as pre-existing interface drift.
- If heat becomes nonzero before any abort, separate chemistry activity from pure interface stability.
- Do not proceed to longer/full cases until the corresponding t130 case passes.
