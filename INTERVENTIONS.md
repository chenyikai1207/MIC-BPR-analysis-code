# Matched label binding, source sensitivity and direction replacement

This extension packages existing MIC computations as explicit-array functions
and local CLI stages. It does not alter the scientific estimands, choose new
people or codebooks, or contain real arrays, numerical result tables or papers.
The existing dependency profile and licensing status remain unchanged.

Use Python 3.12.14 and `requirements-analysis.txt`. From this repository:

```text
python -B code/mic_interventions_cli.py environment
python -B code/mic_interventions_cli.py --help
```

Run bounded synthetic tests in a writable directory **outside** this repository:

```powershell
$env:MIC_SYNTHETIC_SCRATCH = '../local/SYNTHETIC_TEST_ONLY'
python -B code/mic_interventions_cli.py synthetic-test
```

On POSIX shells, set the same variable with `export` before the command. Tests
generate all values and identifiers themselves. Their artifacts are retained,
labeled SYNTHETIC_TEST_ONLY and excluded from the public tree. They are numerical
and interface checks, not real-data reproduction.

## Input qualification

Copy `INTERVENTION_QUALIFICATION.example.json` outside the repository and complete
it only after independent identity, legal-access/license, manifest/readability,
event/UID ordering, eligibility, feature and role checks. A Boolean declaration
does not itself perform these checks. All CLI data inputs and outputs must be
outside this repository; NPZ loading disallows object/pickle arrays.

`INTERVENTION_SCHEMAS.json` documents NPZ names, shapes, order and qualification
fields. Real factorial inputs retain the frozen 199 codebooks and all 35 target
people, with 9,999 **supplied** whole-person resamples. The functions never redraw
codes or bootstrap samples. Use behavior-log onset seconds, never native sample
indices, for MEG matching. The source and target people must be disjoint.

## Four-cell intervention

For imagery row j, k is a nonself same-code donor in the same true-melody and
pause-half stratum. l is a shared opposite-code partner in that stratum. Both
minimize absolute behavior-log time displacement, breaking ties by original
row index. Unsupported cells remain unsupported; no fallback stratum is used.
All cells and representations use the same eligible scoring support.

| Arm | Deleted P row | Training label change |
|---|---|---|
| OWN_C | k | none |
| OTHER_C | j | none |
| OWN_X | k | exchange retained labels j and l |
| OTHER_X | j | exchange retained labels k and l |

Each training set has N-1 rows. Within each input condition the C/X neural rows
are identical; the exchange preserves both supervised class counts. Imagery is
always evaluated against its **original** code. It is never a fitting input.
Exact zero evidence predicts class 0. Singular covariance stops the calculation
without a substituted classifier or support change.

```text
python -B code/mic_interventions_cli.py plan --qualification ../local/q.json --input ../local/target.npz --codes ../local/frozen-codes.npz --output ../local/plan-new.npz
python -B code/mic_interventions_cli.py four-cell --qualification ../local/q.json --input ../local/target.npz --model ../local/full-source.npz --plan ../local/plan-new.npz --output ../local/four-cell-new.npz
```

The first command performs metadata matching only. The second projects qualified
phase features through the supplied complete source map, then fits the four
target calibration cells for B4 and PCA16. `--readouts B4` is available for the
complete source-deletion analysis. Unsupported evidence is NaN, not a scored
error. No code is silently dropped when either scored class has no support.

Class recalls are averaged within code, codes within person, then people equally.
In percentage points:

```text
E_C = 100 * mean_codes(BA_OWN_C - BA_OTHER_C)
E_X = 100 * mean_codes(BA_OWN_X - BA_OTHER_X)
J   = E_C - E_X
```

This arbitrary-code interaction is a computational calibration contrast, not a
gain in real-melody recognition. The B4-minus-PCA16 J contrast is paired within
the same person/code/support. Assemble one ordered person-by-code BA tensor from
local scoring outputs, with the original frozen resample indices:

```text
python -B code/mic_interventions_cli.py aggregate-factorial --qualification ../local/q.json --input ../local/all-person-factorial.npz --output ../local/factorial-summary-new.json
```

The percentile interval resamples whole target people, keeping all endpoint
columns paired. It conditions on the fixed source map. Codes are Monte Carlo
draws within people, not independent statistical units.

## Complete source deletion and paired sensitivity

For each of the 35 source people, omit that person's **entire** P/I row set and
refit StandardScaler, PCA95 and B4 from the remaining 34 people. Existing
`meg_source.fit_source_arrays` supplies that complete numerical block:

```text
python -B code/mic_interventions_cli.py refit-source --qualification ../local/q-source.json --input ../local/source35.npz --delete-person SOURCE_ID --output ../local/delete-one-source-new.npz
```

Use each map with the identical target data and frozen plan in `four-cell`; report
every deletion, without outcome-dependent selection. The CLI fits one deletion
per invocation; it does not run an uncontrolled cohort loop. Original-label
scores may be reused only after exact map/UID/code/support correspondence checks.

Supply all source-by-target contrasts in the same target order. For a paired
contrast, subtract endpoint A minus B within each target/map first; do not combine
two separately unpaired intervals. `source-sensitivity` accepts either those
contrasts or explicit two-endpoint tensors:

```text
python -B code/mic_interventions_cli.py source-sensitivity --qualification ../local/q-source.json --input ../local/complete-source-target-contrasts.npz --output ../local/sensitivity-new.json
```

With S source deletions and T targets, theta is the full-source target mean,
d_s is each deletion's target mean, V_s=(S-1)/S sum_s(d_s-mean(d))^2, and
V_t=sample_variance(full-source target contrasts)/T. The reported approximation is
theta +/- t_(S-1,0.975) sqrt(V_s+V_t). The manuscript design has S=T=35.
Disjoint independent source/target sampling, finite variance, stable complete
refits and additive first-order influences are assumptions. Overlapping maps
are not independent replications; exact coverage and higher-order interactions
are not established. Complete-refit and identical-support checks are external
prerequisites, not inferred from a matrix's shape.

## Source-derived direction replacement

The source direction is the equal-source-person difference of true-class P
means, projected through the corresponding source B4 map. For each scored target
imagery row, its own P and label are excluded. With remaining class means mu0/mu1:

```text
delta_t = mu1 - mu0; midpoint = (mu0 + mu1)/2
delta_lambda = (1-lambda)*delta_t + lambda*delta_s
w = inverse(target auto-shrinkage pooled covariance) * delta_lambda
intercept = -midpoint dot w + log(n1/n0)
```

The original midpoint, target covariance and empirical priors remain fixed.
Original LDA, diagonal LDA, pure-source direction and the selected candidate share
the same N-1 target budget. At lambda=1 the candidate equals the preset pure-source
control; this is classifier identity, not an equivalence test.

```text
python -B code/mic_interventions_cli.py direction-prior --qualification ../local/q-source.json --input ../local/allowed-source.npz --model ../local/applicable-source-model.npz --output ../local/source-direction-new.npz
python -B code/mic_interventions_cli.py direction-grid --qualification ../local/q-source-fold.json --input ../local/heldout-source-person.npz --model ../local/source-excluding-heldout.npz --prior ../local/fold-direction.npz --output ../local/source-fold-grid-new.npz
python -B code/mic_interventions_cli.py select-direction --qualification ../local/q-source-fold.json --input ../local/all35-source-fold-BA.npz --output ../local/selection-frozen-new.json
python -B code/mic_interventions_cli.py direction-score --qualification ../local/q-target.json --input ../local/target.npz --model ../local/full-source.npz --prior ../local/full-source-direction.npz --selection ../local/selection-frozen-new.json --task recall --output ../local/target-direction-new.npz
```

Each source fold must exclude its held-out person from scaler, PCA, B4 and the
source direction. Selection averages all source people equally over the fixed
grid [0,.25,.5,.75,1]; ties within 1e-12 choose the smaller value. Freeze the JSON
**before** target scoring. No target score enters selection. `--task reversal`
requires intended true imagery labels equal to 1 minus heard training labels.
Freeze selection and evaluate all retained targets without a rescue search.
Prior/map bindings use local path, byte length, modification time and source
membership; these structural checks do not certify historical signal identity.
Whole-person intervals for the method difference remain conditional on both
source fitting and the selected weight. Historical exposure is not removed by
source-only selection, so this route remains exploratory.

## Acquisition and remaining raw-to-report work

Obtain the original MEG **preprocessed derivatives and logs** from the provider's
[Zenodo data record, Initial upload](https://zenodo.org/records/13760720), and
original-study software from [Zenodo version 1.0](https://zenodo.org/records/13760787).
These are not unprocessed sensor recordings. Speech EEG has its separate
[OpenNeuro ds004306 source](https://openneuro.org/datasets/ds004306).
Check the exact provider version/license, archive/member identity, legal access,
manifest and readability independently; register read-only public caches locally.
No data acquisition occurs in this extension and no digest calculation is needed.

The full cohort acquisition, eligibility/log-event joins, native-file compatibility,
feature materialization, original UID/row and frozen-role instances, resource
watchdogs/checkpoints, assembly of person-by-code/deletion tensors and final
report/figure exports are not a complete automatic public pipeline. The existing
epoch/window/source helpers cover parts of that route. The older exact MEG
serialization/software-run binding was not fully recovered; the acquired EEG
object snapshot was not certified equivalent to a named release tag. Neither
identity is manufactured here. See `SCOPE_AND_GAPS.md`.

The extension fixes BLAS threads to two and checks elapsed budgets between
operations. It does not interrupt an active LAPACK operation or supply a memory,
scratch or OS-affinity watchdog. Supervise complete source fits separately.
Synthetic checks and local reaggregation of saved real scores do not establish
a new raw-data reproduction, independent validation or biological mechanism.

## Implementation provenance

`mic_interventions.py` extracts the executed `masked_binary_lda` and
`intervention_batches` function text from `mic_binding_common.py`. Its
`loo_statistics` retains the original `loo_coefficients` covariance/mean/count
statements and stops before solving coefficients. `mic_intervention_api.py`
provides portable validation, matching, aggregation and the executed direction
formula; historical host IO/checkpoint/digest helpers are excluded.
`mic_interventions_cli.py` adds explicit local file interfaces. Original released
scientific modules are unchanged. The pure numerical blocks are checked against
direct scikit-learn fits and synthetic historical-function oracles; local saved
real-score reaggregation is distinct from these synthetic checks.
