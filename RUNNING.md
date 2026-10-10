# Running the core interfaces

The analysis profile uses Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1,
scikit-learn 1.9.1, threadpoolctl 3.7.0 and joblib 1.6.0. Keep these pins:
the LDA implementation uses scikit-learn's private `_is_constant_feature` helper.
Requirements pin direct dependencies, not a complete OS/BLAS/container lock.

Optional plotting is a separate tested profile: Python 3.12.14, NumPy 2.5.2
and Matplotlib 3.11.1, with Arial for matching typography. Do not substitute the
plotting environment for scientific model computation. Install profiles into
separate environments if needed. No installation or network request is triggered
by code import, help, environment inspection or the synthetic checks.

## Qualified-array commands

These examples expect local inputs you have already qualified. Paths are examples,
not supplied files. All outputs must be new files outside the repository.

```text
python -B code/mic_cli.py validate-input --modality MEG --qualification ../local/meg-qualified.json --input ../local/meg-source.npz
python -B code/mic_cli.py fit-source --modality MEG --qualification ../local/meg-qualified.json --input ../local/meg-source.npz --output ../local/meg-model-new.npz
python -B code/mic_cli.py score-target --modality MEG --task recall --qualification ../local/meg-qualified.json --input ../local/recall-target.npz --model ../local/meg-model-new.npz --assignments ../local/recall-assignment.npz --output ../local/recall-score-new.npz
python -B code/mic_cli.py score-target --modality MEG --task manipulation --qualification ../local/meg-qualified.json --input ../local/manipulation-target.npz --model ../local/meg-model-new.npz --assignments ../local/manipulation-assignment.npz --output ../local/manipulation-score-new.npz
python -B code/mic_cli.py fit-source --modality EEG --qualification ../local/eeg-qualified.json --input ../local/eeg-source.npz --output ../local/eeg-model-new.npz
python -B code/mic_cli.py score-target --modality EEG --qualification ../local/eeg-qualified.json --input ../local/eeg-target.npz --model ../local/eeg-model-new.npz --assignments ../local/eeg-assignment.npz --output ../local/eeg-score-new.npz
python -B code/mic_cli.py summarize-eeg --qualification ../local/eeg-qualified.json --input ../local/eeg-all-contexts.npz --output ../local/eeg-inference-new.json
```

MEG recall arrays use physical labels; manipulation calibration uses heard H and
true evaluation uses intended 1-H. Random-label manipulation evaluation keeps the
same z. For both tasks use behavior-log `onset_log_sec`, not event sample indices,
for donor matching. EEG onsets instead use their frozen native `P_onset_sample`.
All assignments must retain original UID order and the 199 frozen draws.

Recall output evidence has shape `[200,N,2,3]`: B4/PCA16 and
include-own-P/exclude-own-P/full-N. Manipulation uses `[200,N,2,2]` with the two
inclusion arms. `analysis_adapters.score_meg_correspondence` accepts fixed
full-N predictions plus near/far within-stratum maps. Supply floating-point
behavior-log seconds to `meg_binary.mismatch`; its original solver uses infinity
on the diagonal.

EEG source fitting takes sorted source10/source11 people. Target scoring includes
all sessions of one person. Primary inference requires 12 main fractions and a
12 by 12 matrix with deleted-person rows, target-person columns and NaN diagonal.
Every off-diagonal value must result from the appropriate full source10 refit;
it must not be substituted with simple deletion of an already fitted result.

The CLI fixes numerical-library threads at two, but does not reproduce the old
host-specific affinity, memory watchdog or checkpoint/recovery manager. Large
source fits need independent resource supervision. They are not run by the tests.
Legacy unpickling helpers are not a security sandbox; use only trusted,
independently qualified original data, never arbitrary pickle files.

## Optional plots

No CSV or numerical result is included. With separately supplied group CSVs:

```text
python -B code/mic_cli.py group-plots --data-dir ../local/group-csv --output-dir ../local/group-figures-new
```

For external person/deletion/count plots, follow the local root layout and field
definitions in `INPUT_SCHEMAS.json`:

```text
python -B code/plot_external_locally_regenerated_rows.py --data-dir ../local/external-regenerated --output-dir ../local/external-figures-new
```

Those inputs require independent qualification and auditing. Neither group
summaries nor Monte Carlo rows can reconstruct missing individual outcomes.
