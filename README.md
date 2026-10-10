# MIC BPR core analysis code

Core analysis code for **Paired perception calibration contributes to musical
imagery decoding**.

This repository contains scientific computation functions, explicit-array command
line interfaces, input schemas, dependency requirements and small synthetic tests.
It contains **no datasets, numerical result tables, participant records,
manuscripts, figures or model/feature/prediction arrays**. Plotting scripts require
separately supplied local inputs.

The release provides core analysis source. It does not provide a fully automated
raw-data-to-manuscript reproduction. Original experiments had their own executed
data qualification and orchestration scripts; packaging those historical drivers
is a separate task. See [Scope and remaining workflow gaps](SCOPE_AND_GAPS.md).

## Run the small checks

Use Python 3.12.14 and the frozen analysis environment in
`requirements-analysis.txt`. From the repository root:

```text
python -B code/mic_cli.py environment --profile analysis
python -B code/mic_cli.py describe
python -B code/mic_cli.py synthetic-test
```

The tests generate their own `SYNTHETIC_TEST_ONLY` arrays in memory. They check
source maps, exact shrinkage-LDA readouts against scikit-learn, calibration arm
direction, correspondence scoring, jackknife arithmetic and rejected malformed
inputs. They do not load or download real participant data. Tested dependency
versions are described in [RUNNING.md](RUNNING.md).

## What is included

| Modules | Computation |
|---|---|
| `meg_source.py`, `meg_binary.py` | Source scaler/PCA/B4 maps, binary LDA, deterministic donors, correspondence and five-block inference |
| `external_core.py`, `external_codes.py`, `external_statistics.py` | EEG phase-local features, source maps, approved balanced target codes, exact LDA and whole-person inference |
| `analysis_adapters.py`, `mic_cli.py` | Explicit-array source/target interfaces; recovered recall/manipulation scoring blocks |
| `historical_d1.py`, `historical_d6.py`, `historical_listener_origin.py` | Historical spatial-PCA/control functions, distinct from the current source-PCA95 route |
| `feature_windows.py`, `meg_epoch_reader.py`, `native_mat5_metadata.py` | Feature-window and legacy-format helpers; not a full data qualification pipeline |
| `plot_*.py` | Figure scripts requiring separately obtained group or locally regenerated row inputs |

The original 51 extracted function/class bodies retain the historical producer
text, and the existing MEG source numerical block is retained. Array adapters
separate computation from local paths. No estimator, cross-validation rule,
scientific seed or participant-selection rule was changed for publication.

`FROZEN_PARAMETERS_AND_SEEDS.json` contains analysis settings, not results.
The task-specific assignment seeds are **20261019 for MEG recall** and
**20261029 for MEG manipulation**. Both use a single frozen target/draw stream.
MEG donors use original **behavior-log onset seconds**. EEG uses
`SeedSequence([20261032, numeric public participant ID])` with the approved
person/session/true-category balancing rule. These fields are documented from
the actual producer scripts; do not reset MEG generators independently per person.

## Data and inputs

Obtain source data independently from their providers:

- MEG derivatives: [Zenodo record](https://doi.org/10.5281/zenodo.13760720).
  Original-study code: [Zenodo code record](https://doi.org/10.5281/zenodo.13760787).
  [Source article](https://doi.org/10.1371/journal.pbio.3002858).
- Speech EEG: [OpenNeuro ds004306](https://openneuro.org/datasets/ds004306).
  The historical analysis used immutable object versions acquired on 8 October
  2026. Equivalence to tagged v1.0.2 was not certified. The published directory
  13/native basename 12 discrepancy must be explicitly qualified, not silently
  relabeled. The exact acquisition/version manifest is not included here.

Keep all inputs and generated outputs outside this repository. Follow the array
contracts in `INPUT_SCHEMAS.json` and copy `QUALIFICATION.example.json` to a local
work directory. The example deliberately fails qualification until completed.
The CLI's structural checks do not independently certify source identity,
eligibility, original event/UID order or historical role membership.

## Validation and license scope

The release was checked statically and with small synthetic inputs, including
19 direct scikit-learn comparisons. Its mathematical kernels were compared with
the historical producers. No new complete real-data run validates the packaged
interfaces. Test success does not establish reproduction of all paper results.

No code license has yet been selected; public source access is not a claim that
an open-source license has been granted. See [LICENSE_STATUS.md](LICENSE_STATUS.md)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Source-data access and any
supporting-data deposit are separate from this code repository.
