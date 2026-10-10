# Scope and historical workflow gaps

This release prioritizes the core scientific functions. The original work used
separate executed raw qualification, frozen manifests, processing caches and
batch drivers. Their absence from this repository describes a packaging boundary,
not evidence that the original experiments were never run.

The following original producer locations were identified during curation.
Paths below identify scripts in the historical author project; they are not
paths or data instances bundled in this repository.

| Step | Identified historical code | Release boundary |
|---|---|---|
| MEG feature/UID materialization | `materialize_verified_phases.py`, `main()` in `source_correspondence_E2_v1/experiment` | Core windows and epoch reader included; original archive coverage, identity and cache bindings must be independently regenerated |
| MEG recall behavior/time join | `target_pairing_structure_E4_v1/audit_original_time.py` | Correct-response behavior rows, filtered epoch selection, labels and timing are cross-checked; original log inputs and full joining driver not packaged |
| MEG manipulation clock qualification | `manipulation_calibration_input_replication_v1/audit_affine_clock_alignment.py`; `freeze_materialize_manipulation_resumed.py`, `main()` | Existing affine metadata gate and original-log donor times identified; no new clock correction or relaxed criterion introduced |
| Frozen MEG roles and assignments | `source_correspondence_E2_v1/experiment/freeze_E2.py`; `target_label_specificity_E3_v1/freeze_E3.py`; `target_calibration_input_Nminus1_v1/prescore_freeze.py` | Sorted alternating cohort rule and assignment streams documented; original role/UID instances excluded |
| EEG strict adjacent P/I pairs | `EXTERNAL_ALL12_TECHNICAL_QUALIFICATION_20261008/metadata_preflight.py`, `normevent()` and `main()` | Original event normalization and native/BIDS event/channel checks located; download/version records and paired instances excluded |
| EEG window qualification/features | `EXTERNAL_SNAPSHOT_FORMAL_RUN_20261008/technical_feature_stage.py`, `_process_available()` | Own-window feature kernel included; raw SET/FDT, sidecars and technical checkpoint orchestration are not bundled |
| Approved EEG allocation and full LOPO | `prepare_approved_probe.py`, `load_payload()`/`main()`; `run_approved_balanced_probe.py`, `fit_people()`/`main()` | Approved allocation and scoring/inference kernels included; source scheduling, checkpoints and final table exporters remain separate |
| Current source deletion/group summaries | `source_jack_resume_core.py`, `fit_source()`/`score()`; `audit_and_summarize_source_deletions.py`; `run_nminus1.py`; `run_manipulation_input.py`; `run_E4.py` | Core fits/scores and five-block inference included; full cohort orchestration/bootstrap/publication export not packaged |
| Historical D1/D6/listener controls | `d1_formal_core.py`, `load_person()`/`load_all()`; `d6_core.py`, `load()`; `matched_listener_origin.py`, `data()`/`qualify()` | Scientific control functions included; original cache contracts and orchestration excluded |
| Historical 69-source width/calibration route | `scripts/mic_next_4/run_loso.py`, `run()`; `scripts/mic_supplemental/shared_runtime.py`, `frozen_dimensions()`/`build_fold_cache()`; `strict_loso_ablation.py`, `run_reproduction()` | Located historical code relies on frozen outer-fold widths and other project modules; that historical execution graph is outside this core release |

Three different limitations should remain distinct:

1. **Located but not portable yet:** many old scripts bind project paths, archived
   checkpoints and execution records. They require careful packaging, not a new
   scientific analysis or an invented algorithm.
2. **Local inputs deliberately excluded:** original signals, behavior logs,
   person/session records, event/UID manifests, assignments, model caches and
   outcomes are not code. Obtain data from the named providers and regenerate
   qualified local inputs. No instance records are published here.
3. **Historical identity evidence not fully recovered:** the MEG reconstruction
   explicitly did not recover an older exact UID-row serialization/software-run
   binding; equal label order alone cannot certify historical signal-row identity.
   For EEG, exact equivalence of the acquired object-version snapshot to the named
   release tag was not certified. Do not manufacture either equivalence.

The scalar/core tests validate small synthetic calculations and structural
contracts. They do not exercise complete native-file qualification, full cohort
processing, all historical drivers, or an end-to-end real-data manuscript rerun.
No claim of complete one-command reproduction is made.
