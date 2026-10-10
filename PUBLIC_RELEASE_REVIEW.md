# Code-only release review

The intervention extension contains source, documentation, input schema and
synthetic tests. It adds no dataset, participant/trial record, assignment instance,
fitted model, outcome table, manuscript, figure or private execution history.
No new code license is selected; `LICENSE_STATUS.md` and third-party terms remain.

Before publishing, check these conditions locally:

- The remote is the explicitly authorized public MIC core-code repository and
  the account has write permission; the original private repository is untouched.
- Only explicit source/document/schema paths are staged. No local QA, arrays,
  numerical group results, papers, rendered images or credentials are included.
- All existing scientific module and dependency bytes remain unchanged, apart
  from documentation pointing to the added extension.
- Frozen numerical versions match; all synthetic checks pass, including direct
  LDA oracles, original scoring-label invariance, donor/support/count constraints,
  paired whole-person aggregation, complete-refit arithmetic and direction rules.
- Local saved-evidence reaggregation is checked separately and is described as
  arithmetic verification, with no new real fitting, prediction or random draw.
- Public documentation describes actual provider-level inputs and the remaining
  native qualification, historical binding and cohort/report orchestration gaps.
- The published tree is retrieved independently after push and its file bytes
  match the reviewed candidate. No digest values are computed or recorded.

Test success must not be represented as an untouched test, full raw-data rerun,
public derived-data deposit or complete one-command manuscript reproduction.
Participant and trial outputs remain local even when their source data are public.
