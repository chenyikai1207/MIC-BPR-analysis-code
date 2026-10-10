"""Local qualified-array CLI. No discovery, download, or experiment on import."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def environment(profile):
    rows = []
    for line in (ROOT / f'requirements-{profile}.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        rows.append({'package': name, 'expected': expected, 'installed': actual,
                     'matches': actual == expected})
    return {'profile': profile, 'python': sys.version.split()[0],
            'python_matches': sys.version_info[:2] == (3, 12), 'packages': rows}


def external_path(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        raise ValueError('Keep locally regenerated arrays/rows outside the candidate packet')
    return path


def load_arrays(path):
    import numpy as np
    with np.load(external_path(path), allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def qualified(path, modality):
    q = json.loads(external_path(path).read_text(encoding='utf-8'))
    require(q.get('status') == 'QUALIFIED_LOCALLY', 'Qualification example is not completed')
    require(q.get('modality') == modality, 'Qualification modality mismatch')
    for key in ('snapshot_identity_verified', 'eligibility_verified',
                'event_uid_row_order_verified', 'feature_contract_verified',
                'frozen_roles_verified'):
        require(q.get(key) is True, f'Missing upstream qualification: {key}')
    require(bool(q.get('local_qualification_evidence')), 'Record upstream qualification evidence')
    return q


def features(z, modality):
    import numpy as np
    width = 24480 if modality == 'MEG' else 9920
    for key in ('P', 'I', 'y', 'original_uid', 'owner'):
        require(key in z, f'Missing feature key: {key}')
    p, i, y = z['P'], z['I'], z['y']
    require(p.ndim == 2 and p.shape == i.shape and p.shape[1] == width,
            f'{modality}: expected aligned N by {width} P/I arrays')
    n = len(p)
    require(n > 0 and np.isfinite(p).all() and np.isfinite(i).all(), 'Nonfinite/empty features')
    for key in ('y', 'original_uid', 'owner'):
        require(z[key].shape == (n,), f'{key} must follow original row order')
    require(z['original_uid'].dtype.kind in 'US' and z['owner'].dtype.kind in 'US',
            'UID and owner must be non-object string arrays')
    require(len(set(z['original_uid'])) == n, 'Duplicate original UID')
    require(y.dtype.kind in 'iu' and set(y.tolist()) == set(range(2 if modality == 'MEG' else 3)),
            'Wrong true category encoding')
    if modality == 'EEG':
        for key in ('session', 'pitch', 'P_onset_sample'):
            require(key in z and z[key].shape == (n,), f'Missing EEG metadata: {key}')
        require(z['session'].dtype.kind in 'US' and z['pitch'].dtype.kind in 'US',
                'Session and pitch must be frozen non-object strings')
        require(z['P_onset_sample'].dtype.kind in 'iu', 'Onsets must be native sample indices')
    return n


def validate_model(m, z, modality):
    import numpy as np
    for key in ('scaler_mean', 'scaler_scale', 'pca_mean', 'pca_components',
                'p_mean', 'i_mean', 'B4_P_weights', 'B4_I_weights',
                'source_ids', 'source_original_uids', 'modality'):
        require(key in m, f'Missing source model field: {key}')
    require(str(m['modality']) == modality, 'Source model modality mismatch')
    require(not set(m['source_ids']) & set(z['owner']), 'Source and target people overlap')
    require(not set(m['source_original_uids']) & set(z['original_uid']), 'Source/target UIDs overlap')
    require(m['pca_components'].shape[0] == z['P'].shape[1], 'Feature width/model mismatch')
    require(m['B4_P_weights'].shape[1] == m['B4_I_weights'].shape[1] == 16,
            'Keep fixed 16-dimensional representations')
    require(np.isfinite(m['scaler_scale']).all() and np.all(m['scaler_scale'] > 0), 'Invalid scaler')


def validate_assignment(a, z, modality):
    import numpy as np
    n = len(z['y'])
    for key in ('labels', 'donor', 'original_uid'):
        require(key in a, f'Missing assignment: {key}')
    labels, donor = a['labels'], a['donor']
    require(labels.shape == donor.shape == (200, n), 'Exactly true + 199 frozen draws required')
    require(np.array_equal(a['original_uid'], z['original_uid']), 'Assignment UID/order mismatch')
    require(np.array_equal(labels[0], z['y']), 'First row must contain original true labels')
    classes = range(2 if modality == 'MEG' else 3)
    require(labels.dtype.kind in 'iu' and np.isin(labels, list(classes)).all(), 'Invalid labels')
    require(donor.dtype.kind in 'iu' and np.all((donor >= 0) & (donor < n)), 'Invalid donor index')
    require(np.all(donor != np.arange(n)[None, :]), 'Self donor prohibited')
    require(np.array_equal(labels, np.take_along_axis(labels, donor, axis=1)), 'Donor code differs')
    for c in classes:
        require(np.all((labels == c).sum(1) >= 3), 'Every deletion must retain two per class')
    if modality == 'MEG':
        for c in classes:
            require(np.all((labels == c).sum(1) == (labels[0] == c).sum()),
                    'Frozen MEG label multiset changed')
        for key in ('pause_halfblock', 'onset_log_sec'):
            require(key in z and z[key].shape == (n,), f'Qualified MEG timing missing: {key}')
        require(np.isin(z['pause_halfblock'], [0, 1]).all(), 'Pause half must remain binary')
        require(np.isfinite(z['onset_log_sec']).all(), 'Log onset seconds must be finite')
        from meg_binary import donor_map
        expected, _ = donor_map(labels, z['y'], z['pause_halfblock'], z['onset_log_sec'])
        require(np.array_equal(donor, expected), 'Donor differs from frozen deterministic rule')
    else:
        from external_codes import draw_proposed_codes
        from external_core import donor_map
        owner = str(z['owner'][0])
        require(owner.startswith('sub-') and owner[4:].isdigit(), 'Use published numeric EEG ID')
        expected_codes = draw_proposed_codes(z['y'], z['session'], int(owner[4:]))
        require(np.array_equal(labels[1:], expected_codes), 'Frozen EEG seed/order/allocation mismatch')
        expected, tier = donor_map(labels, z['y'], z['session'], z['pitch'], z['P_onset_sample'])
        require(np.array_equal(donor, expected) and np.all(tier <= 1), 'Frozen EEG donor mismatch')
    return labels, donor


def save_arrays(path, arrays):
    import numpy as np
    path = external_path(path)
    require(not path.exists(), 'Choose a new output file; existing work is preserved')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)


def compute(args):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '2'
    check = environment('analysis')
    require(check['python_matches'] and all(p['matches'] for p in check['packages']),
            'Use the frozen analysis profile; run environment --profile analysis')
    import numpy as np
    from threadpoolctl import threadpool_limits
    modality = args.modality
    qualified(args.qualification, modality)
    output = None if args.command == 'validate-input' else external_path(args.output)
    if output is not None:
        require(not output.exists(), 'Choose a new output')
    z = load_arrays(args.input)
    with threadpool_limits(2):
        if args.command == 'summarize-eeg':
            require(modality == 'EEG', 'summarize-eeg requires EEG qualification')
            from external_statistics import summarize_lopo
            require('person_order' in z and z['person_order'].shape == (12,), '12 frozen person IDs required')
            require(z['person_order'].tolist() == sorted(set(z['person_order'].tolist())), 'Retain sorted person order')
            result = summarize_lopo(z['primary_fraction'], z['outer_delete_fraction'])
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x', encoding='utf-8') as f:
                json.dump(result, f, indent=2, allow_nan=False)
            return
        features(z, modality)
        if args.command == 'validate-input':
            print(json.dumps({'status': 'ARRAY_CONTRACT_PASS_ONLY', 'modality': modality,
                              'rows': len(z['y']), 'upstream_qualification_independently_verified': False}))
            return
        if args.command == 'fit-source':
            owners = list(dict.fromkeys(z['owner'].tolist()))
            if modality == 'EEG':
                require(len(owners) in (10, 11) and owners == sorted(owners), 'EEG source10/11 sorted people required')
            else:
                require(len(owners) in (28, 34, 35), 'MEG complete35/delete1/delete7 source count required')
            require(z['owner'].tolist() == [p for p in owners for _ in range(int(np.sum(z['owner'] == p)))],
                    'Source rows must stay grouped in frozen person order')
            if modality == 'MEG':
                from meg_source import fit_source_arrays
                result = fit_source_arrays(z['P'], z['I'], z['y'], owners, z['original_uid'])
            else:
                from external_core import fit_source
                result = fit_source(z['P'], z['I'], z['y'])
                result.update(source_ids=np.array(owners), source_original_uids=z['original_uid'])
            result['modality'] = np.array(modality)
            save_arrays(output, result)
            return
        require(len(set(z['owner'])) == 1, 'Score one target person with all frozen sessions')
        m = load_arrays(args.model)
        validate_model(m, z, modality)
        a = load_arrays(args.assignments)
        labels, donor = validate_assignment(a, z, modality)
        if modality == 'EEG':
            from external_core import score_target, balanced_session_means
            result, pred = score_target(m, z, {'labels': labels, 'donor': donor}, return_predictions=True)
            ba = balanced_session_means(labels, pred, z['session'])
            result.update(original_uid=z['original_uid'], predictions=pred,
                          arbitrary_mean_arm_macroBA_fraction=ba[1:].mean(0), real_arm_macroBA_fraction=ba[0])
            result = {k: np.asarray(v) for k, v in result.items()}
        elif args.task == 'recall':
            from analysis_adapters import score_meg_recall
            result = score_meg_recall(m, z['P'], z['I'], labels, donor)
        else:
            from analysis_adapters import score_meg_manipulation
            require('evaluation_labels' in a, 'Manipulation requires explicit frozen evaluation labels')
            result = score_meg_manipulation(m, z['P'], z['I'], labels, a['evaluation_labels'], donor)
        result.update(original_uid=z['original_uid'], labels=labels)
        save_arrays(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('environment', help='Report exact installed profile versions, no model imports')
    p.add_argument('--profile', choices=['analysis', 'plotting'], required=True)
    sub.add_parser('describe', help='Print contracts and frozen parameters, without opening data')
    sub.add_parser('synthetic-test', help='Run bounded synthetic-only kernel checks')
    p = sub.add_parser('group-plots', help='Rebuild Figures 1/2/S1/S2 from separately supplied group CSVs')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True, help='Local group CSV directory; no data are bundled')
    for name in ('validate-input', 'fit-source', 'score-target', 'summarize-eeg'):
        p = sub.add_parser(name, help='Explicit qualified local input required; no raw discovery')
        p.add_argument('--modality', choices=['MEG', 'EEG'], required=name != 'summarize-eeg', default='EEG')
        p.add_argument('--qualification', type=Path, required=True)
        p.add_argument('--input', type=Path, required=True)
        p.add_argument('--output', type=Path, required=name != 'validate-input', default=Path('unused-validation-output'))
        if name == 'score-target':
            p.add_argument('--model', type=Path, required=True)
            p.add_argument('--assignments', type=Path, required=True)
            p.add_argument('--task', choices=['recall', 'manipulation'], default='recall', help='MEG only')
    args = parser.parse_args()
    if args.command == 'environment':
        report = environment(args.profile)
        print(json.dumps(report, indent=2))
        return 0 if report['python_matches'] and all(p['matches'] for p in report['packages']) else 2
    if args.command == 'describe':
        print(json.dumps({name: json.loads((ROOT/name).read_text(encoding='utf-8'))
                          for name in ('INPUT_SCHEMAS.json', 'FROZEN_PARAMETERS_AND_SEEDS.json')}, indent=2))
    elif args.command == 'synthetic-test':
        for name in ('test_synthetic_consistency.py', 'test_release_interfaces.py'):
            subprocess.run([sys.executable, '-B', str(ROOT/'code'/name)], check=True)
    elif args.command == 'group-plots':
        dest = external_path(args.output_dir)
        data = external_path(args.data_dir)
        require(data.is_dir(), 'Supply an existing local group CSV directory')
        require(not dest.exists(), 'Choose a new output directory')
        for name, folder in [('plot_current_meg.py', 'main'), ('plot_current_supplement_group.py', 'supplement')]:
            subprocess.run([sys.executable, '-B', str(ROOT/'code'/name), '--data-dir', str(data), '--output-dir', str(dest/folder)], check=True)
    else:
        compute(args)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, AssertionError) as error:
        sys.exit(f'Input/contract check failed: {error}')
