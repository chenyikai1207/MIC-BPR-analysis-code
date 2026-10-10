"""Portable MIC intervention CLI; requires independently qualified local inputs."""
from pathlib import Path
import argparse
import json
import os
import sys
import time


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('environment')
    sub.add_parser('synthetic-test', help='SYNTHETIC_TEST_ONLY; no real data/outputs')
    names = ['plan', 'four-cell', 'aggregate-factorial', 'source-sensitivity',
             'refit-source', 'direction-prior', 'direction-grid', 'select-direction', 'direction-score']
    for name in names:
        p = sub.add_parser(name)
        p.add_argument('--qualification', type=Path, required=True)
        p.add_argument('--input', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--budget-seconds', type=int, default=1800)
        if name == 'plan':
            p.add_argument('--codes', type=Path, required=True)
        if name in ['four-cell', 'direction-prior', 'direction-grid', 'direction-score']:
            p.add_argument('--model', type=Path, required=True)
        if name == 'four-cell':
            p.add_argument('--plan', type=Path, required=True)
            p.add_argument('--readouts', choices=['B4', 'B4,PCA16'], default='B4,PCA16')
        if name == 'refit-source':
            p.add_argument('--delete-person', required=True)
        if name in ['direction-grid', 'direction-score']:
            p.add_argument('--prior', type=Path, required=True)
        if name == 'direction-score':
            p.add_argument('--selection', type=Path, required=True)
            p.add_argument('--task', choices=['recall', 'reversal'], default='recall')
    args = parser.parse_args()
    from mic_cli import environment, external_path, load_arrays, save_arrays, qualified, features, validate_model
    if args.command == 'environment':
        print(json.dumps(environment('analysis'), indent=2))
        return
    if args.command == 'synthetic-test':
        import subprocess
        for name in ['test_interventions_synthetic.py', 'test_intervention_summaries_synthetic.py']:
            subprocess.run([sys.executable, '-B', str(Path(__file__).with_name(name))], check=True)
        return
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS']:
        os.environ[key] = '2'
    profile = environment('analysis')
    from mic_intervention_api import (require, strict_factorial_plan, score_four_cell, balanced_code_accuracies,
        factorial_person_endpoints, participant_bootstrap, source_target_sensitivity, paired_endpoint_difference,
        source_direction, direction_calibrators, select_source_lambda, LAMBDA_GRID, ARMS)
    import numpy as np
    from threadpoolctl import threadpool_limits
    from meg_binary import projection
    require(profile['python_matches'] and all(x['matches'] for x in profile['packages']), 'Use the frozen analysis profile')
    require(0 < args.budget_seconds <= 2700, 'Budget must be between 1 and 2700 seconds')
    start = time.monotonic()
    def budget():
        require(time.monotonic() - start <= args.budget_seconds, 'Budget exceeded: stop without changing model/support')
    q = qualified(args.qualification, 'MEG')
    scope = q.get('scope')
    require(scope in ['SYNTHETIC_TEST_ONLY', 'USER_SUPPLIED_ADMITTED_DATA'], 'Explicit data scope required')
    if scope != 'SYNTHETIC_TEST_ONLY':
        require(q.get('data_access_legal_and_license_verified') is True, 'Identity/license/legality must be independently verified')
    output = external_path(args.output)
    require(not output.exists(), 'Existing outputs are preserved; choose a new file')
    z = load_arrays(args.input)
    def write_json(result):
        budget()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as f:
            json.dump(jsonable({'data_scope': scope, **result}), f, indent=2, allow_nan=False)
            f.write('\n')
    def save(result):
        budget()
        save_arrays(output, {'scope': np.array(scope), **result})
    def order(key):
        a = z[key]
        require(a.ndim == 1 and a.dtype.kind in 'US' and len(set(a.tolist())) == len(a), 'Unique ordered person strings required')
        return a
    with threadpool_limits(2):
        if args.command == 'aggregate-factorial':
            people, ba, idx, readouts = order('person_order'), z['per_person_code_BA'], z['participant_row_resamples'], z['readout_order'].tolist()
            require(readouts in [['B4'], ['B4', 'PCA16']], 'Retain fixed readout names/order')
            require(ba.ndim == 4 and ba.shape[0] == len(people) and ba.shape[2:] == (len(readouts), 4), 'Person/code/readout/arm axes mismatch')
            if scope != 'SYNTHETIC_TEST_ONLY':
                require(ba.shape[:2] == (35, 199) and idx.shape == (9999, 35), 'Retain all35/199 and frozen9999 person draws')
            rows = [factorial_person_endpoints(person, readouts) for person in ba]
            result = {k: participant_bootstrap(np.array([row[k] for row in rows]), idx) for k in rows[0]}
            write_json({'endpoints': result, 'codes_averaged_within_person': ba.shape[1], 'equal_person_weighting': True})
            return
        if args.command == 'source-sensitivity':
            sources, targets = order('source_order'), order('target_order')
            require(not set(sources) & set(targets), 'Source/target person roles overlap')
            require(q.get('complete_source_refits_verified') is True and q.get('same_target_support_verified') is True, 'Verify complete refits and identical target support first')
            if 'full_endpoint_pairs_pp' in z:
                require(z['full_endpoint_pairs_pp'].shape == (len(targets), 2) and z['deleted_endpoint_pairs_pp'].shape == (len(sources), len(targets), 2), 'Paired endpoint axes mismatch')
                full = paired_endpoint_difference(z['full_endpoint_pairs_pp'][:, 0], z['full_endpoint_pairs_pp'][:, 1])
                deleted = paired_endpoint_difference(z['deleted_endpoint_pairs_pp'][:, :, 0], z['deleted_endpoint_pairs_pp'][:, :, 1])
            else:
                full, deleted = z['full_person_contrasts_pp'], z['deleted_source_person_contrasts_pp']
            require(full.shape == (len(targets),) and deleted.shape == (len(sources), len(targets)), 'Person order/deletion matrix mismatch')
            if scope != 'SYNTHETIC_TEST_ONLY':
                require(deleted.shape == (35, 35), 'Complete35source-by35target sensitivity required')
            write_json(source_target_sensitivity(full, deleted))
            return
        if args.command == 'select-direction':
            people = order('source_person_order')
            require(q.get('source_LOPO_exclusion_verified') is True, 'Verify every source LOPO map and direction excludes held person')
            require(z['source_person_BA'].shape == (len(people), 5), 'Source selection order mismatch')
            if scope != 'SYNTHETIC_TEST_ONLY':
                require(len(people) == 35, 'All35 source folds required')
            write_json({**select_source_lambda(z['source_person_BA'], z['lambda_grid']), 'source_people': len(people), 'selection_frozen': True})
            return
        features(z, 'MEG')
        if args.command == 'refit-source':
            from meg_source import fit_source_arrays
            owners = sorted(set(z['owner'].tolist()))
            require(len(owners) == 35 and args.delete_person in owners, 'Delete exactly one of the full35 source people')
            require(z['owner'].tolist() == [p for p in owners for _ in range(int(np.sum(z['owner'] == p)))], 'Source rows must remain grouped in frozen sorted person order')
            keep = z['owner'] != args.delete_person
            retained = [p for p in owners if p != args.delete_person]
            result = fit_source_arrays(z['P'][keep], z['I'][keep], z['y'][keep], retained, z['original_uid'][keep], budget)
            require(set(result['source_ids']) == set(retained), 'Source deletion identity mismatch')
            save({'modality': np.array('MEG'), 'deleted_source_person': np.array(args.delete_person), **result})
            return
        if args.command == 'plan':
            require(len(set(z['owner'])) == 1, 'One target person required')
            a = load_arrays(args.codes)
            require(np.array_equal(a['original_uid'], z['original_uid']), 'Frozen code UID/order mismatch')
            codes = a['labels']
            require(codes.shape == (199, len(z['y'])) if scope != 'SYNTHETIC_TEST_ONLY' else codes.ndim == 2, 'Exactly199 frozen arbitrary codebooks required')
            require(all(np.all((codes == c).sum(1) == np.sum(z['y'] == c)) for c in [0, 1]), 'Frozen code class counts changed')
            result = strict_factorial_plan(codes, z['y'], z['pause_halfblock'], z['onset_log_sec'], a.get('original_donor'))
            save({'labels': codes, 'original_uid': z['original_uid'], 'original_true_labels': z['y'],
                  'pause_halfblock': z['pause_halfblock'], 'onset_log_sec': z['onset_log_sec'], **result})
            return
        m = load_arrays(args.model)
        if args.command == 'direction-prior':
            owners = sorted(set(z['owner'].tolist()))
            require(np.array_equal(m['source_ids'], np.array(owners)) and np.array_equal(m['source_original_uids'], z['original_uid']), 'Source map/person/UID order mismatch')
            require(m['pca_components'].shape[0] == z['P'].shape[1], 'Source feature width mismatch')
            require(all(np.any((z['owner'] == p) & (z['y'] == c)) for p in owners for c in [0, 1]), 'Each source person needs both true P classes')
            means = np.stack([[z['P'][(z['owner'] == p) & (z['y'] == c)].mean(0, dtype=np.float64) for c in [0, 1]] for p in owners])
            model_path = external_path(args.model)
            stat = model_path.stat()
            save({'source_delta': source_direction(means, m), 'source_ids': m['source_ids'],
                  'source_original_uids': m['source_original_uids'], 'model_reference': np.array(str(model_path)),
                  'model_bytes': np.array(stat.st_size), 'model_mtime_ns': np.array(stat.st_mtime_ns)})
            return
        require(len(set(z['owner'])) == 1, 'One target/heldout-source person required')
        validate_model(m, z, 'MEG')
        pp, ii = projection(z['P'], m), projection(z['I'], m)
        xp, xi = (pp - m['p_mean']) @ m['B4_P_weights'], (ii - m['i_mean']) @ m['B4_I_weights']
        if args.command == 'four-cell':
            plan = load_arrays(args.plan)
            for a, b in [('original_uid', 'original_uid'), ('original_true_labels', 'y'), ('pause_halfblock', 'pause_halfblock'), ('onset_log_sec', 'onset_log_sec')]:
                require(np.array_equal(plan[a], z[b]), 'Frozen plan row/metadata mismatch')
            fresh = strict_factorial_plan(plan['labels'], z['y'], z['pause_halfblock'], z['onset_log_sec'])
            for key in ['k', 'l', 'eligible', 'reason_code']:
                require(np.array_equal(plan[key], fresh[key]), 'Plan violates strict matching')
            readouts = args.readouts.split(',')
            reps = [(xp, xi)] + ([(pp[:, :16], ii[:, :16])] if len(readouts) == 2 else [])
            ev = np.stack([score_four_cell(p, i, plan['labels'], plan, budget) for p, i in reps], axis=2)
            ba = balanced_code_accuracies(plan['labels'], plan['eligible'], ev)
            ep = factorial_person_endpoints(ba, readouts)
            save({'decision_evidence': ev, 'per_code_BA': ba, 'labels': plan['labels'], 'eligible': plan['eligible'],
                  'original_uid': z['original_uid'], 'arm_order': np.array(ARMS), 'readout_order': np.array(readouts),
                  'endpoint_names': np.array(list(ep)), 'endpoint_values': np.array(list(ep.values()))})
            return
        prior = load_arrays(args.prior)
        model_path = external_path(args.model)
        require(str(prior['model_reference']) == str(model_path) and int(prior['model_bytes']) == model_path.stat().st_size and int(prior['model_mtime_ns']) == model_path.stat().st_mtime_ns, 'Prior uses a different/changed source map')
        require(np.array_equal(prior['source_ids'], m['source_ids']) and np.array_equal(prior['source_original_uids'], m['source_original_uids']), 'Direction source binding mismatch')
        if args.command == 'direction-grid':
            require(q.get('source_LOPO_exclusion_verified') is True, 'This command is only for qualified source-heldout folds')
            lambdas, evaluation = LAMBDA_GRID, z['y']
        else:
            selection = json.loads(external_path(args.selection).read_text(encoding='utf-8'))
            require(selection.get('selection_frozen') is True and selection.get('target_scores_used') == 0 and np.array_equal(selection['lambda_grid'], LAMBDA_GRID), 'Require source-only selection frozen before target scores')
            require(selection.get('data_scope') == scope, 'Selection/data scope mismatch')
            chosen = selection['chosen_lambda']
            require(chosen in LAMBDA_GRID, 'Selected weight outside fixed grid')
            lambdas = [0., 1., chosen]
            evaluation = z['y'] if args.task == 'recall' else z['evaluation_y']
            require(np.array_equal(evaluation, z['y'] if args.task == 'recall' else 1 - z['y']), 'True imagery target definition mismatch')
        w, a, wd, ad = direction_calibrators(xp, z['y'], prior['source_delta'], lambdas)
        ev = np.einsum('nld,nd->nl', w, xi) + a
        if args.command == 'direction-grid':
            means = balanced_code_accuracies(evaluation[None, :], np.ones((1, len(evaluation)), bool), ev[None, :, :])[0]
            save({'source_fold_BA': means, 'lambda_grid': LAMBDA_GRID, 'original_uid': z['original_uid'], 'labels': evaluation, 'decision_evidence': ev})
        else:
            diagonal = np.einsum('nd,nd->n', wd, xi) + ad
            evidence = np.column_stack([ev[:, 0], diagonal, ev[:, 1], ev[:, 2]])
            ba = balanced_code_accuracies(evaluation[None, :], np.ones((1, len(evaluation)), bool), evidence[None, :, :])[0]
            save({'decision_evidence': evidence, 'method_BA': ba, 'evaluation_true_labels': evaluation,
                  'training_true_labels': z['y'], 'original_uid': z['original_uid'], 'chosen_lambda': np.array(chosen),
                  'method_order': np.array(['ORIGINAL_LDA', 'DIAGONAL_LDA', 'PURE_SOURCE_DIRECTION', 'SOURCE_ANCHORED_LDA'])})
        budget()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, AssertionError, OSError) as error:
        sys.exit(f'Input/contract check failed: {error}')
