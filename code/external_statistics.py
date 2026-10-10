"""Exact whole-person delete-and-refit approximate inference."""
import numpy as np
from scipy.stats import t

def summarize_lopo(primary_person_fraction,outer_delete_person_fraction):
 d=np.asarray(primary_person_fraction,np.float64);outer=np.asarray(outer_delete_person_fraction,np.float64)
 assert d.shape==(12,) and outer.shape==(12,12) and np.isfinite(d).all()
 assert np.isnan(np.diag(outer)).all()
 mask=~np.eye(12,dtype=bool);assert np.isfinite(outer[mask]).all()
 theta=float(d.mean());deleted=np.array([outer[k,mask[k]].mean() for k in range(12)])
 variance=11/12*np.sum((deleted-deleted.mean())**2);se=float(np.sqrt(variance));half=float(t.ppf(.975,11)*se)
 pseudo=12*theta-11*deleted;bias_corrected=float(pseudo.mean())
 return {'primary_mean_pp':theta*100,'primary_person_values_pp':(d*100).tolist(),
  'whole_participant_delete_refit_estimates_pp':(deleted*100).tolist(),
  'jackknife_SE_pp':se*100,'approximate_participant_jackknife_95_interval_pp':([100*(theta-half),100*(theta+half)] if se>0 else None),
  'interval_reference_df':11,'jackknife_pseudovalues_pp':(pseudo*100).tolist(),
  'jackknife_bias_corrected_mean_secondary_pp':bias_corrected*100,
  'target_person_clusters':12,'independent_source_training_sets_claimed':False,
  'ordinary_iid_fold_t_test_used':False,'confirmatory_p_value':None,
  'inference_scope':'Whole participant deletion reruns all affected source and target fits. This accounts computationally for training overlap; t11 jackknife interval is small-sample/stability-dependent approximation, not exact coverage or consciousness/neural specificity.'}
