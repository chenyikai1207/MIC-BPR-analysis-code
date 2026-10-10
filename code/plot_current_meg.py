"""Replot unchanged aggregate results; no source or target models are run."""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

import argparse
ap=argparse.ArgumentParser(description='Current Figures1/2 from group CSVs; no refits')
ap.add_argument('--data-dir',type=Path,required=True)
ap.add_argument('--output-dir',type=Path,required=True)
args=ap.parse_args()
assert not args.output_dir.exists(), 'Choose a new output directory'
args.output_dir.mkdir(parents=True)
D=Path(__file__).resolve().parents[1]
TEAL='#00A89D';MAGENTA='#ED388A';BLUE='#347BDA';TEXT='#263244'
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,
 'svg.fonttype':'none','pdf.fonttype':42,'text.color':TEXT,'axes.labelcolor':TEXT,
 'xtick.color':TEXT,'ytick.color':TEXT,'axes.edgecolor':TEXT,'axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'savefig.facecolor':'white'})
def read(name):
    with (args.data_dir/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def keyed(name):return {r['endpoint']:r for r in read(name)}
R=keyed('Figure3_recall_input_contrasts.csv');M=keyed('Figure3_manipulation_input_contrasts.csv')
S=keyed('Figure3_source_sensitivity.csv')
keys=['REAL_B4_INCLUDE_MINUS_EXCLUDE','REAL_PCA16_INCLUDE_MINUS_EXCLUDE',
      'REAL_B4_MINUS_PCA_DIFFERENCE_OF_DIFFERENCES','RANDOM_B4_INCLUDE_MINUS_EXCLUDE',
      'RANDOM_PCA16_INCLUDE_MINUS_EXCLUDE','RANDOM_B4_MINUS_PCA_DIFFERENCE_OF_DIFFERENCES']
colors=[TEAL,MAGENTA,BLUE,TEAL,MAGENTA,BLUE]
def label(ax,letter,title):
    ax.text(-.11,1.07,letter,transform=ax.transAxes,weight='bold',fontsize=12)
    ax.set_title(title,loc='left',pad=9)
def grid(ax):
    ax.axvline(0,color=TEXT,lw=.7,alpha=.65);ax.grid(axis='x',color='#E7ECF2',lw=.6);ax.set_axisbelow(True)
def forest(ax,data,lo,hi,labels):
    for i,(k,c) in enumerate(zip(keys,colors)):
        r=data[k];mean=100*float(r['mean_BA_difference']);a=100*float(r[lo]);b=100*float(r[hi]);y=5-i
        ax.errorbar(mean,y,xerr=[[mean-a],[b-mean]],fmt='o',color=c,ms=4,capsize=2,lw=1.2)
    ax.set_yticks(range(6),list(reversed(labels)));ax.set_ylim(-.55,5.55);grid(ax)
    ax.set_xlabel('Own-perception inclusion effect (pp)')

fig,axs=plt.subplots(2,2,figsize=(7.0,5.6),gridspec_kw={'wspace':.77,'hspace':.69})
a,b,c,d=axs.flat
a.set_axis_off();label(a,'a','Exchange one calibration observation')
def box(x,y,w,h,text,color,fill):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',lw=.9,
                              edgecolor=color,facecolor=fill,transform=a.transAxes))
    a.text(x+w/2,y+h/2,text,ha='center',va='center',transform=a.transAxes,fontsize=8)
box(.10,.71,.82,.22,'Score the same imagery phase Ij',BLUE,'#F4F7FB')
box(.02,.32,.42,.23,'Include own Pj\nRemove Pk',TEAL,'#DDF4F1')
box(.57,.32,.40,.23,'Exclude own Pj\nRetain Pk',MAGENTA,'#FBDDEB')
for x in [.23,.79]:a.annotate('',xy=(x,.70),xytext=(x,.57),xycoords='axes fraction',
                              arrowprops={'arrowstyle':'->','color':BLUE,'lw':1.1})
a.text(.51,.10,'Both: N − 1 perception observations\nSame true melody and supervision counts',
       transform=a.transAxes,ha='center',va='center',fontsize=8)
label(b,'b','Recall')
forest(b,R,'target_cluster_bootstrap_ci95_low','target_cluster_bootstrap_ci95_high',
       ['True B4','True PCA16','True\nB4 − PCA16','Arbitrary B4','Arbitrary PCA16','Arbitrary\nB4 − PCA16'])
b.set_xlim(-1.7,3.25)
label(c,'c','Reversal with the recall-trained maps')
forest(c,M,'conditional_target_cluster95low','conditional_target_cluster95high',
       ['Goal B4','Goal PCA16','Goal\nB4 − PCA16','Arbitrary B4','Arbitrary PCA16','Arbitrary\nB4 − PCA16'])
c.set_xlim(-4.2,3.45)
label(d,'d','Recall after complete source refits')
for i,(k,col) in enumerate(zip(keys,colors)):
    r=S[k];mean=100*float(r['full35_source_estimate']);lo=100*float(r['source_plus_target_approx95low']);hi=100*float(r['source_plus_target_approx95high']);y=5-i
    d.plot([lo,hi],[y,y],color=col,alpha=.30,lw=4.5,solid_capstyle='round')
    z=R[k];d.plot([100*float(z['target_cluster_bootstrap_ci95_low']),100*float(z['target_cluster_bootstrap_ci95_high'])],[y,y],color=col,lw=1.1)
    d.plot(mean,y,'o',ms=4,color=col)
d.set_yticks(range(6),list(reversed(['True B4','True PCA16','True\nB4 − PCA16','Arbitrary B4','Arbitrary PCA16','Arbitrary\nB4 − PCA16'])))
d.set_ylim(-.55,5.55);d.set_xlim(-9,9);grid(d);d.set_xlabel('Own-perception inclusion effect (pp)')
d.legend([Line2D([0],[0],color=TEXT,lw=1.1),Line2D([0],[0],color=TEXT,lw=4.5,alpha=.30)],
         ['Source fixed','Source + target approximate'],loc='lower left',bbox_to_anchor=(-.04,-.49),fontsize=7.2)
fig.subplots_adjust(left=.15,right=.99,top=.91,bottom=.15)
for ext in ['svg','pdf','png']:fig.savefig(args.output_dir/f'Figure_1_Trial_Calibration.{ext}',dpi=400)
plt.close(fig)

F=read('Figure2_full_calibration.csv');L=keyed('Figure2_label_specificity.csv')
W=read('Figure2_within_stratum.csv');C=keyed('Figure2_correspondence_contrasts.csv')
fig,axs=plt.subplots(2,2,figsize=(7.0,5.5),gridspec_kw={'wspace':.70,'hspace':.66})
a,b,c,d=axs.flat;label(a,'a','Scores follow the calibration labels')
modes=['REAL_TRUE_EVAL','RANDOM_CAL_COUPLED_NULL_EVAL','RANDOM_CAL_FIXED_TRUE_EVAL']
for i,mode in enumerate(modes):
    for ro,col,off in [('B4',TEAL,-.085),('PCA16',MAGENTA,.085)]:
        r=next(z for z in F if z['evaluation_mode']==mode and z['readout']==ro)
        mean=100*float(r['mean_BA']);lo=100*float(r['conditional_ci95_low']);hi=100*float(r['conditional_ci95_high'])
        a.errorbar(i+off,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=col,ms=4,capsize=2,lw=1.1)
a.set_xticks(range(3),['True melody','Arbitrary code','Random cal\ntrue evaluation']);a.set_ylim(48,57)
a.axhline(50,color=TEXT,ls='--',lw=.7,alpha=.6);a.set_ylabel('Balanced accuracy (%)')
a.legend([Line2D([0],[0],color=TEAL,marker='o',ls=''),Line2D([0],[0],color=MAGENTA,marker='o',ls='')],['B4','PCA16'],fontsize=7.5,loc='upper right')
label(b,'b','Direct contrast of label increments')
for i,(k,t) in enumerate([('REAL_INCREMENT_B4_MINUS_PCA16','True melody'),('NULL_COUPLED_INCREMENT','Arbitrary code'),('PRIMARY_COUPLED_SPECIFICITY','True − arbitrary')]):
    r=L[k];mean=100*float(r['mean_BA']);lo=100*float(r['conditional_ci95_low']);hi=100*float(r['conditional_ci95_high'])
    b.errorbar(mean,2-i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=BLUE,ms=4,capsize=2,lw=1.2)
b.set_yticks(range(3),['True − arbitrary','Arbitrary code','True melody']);b.set_ylim(-.5,2.5);b.set_xlim(-1.1,7.2);grid(b);b.set_xlabel('B4 − PCA16 increment (pp)')
label(c,'c','Reassign within melody and pause strata')
# Source column names are asserted, never guessed or recomputed from raw inputs.
assert {'readout','mode','mean_BA','conditional_ci95_low','conditional_ci95_high'} <= set(W[0])
for ro,col in [('B4',TEAL),('PCA16',MAGENTA)]:
    rr=[next(z for z in W if z['readout']==ro and z['mode']==co) for co in ['CORRECT_SYNCHRONIZED_NULL','NEAR_CLASS_HALF_MISMATCH_NULL','FAR_CLASS_HALF_MISMATCH_NULL']]
    means=np.array([100*float(x['mean_BA']) for x in rr]);los=np.array([100*float(x['conditional_ci95_low']) for x in rr]);his=np.array([100*float(x['conditional_ci95_high']) for x in rr])
    c.errorbar(range(3),means,yerr=[means-los,his-means],color=col,marker='o',ms=4,capsize=2,lw=1.2)
c.set_xticks(range(3),['Correct','Near','Far']);c.set_ylim(48.6,53.1);c.set_ylabel('Arbitrary-code accuracy (%)');c.axhline(50,color=TEXT,ls='--',lw=.7,alpha=.6)
label(d,'d','Correspondence contrasts for B4')
for i,k in enumerate(['NULL_B4_CORRECT_MINUS_NEAR','NULL_B4_CORRECT_MINUS_FAR','NULL_B4_NEAR_MINUS_FAR']):
    r=C[k];mean=100*float(r['mean_BA']);lo=100*float(r['conditional_ci95_low']);hi=100*float(r['conditional_ci95_high'])
    d.errorbar(mean,2-i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=TEAL,ms=4,capsize=2,lw=1.2)
d.set_yticks(range(3),['Near − far','Correct − far','Correct − near']);d.set_ylim(-.5,2.5);d.set_xlim(-1.05,3.0);grid(d);d.set_xlabel('Balanced-accuracy difference (pp)')
fig.subplots_adjust(left=.15,right=.99,top=.91,bottom=.13)
for ext in ['svg','pdf','png']:fig.savefig(args.output_dir/f'Figure_2_Label_Correspondence.{ext}',dpi=400)
plt.close(fig)
(args.output_dir/'FIGURE_STORY_SOURCE_RECEIPT.json').write_text(json.dumps({'figures':['Figure_1_Trial_Calibration','Figure_2_Label_Correspondence'],'model_or_score_recomputed':False,'source_CSV_units_converted_fraction_to_pp_only':True,'source_delete_one_approximate_interval_used_in_Figure1d':True,'palette':[TEAL,MAGENTA,BLUE,TEXT],'SVG_text_editable':True,'PDF_TrueType_fonttype':42,'PNG_dpi':400},indent=2)+'\n',encoding='utf-8')
print('PASS Figures1/2 replotted from unchanged aggregate results')
