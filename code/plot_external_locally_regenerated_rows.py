"""Publication figure from audited aggregate outputs; no new model or signal work."""
from pathlib import Path
import json,csv,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

O=Path(__file__).resolve().parent
if not (O/'source_data').exists() and (O.parent/'source_data').exists():O=O.parent
PALETTE={'teal':'#00A89D','magenta':'#ED388A','blue':'#347BDA','text':'#263244','white':'#FFFFFF','grid':'#E7ECF2'}
def load(name):return list(csv.DictReader((O/'source_data'/name).open(encoding='utf-8',newline='')))
def style(ax):
    ax.spines[['top','right']].set_visible(False)
    for spine in ax.spines.values():spine.set_color('#CDD5E0');spine.set_linewidth(.6)
    ax.tick_params(length=2.5,width=.6,color='#A9B4C4');ax.set_axisbelow(True)
def main():
    global O
    parser=argparse.ArgumentParser(description='Figure3/S3 require locally regenerated row data, deliberately excluded from release')
    parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    O=args.data_dir.resolve();assert not args.output_dir.exists(), 'Choose new output directory'
    auditfile=O/'FINAL_SCIENTIFIC_RESULTS_AUDIT.json'
    if not auditfile.exists():auditfile=O/'qa/FINAL_SCIENTIFIC_RESULTS_AUDIT.json'
    audit=json.loads(auditfile.read_text(encoding='utf-8'));assert audit['status'].startswith('PASS')
    people=load('EXTERNAL_PRIMARY_PERSON_ESTIMATES.csv');deleted=load('EXTERNAL_WHOLEPERSON_DELETION_ESTIMATES.csv')

    f=args.output_dir;f.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'text.color':PALETTE['text'],'axes.labelcolor':PALETTE['text'],'xtick.color':PALETTE['text'],'ytick.color':PALETTE['text'],'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white','axes.titlesize':9,'axes.titleweight':'bold'})
    fig,axes=plt.subplots(1,3,figsize=(7.4,3.75),gridspec_kw={'width_ratios':[1.25,.85,1.05]})
    ax=axes[0];y=np.arange(12)[::-1]+1;v=np.array([float(r['primary199_DoD_pp']) for r in people])
    ax.axvline(0,color='#9BA8BA',lw=.7,ls=(0,(3,3)))
    ax.scatter(v,y,s=20,color=PALETTE['blue'],edgecolors='white',linewidths=.5,zorder=3)
    mean=audit['primary_mean_pp'];lo,hi=audit['approximate_95_interval_pp']
    ax.errorbar(mean,-.25,xerr=[[mean-lo],[hi-mean]],fmt='D',color=PALETTE['teal'],markersize=5,elinewidth=1.5,capsize=2.5,zorder=4)
    ax.set_yticks([*y,-.25],[r['person'].replace('sub-','') for r in people]+['All 12'])
    ax.set_ylim(-1,13);ax.set_xlabel('B4 minus PCA16\ninclusion effect (pp)');ax.set_ylabel('Published participant ID')
    ax.set_title('Primary contrast',loc='left',pad=10)
    ax.grid(axis='x',color=PALETTE['grid'],lw=.6);ax.xaxis.set_major_locator(MaxNLocator(4));style(ax)
    ax=axes[1];b=np.array([float(r['B4_primary199_inclusion_pp']) for r in people]);p=np.array([float(r['PCA16_primary199_inclusion_pp']) for r in people])
    for j in range(12):ax.plot([0,1],[b[j],p[j]],color='#C9D2DF',lw=.7,alpha=.8,zorder=1)
    ax.scatter(np.zeros(12),b,s=17,color=PALETTE['teal'],edgecolors='white',linewidths=.4,zorder=3)
    ax.scatter(np.ones(12),p,s=17,color=PALETTE['magenta'],edgecolors='white',linewidths=.4,zorder=3)
    ax.plot([-.15,.15],[b.mean()]*2,color=PALETTE['text'],lw=1.8);ax.plot([.85,1.15],[p.mean()]*2,color=PALETTE['text'],lw=1.8)
    ax.axhline(0,color='#9BA8BA',lw=.7,ls=(0,(3,3)));ax.set_xlim(-.4,1.4);ax.set_xticks([0,1],['B4','PCA16'])
    ax.set_ylabel('Include own P minus exclude (pp)');ax.set_title('Paired readouts',loc='left',pad=10);ax.grid(axis='y',color=PALETTE['grid'],lw=.6);style(ax)
    ax=axes[2];v=np.array([float(r['wholeperson_delete_refit_mean_pp']) for r in deleted]);y=np.arange(12)[::-1]
    ax.axvline(0,color='#9BA8BA',lw=.7,ls=(0,(3,3)));ax.axvline(mean,color=PALETTE['teal'],lw=1.1)
    ax.scatter(v,y,s=20,color=PALETTE['blue'],edgecolors='white',linewidths=.5,zorder=3)
    ax.set_yticks(y,[r['deleted_person'].replace('sub-','') for r in deleted]);ax.set_ylabel('Deleted participant ID')
    ax.set_xlabel('Whole person refit\nmean contrast (pp)');ax.set_title('Full pipeline sensitivity',loc='left',pad=10)
    ax.grid(axis='x',color=PALETTE['grid'],lw=.6);ax.xaxis.set_major_locator(MaxNLocator(4));style(ax)
    for j,ax in enumerate(axes):ax.text(-.18,1.07,'abc'[j],transform=ax.transAxes,fontweight='bold',fontsize=12,color=PALETTE['text'])
    fig.subplots_adjust(left=.095,right=.99,bottom=.21,top=.85,wspace=.68)
    for ext in ('svg','pdf','png'):
        target=f/('Figure_3_External_Speech_Probe.'+ext);assert not target.exists();fig.savefig(target,dpi=350,facecolor='white')
    plt.close(fig)
    # Data qualification figure: true counts, not experimental sample replication.
    countfile=O/'source_data/FINAL15_PERSON_SESSION_COUNTS.csv'
    assert countfile.exists(), 'Regenerate counts from qualified public data; row table is excluded from release'
    counts=list(csv.DictReader(countfile.open(encoding='utf-8',newline='')))
    fig,ax=plt.subplots(figsize=(7.4,3.1));x=np.arange(15)
    for j,(cat,color) in enumerate([('flower','teal'),('penguin','blue'),('guitar','magenta')]):
        ax.bar(x+(j-1)*.25,[int(r[cat]) for r in counts],width=.23,color=PALETTE[color],label=cat.title())
    ax.set_xticks(x,[r['person'].replace('sub-','')+'/'+r['session'].replace('ses-','')+('*' if r['person']=='sub-013' else '') for r in counts],rotation=45,ha='right',fontsize=7)
    ax.set_ylabel('Eligible original P/I pairs');ax.set_xlabel('Published participant / session');ax.set_ylim(0,55);ax.set_yticks([0,10,20,30,40,50])
    ax.legend(loc='upper right',ncol=3,frameon=False,fontsize=8);ax.grid(axis='y',color=PALETTE['grid'],lw=.6);style(ax)
    fig.subplots_adjust(left=.08,right=.99,bottom=.30,top=.95)
    for ext in ('svg','pdf','png'):
        target=f/('Figure_S3_External_Eligible_Counts.'+ext);assert not target.exists();fig.savefig(target,dpi=350,facecolor='white')
    plt.close(fig)
    (f/'FIGURE_PALETTE_AND_SOURCE_RECEIPT.json').write_text(json.dumps({'palette':PALETTE,'Figure3_source':['EXTERNAL_PRIMARY_PERSON_ESTIMATES.csv','EXTERNAL_WHOLEPERSON_DELETION_ESTIMATES.csv'],'FigureS3_source':'FINAL15_PERSON_SESSION_COUNTS.csv','independent_n':12,'MC_draws_are_not_people':True,'Figure3_errorbar':'Approximate whole-person delete/refit t11 95% interval for aggregate primary only; deletion dots are estimates, not confidence limits','sub013_asterisk':'Published directory13 mapping; original FDT basename12 anomaly disclosed'},indent=2)+'\n',encoding='utf-8')
    print('PUBLICATION_FIGURES3_AND_S3_CREATED_FROM_AUDITED_AGGREGATES',flush=True)
if __name__=='__main__':main()
