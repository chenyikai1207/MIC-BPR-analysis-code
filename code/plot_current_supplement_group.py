"""Current S1 schematic and S2 group summary; same source geometry, current palette."""
from pathlib import Path
import csv,json,argparse
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
D=args.data_dir;F=args.output_dir;assert not F.exists();F.mkdir(parents=True)
BLUE='#00A89D';RUST='#ED388A';GRAY='#D2D9DE';INK='#263244';GOLD='#A57A18'
plt.rcParams.update({'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':INK,'font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))

def clean(ax):ax.tick_params(length=3,color='#777777');ax.grid(axis='x',color='#E7ECF2',lw=.5,zorder=0);ax.set_axisbelow(True)

def panel(ax,letter,title):ax.set_title(title,loc='left',pad=9);ax.text(-.12,1.09,letter,transform=ax.transAxes,weight='bold',size=12,va='top')

def save(fig,name):
 for ext in ('png','svg','pdf'):fig.savefig(F/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.09)
 plt.close(fig)

def box(ax,x,y,w,h,text,color='#F4F7FB',edge='#347BDA',size=9):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.015',fc=color,ec=edge,lw=.8,clip_on=False));ax.text(x+w/2,y+h/2,text,ha='center',va='center',size=size,color=INK)

def arrow(ax,x,y,xx,yy,color='#347BDA'):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=10,lw=1,color=color))

def fig1():
 fig=plt.figure(figsize=(7.15,5.1));gs=fig.add_gridspec(2,2,height_ratios=[.95,1.15],hspace=.45,wspace=.35);a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
 for ax in (a,b,c):ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
 panel(a,'a','A heard melody and its imagined counterpart');box(a,.02,.52,.43,.25,'Perception\n0–2 s',color='#DDEBFB');box(a,.53,.52,.43,.25,'Imagery\n2–4 s',color='#FBDDEB');arrow(a,.45,.65,.53,.65);a.text(.03,.30,'Recall',weight='bold');a.text(.35,.30,'A–C#–E  →  A–C#–E');a.text(.03,.09,'Reversal',weight='bold');a.text(.35,.09,'A–C#–E  →  E–C#–A')
 panel(b,'b','Source fitting and target calibration');box(b,.06,.66,.85,.26,'Source listeners  ·  recall originals\nLearn fixed phase transformations',color='#DDEBFB',size=8.5);arrow(b,.49,.65,.49,.47);box(b,.06,.17,.85,.29,'Different target listeners\nCalibrate on perception\nScore imagery',color='#FBDDEB',size=8.5);b.text(.49,.025,'Distinct recall and reversal originals',ha='center',size=8)
 panel(c,'c','Three predictions distinguish the information supporting readout');labels=[('True melody or arbitrary code','Synchronized arbitrary codes\ncan expose original association'),('Correct or reassigned evaluation','Within melody and pause half\ncoarse stratum information is held'),('Own perception included or excluded','Same N − 1 size and class counts\nactual calibration input is exchanged')]
 for j,(head,detail) in enumerate(labels):
  x=.01+j*.335;box(c,x,.30,.305,.50,head.replace(' or ','\nor\n') if j<2 else 'Own perception\nincluded or excluded',color='#F4F7FB',size=9);c.text(x+.1525,.105,detail,ha='center',va='center',size=8)
 save(fig,'Figure_S1')

def figS1():
 r=rows(D/'FigureS1_strict_predictions.csv')
 fig,ax=plt.subplots(figsize=(7.15,3.0));fig.subplots_adjust(left=.29,bottom=.23,right=.96,top=.87);panel(ax,'a','Earlier independent and matched transfer contrasts')
 for y,q in enumerate(r):
  v=float(q['mean_pp']);ax.plot([float(q['source_approx95low']),float(q['source_approx95high'])],[y,y],color=GRAY,lw=4);ax.errorbar(v,y,xerr=[[v-float(q['conditional95low'])],[float(q['conditional95high'])-v]],fmt='o',ms=4,lw=1.5,capsize=2,color=BLUE)
 ax.set_yticks(range(len(r)),[q['label'] for q in r]);ax.invert_yaxis();ax.axvline(0,color='#888888',lw=.7);ax.set_xlim(-7,7);ax.set_xlabel('Balanced accuracy difference (pp)');clean(ax);save(fig,'Figure_S2')
fig1();figS1()
