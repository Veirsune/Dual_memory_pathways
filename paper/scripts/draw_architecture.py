"""Draw the GDSM layer architecture as editable vector artwork.

Panel structure follows the model-comparison / fast-slow circuit presentation
of the DMP paper. All artwork is drawn here; no source-paper image is embedded.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
INK = '#24313C'
MUTED = '#63717C'
LINE = '#CBD3D9'
FAST = '#4B5660'
CAND = '#477EA7'
STABLE = '#218777'
GATE = '#BC8634'
LIGHT_BLUE = '#EDF4F9'
LIGHT_GREEN = '#EBF5F1'
LIGHT_GOLD = '#FAF2E4'


def draw_architecture(root=ROOT):
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':8,
                         'mathtext.fontset':'dejavusans', 'pdf.fonttype':42,
                         'ps.fonttype':42, 'svg.fonttype':'none'}):
        fig = plt.figure(figsize=(7.4, 5.85), facecolor='white')
        ax = fig.add_axes([.015,.015,.97,.97])
        ax.set(xlim=(0,100), ylim=(0,79)); ax.set_aspect('equal'); ax.axis('off')

        def text(x,y,s,size=8,color=INK,ha='center',weight='normal',va='center',**kw):
            return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va=va,
                           fontweight=weight,linespacing=1.35,**kw)
        def line(points,color=FAST,lw=1.0,ls='-',z=2):
            x,y=zip(*points);ax.plot(x,y,color=color,lw=lw,ls=ls,zorder=z,solid_capstyle='round')
        def arrow(points,color=FAST,lw=1.05,ls='-',head=8,z=3):
            path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
            ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=head,
                                        lw=lw,color=color,linestyle=ls,zorder=z,
                                        capstyle='round',joinstyle='round'))
        def curve(points,color=FAST,lw=1.0,head=7):
            path=MplPath(points,[MplPath.MOVETO]+[MplPath.CURVE4]*3)
            ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=head,lw=lw,color=color))
        def dot(x,y,color=FAST):ax.add_patch(Circle((x,y),.3,fc=color,ec='none',zorder=4))
        def box(x,y,w,h,label='',fc='white',ec=LINE,size=8,color=INK,r=.7,lw=.85):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=lw,zorder=3))
            if label:text(x+w/2,y+h/2,label,size,color,zorder=4)
        def state(x,y,w,h,label,color,small=False):
            box(x,y,w,h,fc='white',ec=color,r=.6)
            # Discrete coordinates make the low-dimensional shared state visible.
            for i in range(5):
                ax.add_patch(Rectangle((x+.45+i*(w-.9)/5,y+.4),(w-.9)/5-.15,.65,fc=color,alpha=.22+.12*(i%3),ec='none',zorder=4))
            text(x+w/2,y+h*.64,label,7.8 if small else 9,color,zorder=4)
        def soma(x,y,r=2.55):
            ax.add_patch(Circle((x,y),r,fc='#F7F5EF',ec=FAST,lw=1,zorder=4))
            text(x,y,'LIF',8,zorder=5)
        def spikes(x,y,color=FAST,width=4,amp=1.4):
            pts=[(x,y)]
            for f in [.2,.52,.76]:
                q=x+width*f;pts += [(q,y),(q,y+amp),(q,y)]
            pts.append((x+width,y));line(pts,color,.8)
        def gate(x,y,scale=1):
            ax.add_patch(Polygon([(x-scale,y),(x,y+scale),(x+scale,y),(x,y-scale)],closed=True,fc=LIGHT_GOLD,ec=GATE,lw=1,zorder=5))
        def panel(x,y,letter,title):
            text(x,y,letter,12,ha='left',weight='bold')
            text(x+3.2,y,title,9.3,ha='left',weight='bold')

        # a: identical neuron symbols and fast paths make changes easy to compare.
        panel(0,77,'a','Shared memory changes what a spiking layer can retain')
        for x in [32.8,66.4]:line([(x,52),(x,73.5)],LINE,.65)
        text(16,72.5,'FSNN',9.3,weight='bold')
        text(49.5,72.5,'DMP',9.3,weight='bold')
        text(83,72.5,'GDSM',9.3,STABLE,weight='bold')

        spikes(2,61.9,width=4);text(3.8,58.5,r'$s_t$',7.5)
        arrow([(6.3,61.9),(23.05,61.9)],lw=1.35);text(13.5,64,r'$W_f$',8)
        soma(25.6,61.9);arrow([(28.15,61.9),(31,61.9)])
        spikes(28.4,62.4,width=2.4,amp=1)
        curve([(25.7,59.3),(29.1,55.7),(20.8,55.7),(23.9,59.8)],head=6)
        text(25.7,57.5,r'$\beta$',7)
        text(16.1,53.2,'Fast neuronal state only',7.5,MUTED)

        spikes(35.1,61.9,width=3.4);text(36.8,58.5,r'$s_t$',7.5)
        arrow([(39,61.9),(57.35,61.9)],lw=1.35);text(46.6,60,r'$W_f$',8)
        soma(59.9,61.9);arrow([(62.45,61.9),(65.1,61.9)])
        curve([(60,59.3),(63.4,55.7),(55.1,55.7),(58.2,59.8)],head=6)
        text(60,57.5,r'$\beta$',7)
        state(46.5,66.3,8.3,3.7,r'$h_t$',CAND,small=True)
        dot(41.3,61.9);arrow([(41.3,61.9),(41.3,68.15),(46.4,68.15)],CAND)
        text(41.3,70.2,r'$x_t\in\mathbb{R}$',7,CAND)
        arrow([(54.9,68.15),(59.9,68.15),(59.9,64.55)],CAND)
        text(49.5,55.2,'Scalar drive',7.5,MUTED)
        text(49.5,52.9,'One continuously updated state',7.3,MUTED)

        spikes(68.6,61.9,width=3.4);text(70.3,58.5,r'$s_t$',7.5)
        arrow([(72.4,61.9),(91.65,61.9)],lw=1.35);text(81,60,r'$W_f$',8)
        soma(94.2,61.9);arrow([(96.75,61.9),(99.6,61.9)])
        curve([(94.3,59.3),(97.7,55.7),(89.4,55.7),(92.5,59.8)],head=6)
        text(94.3,57.5,r'$\beta$',7)
        state(76.3,66.3,6.2,3.7,r'$p_t$',CAND,small=True)
        state(87.1,66.3,6.2,3.7,r'$m_t$',STABLE,small=True)
        dot(73.5,61.9);arrow([(73.5,61.9),(73.5,68.15),(76.2,68.15)],CAND)
        arrow([(82.6,68.15),(87,68.15)],GATE);gate(84.8,68.15,.7)
        text(84.8,70.5,r'$g_t$',7.5,GATE)
        arrow([(79.4,66.2),(79.4,65),(90.6,65),(92.4,63.8)],CAND,.9,head=6)
        arrow([(93.4,68.15),(96.8,68.15),(95.3,64.3)],STABLE,.9,head=6)
        text(83,55.2,'Vector drive',7.5,MUTED)
        text(83,52.9,'Candidate + gated stable state',7.3,MUTED)
        line([(0,50.5),(100,50.5)],LINE,.75)

        # b: expanded fast/slow circuit. Arrow color identifies the state read.
        panel(0,47.7,'b','Candidate–stable memory within one layer')
        # Fast pathway across the top; soma receives both memory reads from below.
        spikes(1.5,40,width=4.5,amp=1.8)
        text(3.4,43.5,'Layer input',7.5,MUTED);text(3.6,37.7,r'$s_t\in\mathbb{R}^{M}$',8)
        arrow([(6.5,40),(58,40)],FAST,1.7,head=10)
        text(29.5,42.7,r'Fast path $W_f s_t$',8.3,FAST)
        soma(61,40,r=3)
        arrow([(64.1,40),(70,40)],FAST,1.3);spikes(66,40.5,width=3.2,amp=1.4)
        text(67.2,36.9,'Spikes',7.5,MUTED)

        # Background groups indicate sharing, rather than an additional neuron.
        ax.add_patch(FancyBboxPatch((10.6,8.5),58.8,25.5,boxstyle='round,pad=0,rounding_size=1.4',fc='#FAFBFC',ec=LINE,lw=.9,linestyle=(0,(4,3)),zorder=0))
        text(50.5,32.2,r'Shared memory ($d\ll N$)',7.9,MUTED)

        dot(13.8,40)
        arrow([(13.8,40),(13.8,26.2)],CAND,1.15)
        box(10.9,20.2,13.3,6,'Vector drive\n'+r'$u_t\in\mathbb{R}^d$',LIGHT_BLUE,CAND,size=8)
        text(17,29,r'$W_u$',8,CAND)
        arrow([(24.3,23.2),(28.5,23.2)],CAND,1.2)
        state(28.6,19.7,13.2,9,'Candidate\n'+r'$p_t\in\mathbb{R}^{d}$',CAND)
        text(35.2,16.7,r'$\bar A p_{t-1}+D u_t$',8,CAND)
        curve([(30.3,19.6),(25.4,13),(44,13),(40.3,19.5)],CAND,head=7)
        text(35.2,11.6,'Integrate every frame',7.1,MUTED)

        # Writes are explicitly delayed one step.
        arrow([(41.9,23.2),(53.1,23.2)],GATE,1.25)
        box(43.1,21.7,4.4,3,r'$z^{-1}$','white',LINE,size=7.3,r=.3)
        gate(50.2,23.2,1.2)
        text(47.8,19.4,r'$p_{t-1}$',7.7,CAND)
        state(53.2,19.7,13.2,9,'Stable\n'+r'$m_t\in\mathbb{R}^{d}$',STABLE)
        curve([(54.9,19.6),(50,13),(68.6,13),(64.9,19.5)],STABLE,head=7)
        text(59.8,16.7,r'$1-g_t$',8,STABLE)
        text(59.8,11.6,'Write selectively',7.1,MUTED)

        # Separate reads, routed with no crossing of memory blocks.
        arrow([(35.2,28.9),(35.2,35.3),(58.7,35.3),(59.5,37.4)],CAND,1.2)
        text(41,36.9,r'$W_p p_t$',8,CAND)
        arrow([(65.5,28.9),(65.5,34),(62.5,37.3)],STABLE,1.2)
        text(68.2,30.8,r'$W_m m_t$',8,STABLE)

        # Causal control is separated from the data path and includes both inputs.
        box(19,2,32.2,5.1,'Causal statistics + past EMA',LIGHT_GOLD,GATE,size=7.7)
        text(2.2,4.6,r'$r_t,\ s_t$',8.4,GATE,ha='left')
        arrow([(10.7,4.6),(18.9,4.6)],GATE,1,ls='--')
        arrow([(51.3,4.6),(51.8,4.6),(51.8,9.1),(50.2,9.1),(50.2,21.8)],GATE,1,ls='--')
        text(53.8,8.7,r'$g_t$',8.5,GATE)
        # Read summation is the actual layer current, separate from neuronal reset.
        text(40.8,.1,r'$I_t=W_f s_t+W_p p_t+W_m m_t+b$',8.8,INK)

        # c: exact closed-gate consequence, no simulated or empirical traces.
        line([(73,0),(73,48)],LINE,.75)
        panel(76,47.7,'c','Closed-gate step')
        text(88,42.7,r'$g_t=\mathbf{0}$',11,GATE)
        text(88,38.9,'No stable write',8.2,weight='bold')
        # Coordinates remain identical across the step.
        for y,label in [(31.3,r'$m_{t-1}$'),(23,r'$m_t$')]:
            text(78,y+1.3,label,8.5,STABLE,ha='left')
            for i,a in enumerate([.34,.7,.47,.9,.56]):
                ax.add_patch(Rectangle((85.5+i*2.5,y),2.2,2.6,fc=STABLE,alpha=a,ec='white',lw=.35))
        arrow([(91.5,30.6),(91.5,26.1)],STABLE,1.1)
        text(88,18.8,r'$m_t=m_{t-1}$',10,STABLE)
        text(88,14.3,'Reuse stable read',8.2,weight='bold')
        text(88,10.3,r'$W_m m_t=W_m m_{t-1}$',8.3,STABLE)
        box(76.7,1.4,22.4,5.5,'Candidate and fast\npath remain active',LIGHT_BLUE,'none',size=7.8,color=CAND)

        folder=root/'figures';folder.mkdir(exist_ok=True)
        for extension in ['pdf','svg','png']:
            fig.savefig(folder/f'architecture.{extension}',dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.03)
        plt.close(fig)

if __name__=='__main__':
    draw_architecture()
