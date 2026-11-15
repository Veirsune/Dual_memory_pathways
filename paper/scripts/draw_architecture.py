"""Figure 1: DMP and GDSM circuits, drawn as black-on-white vector artwork.

Edit NODES, ROUTES and LABELS below to change the layout. Each state appears
once per circuit. Explanations and update equations belong in the caption/text.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Polygon
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
INK = '#000000'
BACKGROUND = '#FFFFFF'
FIGURE_SIZE = (7.4, 3.35)
CANVAS = (101, 45)
FONT_FAMILY = 'DejaVu Sans'
OUTPUT_DPI = 300
# Rectangles: (left, bottom, width, height, label).
NODES = {
    'memory': (12.5, 17, 11, 6, 'Memory\n'+r'$h_t$'),
    'drive': (43.5, 17, 10, 6, 'Vector\ndrive'),
    'candidate': (59, 17, 12, 6, 'Candidate\n'+r'$p_t$'),
    'stable': (84, 17, 12, 6, 'Stable\n'+r'$m_t$'),
    'gate': (55, 2, 17, 5, 'Gate'),
}
# Polyline arrows: ([start, bend(s), end], width, dashed).
ROUTES = {
    'dmp_fast': ([(3,34),(25.1,34)],1.25,False),
    'dmp_drive': ([(7,34),(7,20),(12.4,20)],1,False),
    'dmp_read': ([(23.6,20),(28,20),(28,31.1)],1,False),
    'dmp_output': ([(30.9,34),(33,34)],1,False),
    'gdsm_fast': ([(40,34),(92.1,34)],1.25,False),
    'vector_drive': ([(46.5,34),(46.5,23.1)],1,False),
    'candidate_drive': ([(53.6,20),(58.9,20)],1,False),
    'stable_write': ([(71.1,20),(83.9,20)],1,False),
    'candidate_read': ([(65,23.1),(65,30),(92.5,30),(93.5,31.5)],1,False),
    'stable_read': ([(94,23.1),(94,31.2)],1,False),
    'gate_input': ([(40.8,34),(40.8,4.5),(54.9,4.5)],1,False),
    'gate_control': ([(72.1,4.5),(78,4.5),(78,18.7)],1,True),
    'gdsm_output': ([(97.9,34),(100.5,34)],1,False),
}
# Labels: (x, y, text, fontsize). No explanatory slogans inside the drawing.
LABELS = [
    (1,34,r'$s_t$',9), (14,36.5,r'$W_f$',9),
    (9.5,25,r'$\bar b x_t$',9), (18,14.2,r'$\bar A$',9),
    (38.5,34,r'$s_t$',9), (65,36.5,r'$W_f$',9),
    (56.2,25.5,r'$\bar b\odot u_t$',9),
    (74.5,23.5,r'$p_{t-1}$',9), (65,14.2,r'$\bar A$',9),
    (90,14.2,r'$1-g_t$',9), (77,32,r'$W_p$',9),
    (91.2,26,r'$W_m$',9), (80.2,8,r'$g_t$',9),
    (50.5,11.5,'Shared memory',8),
]


def draw_architecture(root=ROOT):
    with plt.rc_context({'font.family':FONT_FAMILY, 'font.size':9,
                         'text.color':INK, 'mathtext.fontset':'dejavusans',
                         'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'}):
        fig=plt.figure(figsize=FIGURE_SIZE,facecolor=BACKGROUND)
        ax=fig.add_axes([.01,.01,.98,.98])
        ax.set(xlim=(-1,CANVAS[0]),ylim=(0,CANVAS[1]))
        ax.set_aspect('equal');ax.axis('off')

        def label(x,y,s,size=9,weight='normal',ha='center'):
            ax.text(x,y,s,fontsize=size,ha=ha,va='center',color=INK,
                    fontweight=weight,linespacing=1.2,zorder=5)
        def arrow(points,lw=1,dashed=False):
            p=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
            ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>',mutation_scale=8,
                lw=lw,color=INK,linestyle=(0,(4,3)) if dashed else '-',
                joinstyle='miter',capstyle='butt',zorder=2))
        def recurrence(x):
            points=[(x-3.8,16.9),(x-8,11),(x+8,11),(x+3.8,16.9)]
            p=MplPath(points,[MplPath.MOVETO]+[MplPath.CURVE4]*3)
            ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>',mutation_scale=8,lw=1,color=INK,zorder=2))

        label(0,42,'a',12,'bold',ha='left');label(3.5,42,'DMP',10,'bold',ha='left')
        label(37,42,'b',12,'bold',ha='left');label(40.5,42,'GDSM',10,'bold',ha='left')
        # One shared-memory boundary, white inside, black dashed edge.
        ax.add_patch(Rectangle((42.5,9.5),56,18.5,fc=BACKGROUND,ec=INK,lw=.8,linestyle=(0,(4,3)),zorder=0))
        for points,lw,dashed in ROUTES.values():arrow(points,lw,dashed)
        for x,y in [(7,34),(40.8,34),(46.5,34)]:
            ax.add_patch(Circle((x,y),.27,fc=INK,ec=INK,zorder=3))
        for x,y,w,h,s in NODES.values():
            ax.add_patch(Rectangle((x,y),w,h,fc=BACKGROUND,ec=INK,lw=1,zorder=3))
            label(x+w/2,y+h/2,s,8.8)
        for x in [28,95]:
            ax.add_patch(Circle((x,34),2.8,fc=BACKGROUND,ec=INK,lw=1,zorder=3))
            label(x,34,'LIF',8.8)
        # A single write gate, with the incoming candidate labelled p_{t-1}.
        ax.add_patch(Polygon([(76.8,20),(78,21.2),(79.2,20),(78,18.8)],
                            closed=True,fc=BACKGROUND,ec=INK,lw=1,zorder=4))
        for x in [18,65,90]:recurrence(x)
        for item in LABELS:label(*item)
        folder=root/'figures';folder.mkdir(exist_ok=True)
        for extension in ['pdf','svg','png']:
            fig.savefig(folder/f'architecture.{extension}',dpi=OUTPUT_DPI,
                        facecolor=BACKGROUND,bbox_inches='tight',pad_inches=.04)
        plt.close(fig)

if __name__=='__main__':
    draw_architecture()
