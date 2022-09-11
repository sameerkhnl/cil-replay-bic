from matplotlib import markers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def _plot_line_chart(df_avg, df_std, filepath_plot, title, colors=None):

    COLORS = ["grey",  "deepskyblue","indianred", "black", "blue", "yellowgreen", "goldenrod", "red", "darkblue", "brown", "crimson", "saddlebrown", "aqua", "olive", "mediumvioletred", "darkorange", "darkslategray"]
    if colors is None:
        colors = COLORS
    fig,ax = plt.subplots(figsize=(12,8))
    x = [i+1 for i in range(len(df_avg.columns))]
    for (i,r1),(j,r2),c in zip(df_avg.iterrows(), df_std.iterrows(), colors):
        y = r1.to_numpy()
        stdev = r2.to_numpy()
        line = ax.plot(x,y, label=r1.name, linewidth=2, color=c, marker='.')
        ax.errorbar(x,y,yerr=stdev, color=c)
        # ax.fill_between(x,y-stdev, y+stdev, alpha=0.3, color=c)
    leg = ax.legend(shadow=False, ncol=1)
    
    if len(x) > 50:
        _ = ax.set_xticks(np.arange(0, len(x)+1, 5))
    else:
        _ = ax.set_xticks(x)
    ax.set_xlabel('Number of tasks', fontsize=12)
    ax.set_ylabel('Average accuracy (on tasks seen so far)', fontsize=12)
    ax.set_ylim((0,1))
    # ax.set_title(title, fontsize=12)

    plt.xticks(rotation=75)
    plt.savefig(filepath_plot, bbox_inches='tight')

def output_plot(filepath_main,filepath_plot, title):
    df = pd.read_csv(filepath_main).round(3)

    df = df.iloc[:,:-2]
    df.set_index('agent',inplace=True)
    avg = df.groupby(['agent']).mean().round(3)
    stdev = df.groupby(['agent']).std(ddof=0).round(3)
    _plot_line_chart(avg,stdev, filepath_plot, title)