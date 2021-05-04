import numpy as np

import torch
import torch.nn as nn
"""
Captum通过探索有助于PyTorch模型进行预测的特性，帮助您解释和理解PyTorch模型的预测。
"""
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,# 积分梯度
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(3, 2)

        # initialize weights and biases
        self.lin1.weight = nn.Parameter(torch.arange(-4.0, 5.0).view(3, 3))
        self.lin1.bias = nn.Parameter(torch.zeros(1,3))
        self.lin2.weight = nn.Parameter(torch.arange(-3.0, 3.0).view(2, 3))
        self.lin2.bias = nn.Parameter(torch.ones(1,2))

    def forward(self, input):
        return self.lin2(self.relu(self.lin1(input)))


model = ToyModel()
model.eval()

torch.manual_seed(123)
np.random.seed(123)

input = torch.rand(2, 3)
baseline = torch.zeros(2, 3)

ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)

print('IG Attributions:', attributions)
print('Convergence Delta:', delta)

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import cm
from numpy.random import randn

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

vegetables = range(14)
farmers = range(14)

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3, 0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

ax.set_title("Harvest of local farmers (in tons/year)")
im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                   cmap="bwr", cbarlabel="harvest [t/year]")
texts = annotate_heatmap(im, valfmt="{x:.1f} t")

fig.tight_layout()
fig.set_size_inches(18.5, 10.5)
plt.show()