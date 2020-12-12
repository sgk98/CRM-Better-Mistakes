import os
import json
import re
import argparse
import functools
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D
from glob import glob
import sys

XENT_TIERED = [
    ("experiments/crossentropy_tieredimagenet", ""),
]

NSAMPLES=5


def load_from_file(exp_path, section="val"):
    """Load all results for a given experiment as nested dict.
    Args:
        exp_path: The path of the experiment.
        section: Section of the results to get (train/val).
    Returns:
        Dictionary with all the results and the number of epochs.
    """

    json_path = os.path.join(exp_path, "json/")
    results = {}
    epochs = []

    # get and store name
    results["name"] = os.path.basename(os.path.normpath(exp_path))

    # load all json files
    paths = glob(os.path.join(json_path, section, "*"))
    for path in paths:
        m = re.match(r".*epoch\.(\d+)\.json", path)
        if m is not None:
            with open(path) as f:
                epoch = int(m.group(1))
                epochs.append(epoch)
                results[epoch] = json.load(f)
        else:
            raise RuntimeError("Can not match path ", path)

    assert len(epochs), "Could not load data for " + exp_path

    return results, list(sorted(epochs))


def get_optimal_epoch(results, epochs, min_epoch, max_epoch=None, plot_fit=False):
    """
    Get epoch that minimizes the loss.
    Args:
        results: The result dict from get_results().
        epochs: The list of epochs to use.
    Return:
        The optimal epoch number.
    """
    loss_key = [k for k in results[epochs[0]].keys() if k[:5] == "loss/"]
    assert len(loss_key) == 1, "Found multiple losses !"
    loss_key = loss_key[0]
    max_epoch = max_epoch or max(epochs)

    # use a polynomial fit
    fit_epochs = [e for e in epochs if min_epoch <= e <= max_epoch]
    fit_losses = [results[e][loss_key] for e in fit_epochs]
    poly = np.poly1d(np.polyfit(fit_epochs, fit_losses, 4))
    #min_epoch_num=np.array(fit_epochs).argmin()

    # get the minimum of the curve
    crit = poly.deriv().r
    r_crit = crit[crit.imag == 0].real
    test = poly.deriv(2)(r_crit)

    x_min = r_crit[test > 0]
    x_min = x_min[x_min <= max_epoch]
    if len(x_min) == 0:
        x_min = np.array([max_epoch])  # no solution means min is at the end
    if len(x_min) > 1:
        x_min = np.array([x_min[np.argmax(poly(x_min))]])  # two solutions: take the min
    if poly(max_epoch) < poly(x_min):
        x_min = x_min[np.argmax(poly(x_min))]  # consider boundary as well
    y_min = poly(x_min)
    '''
    if plot_fit:
        # plot fit line for visual check
        plt.figure()
        plt.plot(fit_epochs, fit_losses)
        plt.plot(fit_epochs, poly(fit_epochs))
        plt.plot(x_min, y_min, "ro")
        plt.savefig(os.path.join(OUTPUT_DIR, results["name"] + "-loss-fit.pdf"), bbox_inches="tight")
        plt.close()
    '''
    # find index of epoch closest to the min
    x_min=min(fit_losses)
    index = np.argmin([abs(results[e][loss_key] - x_min) for e in epochs])

    return index



def get_results(path, nsamples, min_epoch, max_epoch=None, plot_fit=False):
    """
    Load results for a given path and find the epoch range.
    Args:
        path: The path (relative to base directory) where the results are.
        nsamples: Number of samples to use for computing mean+-std.
        min_epoch, max_epoch: The min and max epoch to perform the fit.
    """
    print("\nLoading results from {}".format(path))
    results, epochs = load_from_file(path)

    if max_epoch is not None:
        epochs = [e for e in epochs if e <= max_epoch]
    nepochs = len(epochs) - 1
    print("Found {} epochs.".format(nepochs))

    # get index of the best epoch
    best = get_optimal_epoch(results, epochs, min_epoch, max_epoch, plot_fit=plot_fit)

    print("Best epoch is {}".format(best))

    # compute start and end indices
    end = best + nsamples // 2
    if end > nepochs:
        end = nepochs
    start = end - nsamples + 1

    print("Selecting epoch range [{}, {}].".format(epochs[start], epochs[end]))

    return results, epochs, start, end


if __name__=="__main__":
    get_results(sys.argv[1],NSAMPLES,0)