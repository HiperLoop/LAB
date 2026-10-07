from glob import glob  # Used only for instructive purposes

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp
from pandas import DataFrame, Series  # for convenience

VIDEO_PATH = './data/Project data/Video'

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel


def main():
    frames = gray(pims.open(f'{VIDEO_PATH}/A001 - 20261007_155450.wmv'))

    plt.imshow(frames[0])

    f = tp.batch(frames, 23, invert=True, minmass=2000, processes=1)

    tp.annotate(f, frames[0]);

    fig, ax = plt.subplots()
    ax.hist(f['mass'], bins=20)

    # Optionally, label the axes.
    ax.set(xlabel='mass', ylabel='count')

    plt.show()

    t = tp.link(f, 5, memory=3)

    #t1 = tp.filter_stubs(t, 25)
    t1 = t
    # Compare the number of particles in the unfiltered and filtered data.
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())

    #t2 = t1[((t1['mass'] > 50) & (t1['size'] < 2.6) & (t1['ecc'] < 0.3))]
    t2 = t1

    plt.figure()
    tp.annotate(t2[t2['frame'] == 0], frames[0])
    plt.show()

    plt.figure()
    tp.plot_traj(t2)
    plt.show()

    d = tp.compute_drift(t2)
    d.plot()
    plt.show()

if __name__ == "__main__":
    main()