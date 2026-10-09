from glob import glob  # Used only for instructive purposes

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp
from pandas import DataFrame, Series  # for convenience

VIDEO_PATH = './data/Project data/Video'

CALIBRATION_FACTOR = 1 # TODO: determine

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

@pims.pipeline
def gray(image):
    return image[:, :, 1]  # Take just the green channel

def loadFramesFromVideo(video_name: str):
    return gray(pims.open(f'{VIDEO_PATH}/{video_name}'))

def processFrames(frames, minmass: int, processes: int = 1, show_plot: bool = False):
    f = tp.batch(frames, 23, invert=True, minmass=minmass, processes=processes)
    if show_plot: tp.annotate(f, frames[0])
    return f

def displayFrame(frame):
    plt.imshow(frame)

def getMassHistogram(processed_frames, bins: int = 20):
    fig, ax = plt.subplots()
    ax.hist(processed_frames['mass'], bins=bins)
    ax.set(xlabel='mass', ylabel='count')
    plt.show()

def calculateTrajectories(processed_frames, show_plot: bool = False):
    t = tp.link(processed_frames, 5, memory=3)

    #t1 = tp.filter_stubs(t, 25)
    t1 = t

    # Compare the number of particles in the unfiltered and filtered data.
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())

    #t2 = t1[((t1['mass'] > 50) & (t1['size'] < 2.6) & (t1['ecc'] < 0.3))]
    t2 = t1
    if show_plot:
        tp.plot_traj(t2)
        plt.show()
    return t2

def calculateOverallDrift(t2, show_plot: bool = False):
    d = tp.compute_drift(t2)
    if show_plot:
        d.plot()
        plt.show()
    return d

def plotDriftCorrectedTrajectories(t2, d):
    tm = tp.subtract_drift(t2.copy(), d)
    ax = tp.plot_traj(tm)
    plt.show()

def getAnnotatedVideo(frames, processed_frames):
    tp.annotate(processed_frames, frames)

def main():
    minmass = 50000
    frames = loadFramesFromVideo("A002 - 20261009_122705.wmv")
    #processed_frames = processFrames(frames, minmass)
    f = tp.locate(frames, 29, invert=True, minmass=minmass)
    getMassHistogram(f)
    tp.annotate(f, frames[0])
    #trajectories = calculateTrajectories(processed_frames)
    #drift = calculateOverallDrift(trajectories)
    #plotDriftCorrectedTrajectories(trajectories, drift)

if __name__ == "__main__":
    main()