from glob import glob  # Used only for instructive purposes

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp
from pandas import DataFrame, Series  # for convenience
from scipy.constants import Boltzmann

VIDEO_PATH = './data/Project data/Video'

CALIBRATION_FACTOR = 1 # TODO: determine

FRAME_RATE = 20 # Frames per second

# Figure style definition
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

@pims.pipeline
def gray(image):
    '''Make image monochromatic by jsut taking the green channel in the RGB values.'''
    return image[:, :, 1]

def loadFramesFromVideo(video_name: str):
    '''Load all video frames and conver them to monochrome.'''
    return gray(pims.open(f'{VIDEO_PATH}/{video_name}'))

def processFrames(frames, feature_size: int, minmass: int, processes: int = 1, show_plot: bool = False):
    '''Annnotate all frames with detected particles.'''
    f = tp.batch(frames, feature_size, invert=True, minmass=minmass, processes=processes)
    if show_plot: tp.annotate(f, frames[0])
    return f

def displayFrame(frame):
    '''Display a single frame.'''
    plt.imshow(frame)

def getMassHistogram(processed_frames, bins: int = 20):
    '''Generate and display mass histogram based on all annotated video frames.'''
    fig, ax = plt.subplots()
    ax.hist(processed_frames['mass'], bins=bins)
    ax.set(xlabel='mass', ylabel='count')
    plt.show()

def calculateTrajectories(processed_frames, min_particle_frames_for_trajectory: int = 25):
    '''Calculate raw observed trajectories of tracked particles from annotated frames.'''
    t = tp.link(processed_frames, 5, memory=3)

    t1 = tp.filter_stubs(t, min_particle_frames_for_trajectory)
    #t1 = t

    # Compare the number of particles in the unfiltered and filtered data.
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())

    #t2 = t1[((t1['mass'] > 50) & (t1['size'] < 2.6) & (t1['ecc'] < 0.3))]
    t2 = t1

    return t2

def calculateOverallDrift(t2, show_plot: bool = False):
    '''Calcualte the drift of the whole frame based on particle trajectories.'''
    d = tp.compute_drift(t2)
    # If show_plot is True, display the plot of ovelall frame drift
    if show_plot:
        d.plot()
        plt.show()
    return d

def calculateDriftCorrectedTrajectories(t2, d):
    '''Correct for the overall frame drift in the particle trajecotries and plot them.'''
    return tp.subtract_drift(t2.copy(), d)

def plotTrajectories(trajectories):
    tp.plot_traj(trajectories)
    plt.show()

# Not finished
def getAnnotatedVideo(frames, processed_frames):
    tp.annotate(processed_frames, frames)

def getOneDimensionalDeltaTimeProbability(drift_corrected_trajectories, dimension):
    return 0

def getAverageMeanSquaredDisplacement(drift_corrected_trajectories):

    return 0

def getViscosityFromDiffusionParameter(temperature: float, particle_radius: float, diffusion_parameter: float):
    '''Calculates viscosity based on the Stokes-Einstein relation.'''
    return (Boltzmann * temperature) / (6 * np.pi * diffusion_parameter * particle_radius)

# Not finished
def calculateDiffusionParameterFromData():
    return 0

def main():
    '''Main function.'''
    minmass = 2000
    feature_size = 23
    frames = loadFramesFromVideo("A005 - 20261014_140457.wmv")
    processed_frames = processFrames(frames, feature_size, minmass)
    f = tp.locate(frames[0], feature_size, invert=True, minmass=minmass)
    getMassHistogram(f)
    tp.annotate(f, frames[0])
    trajectories = calculateTrajectories(processed_frames, len(frames))
    drift = calculateOverallDrift(trajectories)
    plotTrajectories(calculateDriftCorrectedTrajectories(trajectories, drift))

if __name__ == "__main__":
    '''Module helper.'''
    main()