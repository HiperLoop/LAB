import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pims
import trackpy as tp
from pandas import DataFrame, Series
from scipy.constants import Boltzmann

VIDEO_PATH = './data/Adam Data/Video'

CALIBRATION_FACTOR: float = 0.13                        # [mpp] Determined using Fiji

CAMERA_FRAME_RATE: int = 20                             # Frames per second

TEMPERTURE: float = 20.9                                # Temperature in degrees Celsius during video collection

FRAME_LIMIT: int = 60                                   # Number of frames to take from the start of the video and dump the rest. -1 for no limit

""" {
        "video_file": "A005 - 20261014_140457.wmv",
        "particle_radius": 2,
        "feature_size": 23,
        "minmass": 2000
    } """


""" {
        "video_file": "A002 - 20261021_140146.wmv",
        "particle_radius": 1.5,
        "feature_size": 17,
        "minmass": 2000
    }, """

""" {
        "video_file": "A003 - 20261021_141507.wmv",
        "particle_radius": 1,
        "feature_size": 17,
        "minmass": 600
    }, """

""" {
        "video_file": "A006 - 20261021_144131.wmv",
        "particle_radius": 0.75,
        "feature_size": 17,
        "minmass": 550
    }, """

""" {
        "video_file": "A002 - 20261021_140146.wmv",
        "particle_radius": 1.5,
        "feature_size": 17,
        "minmass": 2000
    },
    {
        "video_file": "A003 - 20261021_141507.wmv",
        "particle_radius": 1,
        "feature_size": 17,
        "minmass": 600
    },
    {
        "video_file": "A006 - 20261021_144131.wmv",
        "particle_radius": 0.75,
        "feature_size": 17,
        "minmass": 550
    }, """

""" {
        "video_file": "A018 - 20261021_155449.wmv",
        "particle_radius": 2,
        "feature_size": 17,
        "minmass": 1500
    },
    {
        "video_file": "A002 - 20261021_140146.wmv",
        "particle_radius": 1.5,
        "feature_size": 17,
        "minmass": 2000
    },
    {
        "video_file": "A003 - 20261021_141507.wmv",
        "particle_radius": 1,
        "feature_size": 17,
        "minmass": 600
    },
    {
        "video_file": "A006 - 20261021_144131.wmv",
        "particle_radius": 0.75,
        "feature_size": 17,
        "minmass": 550
    }, """

""" {
        "video_file": "A032 - 20261021_161942.wmv",
        "particle_radius": 0.75,
        "feature_size": 17,
        "minmass": 1000
    },
    {
        "video_file": "A006 - 20261021_144131.wmv",
        "particle_radius": 0.75,
        "feature_size": 17,
        "minmass": 550
    }, """
# Video files to be processed
VIDEO_PROCESSING_DATA: list[dict[str, typing.Any]] = [
    {
        "video_file": "A018 - 20261021_155449.wmv",
        "particle_radius": 2,
        "feature_size": 17,
        "minmass": 1500
    },
    {
        "video_file": "A002 - 20261021_140146.wmv",
        "particle_radius": 1.5,
        "feature_size": 17,
        "minmass": 2000
    },
    {
        "video_file": "A003 - 20261021_141507.wmv",
        "particle_radius": 1,
        "feature_size": 17,
        "minmass": 600
    },
    {
        "video_file": "A006 - 20261021_144131.wmv",
        "particle_radius": 0.75,
        "feature_size": 17,
        "minmass": 550
    },
    {
        "video_file": "A032 - 20261021_161942.wmv",
        "particle_radius": 0.75,
        "feature_size": 17,
        "minmass": 1000
    },
    {
        "video_file": "A038 - 20261021_163829.wmv",
        "particle_radius": 1,
        "feature_size": 17,
        "minmass": 500
    },
]                                 

VERBOSE: bool = True                                    # Determines whether intermediate steps are printed in console and figures shown

# Figure style definition
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

@pims.pipeline  # pyright: ignore[reportUntypedFunctionDecorator, reportAttributeAccessIssue]
def gray(image: Series) -> Series:
    '''Make image monochromatic by jsut taking the green channel in the RGB values.'''
    return image[:, :, 1]

def loadFramesFromVideo(video_name: str) -> Series:
    '''Load all video frames and conver them to monochrome.'''
    return gray(pims.open(f'{VIDEO_PATH}/{video_name}'))

def processFrames(frames: Series, feature_size: int, minmass: int, processes: int = 1, show_plot: bool = False) -> DataFrame:
    '''Annnotate all frames with detected particles.'''
    processed_frames: DataFrame = tp.batch(frames, feature_size, invert=True, minmass=minmass, processes=processes, )
    if show_plot: tp.annotate(processed_frames, frames[0])
    return processed_frames

def displayFrame(frame: DataFrame):
    '''Display a single frame.'''
    plt.imshow(frame)

def plotMassHistogram(processed_frames: DataFrame, bins: int = 20) -> None:
    '''Generate and display mass histogram based on all annotated video frames.'''
    _, ax = plt.subplots()
    ax.hist(processed_frames['mass'], bins=bins)
    ax.set(xlabel='mass', ylabel='count')
    plt.show()

def calculateTrajectories(processed_frames: DataFrame, min_particle_frames_for_trajectory: int = 25, print_before_after_trajectories: bool = False) -> DataFrame:
    '''Calculate raw observed trajectories of tracked particles from annotated frames.'''
    trajectories: DataFrame = tp.link(processed_frames, 5, memory=3)

    trajectories_present_in_all_frames: DataFrame = tp.filter_stubs(trajectories, min_particle_frames_for_trajectory)

    # Compare the number of particles in the unfiltered and filtered data.
    if print_before_after_trajectories:
        print('Before:', trajectories['particle'].nunique())
        print('After:', trajectories_present_in_all_frames['particle'].nunique())

    #t2 = t1[((t1['mass'] > 50) & (t1['size'] < 2.6) & (t1['ecc'] < 0.3))]
    #t2 = t1

    return trajectories_present_in_all_frames

def calculateOverallDrift(trajectories: DataFrame, show_plot: bool = False) -> DataFrame:
    '''Calcualte the drift of the whole frame based on particle trajectories.'''
    drift: DataFrame = tp.compute_drift(trajectories)
    # If show_plot is True, display the plot of ovelall frame drift
    if show_plot:
        drift.plot()
        plt.show()
    return drift

def calculateDriftCorrectedTrajectories(trajectories: DataFrame, drift: DataFrame) -> DataFrame:
    '''Correct for the overall frame drift in the particle trajecotries and plot them.'''
    return tp.subtract_drift(trajectories.copy(), drift)

def plotTrajectories(trajectories: DataFrame) -> None:
    '''Plots trajectories generated by tackpy.'''
    plt.figure()
    tp.plot_traj(trajectories)
    plt.show()

# Not finished
def getAnnotatedVideo(frames: DataFrame, processed_frames: DataFrame):
    tp.annotate(processed_frames, frames)

# Not finished
def getOneDimensionalDeltaTimeProbability(drift_corrected_trajectories: DataFrame, dimension: int):
    return 0

def getViscosityFromDiffusionParameter(temperature: float, particle_radius: float, diffusion_parameter: float) -> float:
    '''Calculates viscosity based on the Stokes-Einstein relation.'''
    return (Boltzmann * temperature) / (6 * np.pi * diffusion_parameter * particle_radius) * 1000

def calculateDiffusionParameterFromData(average_mean_squared_displacement: Series) -> tuple[float, float]:
    '''Uses the slope of the average mean squared displacement over time to calculate the Diffusion parameter. To get the slope a linear fit through the data is used.'''
    tau = average_mean_squared_displacement.index.to_numpy()
    msd_values = average_mean_squared_displacement.to_numpy() * (10**(-6))**2
    slope, offset = np.polyfit(tau, msd_values, 1)
    return slope / 4, offset

def celsiusToKelvin(temperature: float) -> float:
    '''Converts temperature from degrees Celsius to Kelvins'''
    return temperature + 273.15

def trackedParticletracjectories(video_name: str, feature_size: int, minmass: int, show_mass_histogram: bool = False, annotate_frame_zero: bool = False, plot_drift: bool = False, print_filter: bool = False) -> tuple[DataFrame, int]:
    '''Wrapping all frame processing and particle tracking into one function that in the end returns the drift-corrected trajectories.'''
    frames: Series = loadFramesFromVideo(video_name)[:FRAME_LIMIT]
    number_of_frames = len(frames)
    processed_frames = processFrames(frames, feature_size, minmass)
    if show_mass_histogram or annotate_frame_zero:
        f: DataFrame = tp.locate(frames[0], feature_size, invert=True, minmass=minmass)
        if show_mass_histogram: plotMassHistogram(f)
        if annotate_frame_zero: tp.annotate(f, frames[0])
    trajectories = calculateTrajectories(processed_frames, number_of_frames, print_filter)
    drift = calculateOverallDrift(trajectories, plot_drift)
    return calculateDriftCorrectedTrajectories(trajectories, drift), number_of_frames

def viscosityFromTrajectories(drift_corrected_trajectories: DataFrame, number_of_frames: int, temperature: float, particle_radius: float, show_amsd_plot: bool = False, show_trajectories : bool = False, print_viscosity: bool = True) -> float:
    '''Wrapping all mean squared dsiplacement, diffucion coefficient and viscosity calculation into one function that returns the calculated viscosity at the end.'''
    amsd: Series = tp.emsd(drift_corrected_trajectories, mpp=CALIBRATION_FACTOR, fps=CAMERA_FRAME_RATE, max_lagtime=number_of_frames//4)
    if show_amsd_plot:
        plt.figure()
        plt.plot(amsd)
        plt.show()
    if show_trajectories: plotTrajectories(drift_corrected_trajectories)
    D, offset = calculateDiffusionParameterFromData(amsd)
    viscosity = getViscosityFromDiffusionParameter(celsiusToKelvin(temperature), particle_radius * (10**(-6)), D)
    if print_viscosity: print(f'The viscosity is {viscosity} mPa/s')
    return viscosity

def calculateAllViscosities(video_data: list[dict[str, typing.Any]]) -> None:
    viscosities: np.ndarray = np.zeros(len(video_data))
    radii: np.ndarray = np.zeros(len(video_data))
    for i, data in enumerate(video_data):
        radii[i] = data["particle_radius"]
        if VERBOSE:
            trajectories, frames_n = trackedParticletracjectories(data["video_file"], data["feature_size"], data["minmass"], True, True, True, True)
            viscosities[i] = viscosityFromTrajectories(trajectories, frames_n, TEMPERTURE, radii[i], True, True, True)
        else:
            trajectories, frames_n = trackedParticletracjectories(data["video_file"], data["feature_size"], data["minmass"])
            viscosities[i] = viscosityFromTrajectories(trajectories, frames_n, TEMPERTURE, radii[i])

    plt.figure()
    plt.plot(1/radii, viscosities, marker="o")
    plt.plot([0, 2.5], [0, 2.5])
    plt.title("viscosity against 1/radius")
    plt.show()

def main() -> None:
    '''Main function.'''
    calculateAllViscosities(VIDEO_PROCESSING_DATA)

if __name__ == "__main__":
    '''Module helper.'''
    main()