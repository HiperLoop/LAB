import typing

import cv2
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

OLD_DATA: list[dict[str, typing.Any]] = [
    {
        "video_file": "A018 - 20261021_155449.wmv",
        "particle_diameter": 2,
        "feature_size": 17,
        "minmass": 1500,
        "frame_limit": 60
    },
    {
        "video_file": "A002 - 20261021_160146.wmv",
        "particle_diameter": 1.5,
        "feature_size": 17,
        "minmass": 2000,
        "frame_limit": 60
    },
    {
        "video_file": "A003 - 20261021_141507.wmv",
        "particle_diameter": 1,
        "feature_size": 17,
        "minmass": 600,
        "frame_limit": 60
    },
    {
        "video_file": "A042 - 20261023_110455.wmv",
        "particle_diameter": 1,
        "feature_size": 17,
        "minmass": 2000,
        "frame_limit": 80
    },
    {
        "video_file": "A006 - 20261021_144131.wmv",
        "particle_diameter": 0.75,
        "feature_size": 17,
        "minmass": 550,
        "frame_limit": 60
    },
    {
        "video_file": "A032 - 20261021_161942.wmv",
        "particle_diameter": 0.75,
        "feature_size": 17,
        "minmass": 1000,
        "frame_limit": 60
    },
    {
        "video_file": "A050 - 20261023_112259.wmv",
        "particle_diameter": 1.5,
        "feature_size": 17,
        "minmass": 1500,
        "frame_limit": 60
    }
]

""" {
        "video_file": "A057 - 20261023_115737.wmv",
        "particle_diameter": 2,
        "feature_size": 27,
        "minmass": 2000,
        "frame_limit": 60
    } """

""" {
        "video_file": "A018 - 20261021_155449.wmv",
        "particle_diameter": 2,
        "feature_size": 17,
        "minmass": 1500,
        "frame_limit": 60
    },
    {
        "video_file": "A002 - 20261021_160146.wmv",
        "particle_diameter": 1.5,
        "feature_size": 17,
        "minmass": 2000,
        "frame_limit": 60
    },
    {
        "video_file": "A003 - 20261021_141507.wmv",
        "particle_diameter": 1,
        "feature_size": 17,
        "minmass": 600,
        "frame_limit": 60
    },
    {
        "video_file": "A042 - 20261023_110455.wmv",
        "particle_diameter": 1,
        "feature_size": 17,
        "minmass": 2000,
        "frame_limit": 80
    },
    {
        "video_file": "A006 - 20261021_144131.wmv",
        "particle_diameter": 0.75,
        "feature_size": 17,
        "minmass": 550,
        "frame_limit": 60
    },
    {
        "video_file": "A032 - 20261021_161942.wmv",
        "particle_diameter": 0.75,
        "feature_size": 17,
        "minmass": 1000,
        "frame_limit": 60
    }, """

# Video files to be processed
VIDEO_PROCESSING_DATA: list[dict[str, typing.Any]] = [
    {
        "video_file": "A072 - 20261023_123706.wmv",
        "particle_diameter": 0.75,
        "feature_size": 9,
        "minmass": 500,
        "frame_limit": 60
    },
    {
        "video_file": "A073 - 20261023_123735.wmv",
        "particle_diameter": 0.75,
        "feature_size": 9,
        "minmass": 500,
        "frame_limit": 60
    },
    {
        "video_file": "A076 - 20261023_123837.wmv",
        "particle_diameter": 0.75,
        "feature_size": 9,
        "minmass": 500,
        "frame_limit": 60
    },
    {
        "video_file": "A080 - 20261023_123959.wmv",
        "particle_diameter": 0.75,
        "feature_size": 9,
        "minmass": 500,
        "frame_limit": 60
    },
    {
        "video_file": "A064 - 20261023_123011.wmv",
        "particle_diameter": 1,
        "feature_size": 17,
        "minmass": 1500,
        "frame_limit": 60
    },
    {
        "video_file": "A065 - 20261023_123030.wmv",
        "particle_diameter": 1,
        "feature_size": 17,
        "minmass": 1500,
        "frame_limit": 60
    },
    {
        "video_file": "A068 - 20261023_123138.wmv",
        "particle_diameter": 1,
        "feature_size": 17,
        "minmass": 1500,
        "frame_limit": 60
    },
    {
        "video_file": "A069 - 20261023_123205.wmv",
        "particle_diameter": 1,
        "feature_size": 17,
        "minmass": 1500,
        "frame_limit": 60
    },
    {
        "video_file": "A050 - 20261023_112259.wmv",
        "particle_diameter": 1.5,
        "feature_size": 11,
        "minmass": 200,
        "frame_limit": 40
    },
    {
        "video_file": "A058 - 20261023_121301.wmv",
        "particle_diameter": 1.5,
        "feature_size": 11,
        "minmass": 200,
        "frame_limit": 60
    },
    {
        "video_file": "A059 - 20261023_121310.wmv",
        "particle_diameter": 1.5,
        "feature_size": 11,
        "minmass": 200,
        "frame_limit": 60
    },
    {
        "video_file": "A060 - 20261023_121408.wmv",
        "particle_diameter": 1.5,
        "feature_size": 11,
        "minmass": 200,
        "frame_limit": 60
    },
    {
        "video_file": "A061 - 20261023_121415.wmv",
        "particle_diameter": 1.5,
        "feature_size": 11,
        "minmass": 200,
        "frame_limit": 60
    },
    {
        "video_file": "A062 - 20261023_122547.wmv",
        "particle_diameter": 1.5,
        "feature_size": 11,
        "minmass": 200,
        "frame_limit": 60
    },
    {
        "video_file": "A063 - 20261023_122603 - last 1.5.wmv",
        "particle_diameter": 1.5,
        "feature_size": 11,
        "minmass": 200,
        "frame_limit": 60
    },
    {
        "video_file": "A052 - 20261023_114819.wmv",
        "particle_diameter": 2,
        "feature_size": 15,
        "minmass": 1000,
        "frame_limit": 60
    },
    {
        "video_file": "A053 - 20261023_114827.wmv",
        "particle_diameter": 2,
        "feature_size": 17,
        "minmass": 2000,
        "frame_limit": 60
    },
    {
        "video_file": "A056 - 20261023_115426.wmv",
        "particle_diameter": 2,
        "feature_size": 17,
        "minmass": 2000,
        "frame_limit": 60
    },
    {
        "video_file": "A057 - 20261023_115737 - last 2.wmv",
        "particle_diameter": 2,
        "feature_size": 17,
        "minmass": 2000,
        "frame_limit": 60
    },
]               

VERBOSE: bool = False                                    # Determines whether intermediate steps are printed in console and figures shown

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
    tp.quiet()
    processed_frames: DataFrame = tp.batch(list(frames), feature_size, invert=True, minmass=minmass)
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

    def has_moved(group: DataFrame) -> bool:
        dx = group['x'].iloc[-1] - group['x'].iloc[0]
        dy = group['y'].iloc[-1] - group['y'].iloc[0]
        displacement = (dx**2 + dy**2)**0.5
        return displacement > 0.01

    #print(trajectories_present_in_all_frames)

    moving_trajectories: DataFrame = trajectories_present_in_all_frames.groupby('particle').filter(has_moved)

    return moving_trajectories

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

def getAnnotatedVideo(
    frames: Series,
    processed_frames: DataFrame,
    output_filename: str = 'annotated_output.mp4',
):
  """Generates and saves an annotated video with tracking circles on every frame."""
  if len(frames) == 0:
    return

  first_frame = frames[0]
  height, width = first_frame.shape[:2]

  fps = CAMERA_FRAME_RATE
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

  print(f'Generating annotated video: {output_filename}...')
  for i, frame_img in enumerate(frames):
    # Set figure size to match frame dimensions directly in inches (at 100 dpi)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # Remove all margins, axes, and white spaces completely
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Filter detections for the current frame
    frame_particles = processed_frames[processed_frames['frame'] == i]

    tp.annotate(
        frame_particles,
        frame_img,
        ax=ax,
        plot_style={'markersize': 5, 'markeredgewidth': 1},
    )

    fig.canvas.draw()
    img_buf = np.asarray(fig.canvas.buffer_rgba())
    frame_bgr = cv2.cvtColor(img_buf, cv2.COLOR_RGBA2BGR)

    if frame_bgr.shape[:2] != (height, width):
      frame_bgr = cv2.resize(frame_bgr, (width, height))

    out.write(frame_bgr)
    plt.close(fig)

  out.release()
  print(f'Annotated video saved successfully as {output_filename}.')

def getViscosityFromDiffusionParameter(temperature: float, particle_radius: float, diffusion_parameter: float) -> float:
    '''Calculates viscosity based on the Stokes-Einstein relation.'''
    return (Boltzmann * temperature) / (6 * np.pi * diffusion_parameter * particle_radius) * 1000

def calculateDiffusionParameterFromData(average_mean_squared_displacement: Series) -> tuple[float, float]:
    '''Uses the slope of the average mean squared displacement over time to calculate the Diffusion parameter. To get the slope a linear fit through the data is used.'''
    tau = average_mean_squared_displacement.index.to_numpy()
    msd_values = average_mean_squared_displacement.to_numpy() * (10**(-6))**2
    A = tau[:, np.newaxis]
    
    # Solve for slope 'm' with intercept fixed at 0
    slope, residuals, rank, singular = np.linalg.lstsq(A, msd_values, rcond=None)
    
    # slope is an array, take the first element
    D: float = slope[0] / 4
    offset = 0.0 # Intercept is strictly forced to 0
    return D, offset

def celsiusToKelvin(temperature: float) -> float:
    '''Converts temperature from degrees Celsius to Kelvins'''
    return temperature + 273.15

def trackedParticletracjectories(video_name: str, feature_size: int, minmass: int, frame_limit: int, show_mass_histogram: bool = False, annotate_frame_zero: bool = False, plot_drift: bool = False, print_filter: bool = False) -> tuple[DataFrame, int]:
    '''Wrapping all frame processing and particle tracking into one function that in the end returns the drift-corrected trajectories.'''
    frames: Series = loadFramesFromVideo(video_name)[:frame_limit]
    number_of_frames = len(frames)
    processed_frames = processFrames(frames, feature_size, minmass)

    #getAnnotatedVideo(frames, processed_frames, output_filename=f"annotated_{video_name.replace('.wmv', '.mp4')}")

    if show_mass_histogram or annotate_frame_zero:
        for i in range(1):
            f: DataFrame = tp.locate(frames[i], feature_size, invert=True, minmass=minmass)
            if show_mass_histogram: plotMassHistogram(f)
            if annotate_frame_zero: tp.annotate(f, frames[i])
    trajectories = calculateTrajectories(processed_frames, number_of_frames//8, print_filter)
    drift = calculateOverallDrift(trajectories, plot_drift)
    return calculateDriftCorrectedTrajectories(trajectories, drift), number_of_frames

def viscosityFromTrajectories(drift_corrected_trajectories: DataFrame, number_of_frames: int, temperature: float, particle_radius: float, show_amsd_plot: bool = False, show_trajectories : bool = False) -> float:
    '''Wrapping all mean squared dsiplacement, diffucion coefficient and viscosity calculation into one function that returns the calculated viscosity at the end.'''
    amsd: Series = tp.emsd(drift_corrected_trajectories, mpp=CALIBRATION_FACTOR, fps=CAMERA_FRAME_RATE, max_lagtime=number_of_frames//4)
    D, offset = calculateDiffusionParameterFromData(amsd)
    if show_amsd_plot:
        plt.figure()
        plt.subplot(121)
        plt.scatter(amsd.index, amsd.values, color='blue', s=20, label='Data points')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("MSD, time log log plot")
        plt.subplot(122)
        plt.scatter(amsd.index, amsd.values, color='green', s=20, label='Data points')
        plt.plot(amsd.index, (4*D*amsd.index + offset) * 10**12, color='red', linestyle='--', label='Linear fit')
        plt.title("MSD, time lienar plot")
        plt.legend()
        plt.show()
    if show_trajectories: plotTrajectories(drift_corrected_trajectories)
    viscosity = getViscosityFromDiffusionParameter(celsiusToKelvin(temperature), particle_radius * (10**(-6)), D)
    return viscosity

def calculateAllViscosities(video_data: list[dict[str, typing.Any]]) -> None:
    viscosities: np.ndarray = np.zeros(len(video_data))
    radii: np.ndarray = np.zeros(len(video_data))
    for i, data in enumerate(video_data):
        radii[i] = data["particle_diameter"]/2
        if VERBOSE:
            trajectories, frames_n = trackedParticletracjectories(data["video_file"], data["feature_size"], data["minmass"], data["frame_limit"], True, True, True, True)
            if len(trajectories) <= 2:
                viscosities[i] = 0
            else:
                viscosities[i] = viscosityFromTrajectories(trajectories, frames_n, TEMPERTURE, radii[i], True, True)
        else:
            trajectories, frames_n = trackedParticletracjectories(data["video_file"], data["feature_size"], data["minmass"], data["frame_limit"])
            if len(trajectories) <= 2:
                viscosities[i] = 0
            else:
                viscosities[i] = viscosityFromTrajectories(trajectories, frames_n, TEMPERTURE, radii[i])

        print(f'The viscosity of video {data["video_file"]} is {viscosities[i]} mPa/s')

    plt.figure()
    plt.scatter(1/radii, viscosities, marker="o")
    #plt.plot([0, 2.5], [0, 2.5])
    plt.title("viscosity against 1/radius")
    plt.show()

def main() -> None:
    '''Main function.'''
    calculateAllViscosities(VIDEO_PROCESSING_DATA)

if __name__ == "__main__":
    '''Module helper.'''
    main()