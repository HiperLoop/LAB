import os
import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import pims
import trackpy as tp
from pandas import DataFrame, Series

SAVE_PATH: str = "./csv_data/"
VIDEO_PATH: str = './data/Adam Data/Video/'

PARTICLE_DIAMETER_RELATIVE_UNCERTAINTIES: dict[float, float] = {
    2: 0.02,
    3: 0.03,
    4: 0.01,
    6: 0.04,
    8: 0.02
}

VIDEO_DATA: list[dict[str, typing.Any]] = [
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

# Figure style definition
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

@pims.pipeline  # pyright: ignore[reportUntypedFunctionDecorator, reportAttributeAccessIssue]
def gray(image: Series) -> Series:
    '''Make image monochromatic by just taking the green channel in the RGB values.'''
    return image[:, :, 1]

def celsiusToKelvin(temperature: float) -> float:
    '''Converts temperature from degrees Celsius to Kelvins'''
    return temperature + 273.15

def loadFramesFromVideo(video_name: str) -> Series:
    '''Load all video frames and convert them to monochrome.'''
    return gray(pims.open(f'{VIDEO_PATH}{video_name}'))

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

def trackedParticletracjectories(video_name: str, feature_size: int, minmass: int, frame_limit: int, visible_frames_requirement: int, show_mass_histogram: bool = False, annotate_frame_zero: bool = False, plot_drift: bool = False, print_filter: bool = False) -> tuple[DataFrame, int]:
    '''Wrapping all frame processing and particle tracking into one function that in the end returns the drift-corrected trajectories.'''
    frames: Series = loadFramesFromVideo(video_name)[:frame_limit]
    number_of_frames = len(frames)
    processed_frames = processFrames(frames, feature_size, minmass)

    if show_mass_histogram or annotate_frame_zero:
        for i in range(1):
            f: DataFrame = tp.locate(frames[i], feature_size, invert=True, minmass=minmass)
            if show_mass_histogram: plotMassHistogram(f)
            if annotate_frame_zero: tp.annotate(f, frames[i])
    trajectories = calculateTrajectories(processed_frames, visible_frames_requirement, print_filter)
    drift = calculateOverallDrift(trajectories, plot_drift)
    return calculateDriftCorrectedTrajectories(trajectories, drift), number_of_frames

def generateCSV(file_name: str, video_file: str, particle_diameter: float, visible_frames: int, feature_size: int, minmass: int, temperature: float, mmp: float, mmp_u: float, fps: int, frames: int, trajectories: DataFrame):
    file_path = os.path.join(SAVE_PATH, f'{file_name.split(".")[0]}.csv')

    particel_diameter_u: float = PARTICLE_DIAMETER_RELATIVE_UNCERTAINTIES[int(particle_diameter * 4)]
    temperature = celsiusToKelvin(temperature)

    with open(file_path, 'w', newline='') as csvfile:
        csvfile.write("# ================================================================================================\n")
        csvfile.write(f"# This file contains the trajectories of tracked particles fromt the video: {video_file}.\n")
        csvfile.write(f"# The video has {frames} frames at {fps} fps.\n")
        csvfile.write("# Measurement data:\n")
        csvfile.write(f"# Particle diameter: {particle_diameter} +- {particel_diameter_u}%.\n")
        csvfile.write(f"# Temperature during measurement: {temperature} +- 0.1 K.\n")
        csvfile.write(f"# Micrometer to pixel calibration: {mmp} +- {mmp_u}.\n")
        csvfile.write("# Tracking parameters:\n")
        csvfile.write(f"# Feature size: {feature_size}.\n")
        csvfile.write(f"# Minmass: {minmass}.\n")
        csvfile.write(f"# Minimum number of frames a particle must be visible to be recorded: {visible_frames}.\n")
        csvfile.write("# ================================================================================================\n")
        csvfile.write("#\n")
        csvfile.write("frames, fps, particle_diameter, diameter_uncertainty, temperature, mmp, mmp_u, feature_size, minmass, visible_frames\n")
        csvfile.write(f"{frames}, {fps}, {particle_diameter}, {particel_diameter_u}, {temperature}, {mmp}, {mmp_u}, {feature_size}, {minmass}, {visible_frames}\n")
        csvfile.write("#\n")
        trajectories.to_csv(csvfile, index=False)
        csvfile.close()

def main() -> None:
    '''Main function.'''
    manual: bool = str(input("Manual? (Y/n)")).lower() != "n"
    if manual:
        VERBOSE = input("Plot intermediate graphs? (Y/n)").lower() != "n"
        particle_diameter: float = float(input("Particle diameter: "))
        temperature: float = float(input("temperature: "))
        mmp: float = 0.130
        mmp_u: float = 0.003
        camera_fps: int = 20
        frame_limit: int = 60
        video: str = input("Video file: ")
        feature_size: int = int(input("Feature size: "))
        minmass: int = int(input("Minmass: "))
        #visible_frames: int = int(input("How many frames does a particle have to appear for to record its trajectory: "))
        trajectories, frames = trackedParticletracjectories(video, feature_size, minmass, frame_limit, frame_limit//8, VERBOSE, VERBOSE, VERBOSE, VERBOSE)
        generateCSV(video, video, particle_diameter, frame_limit//8, feature_size, minmass, temperature, mmp, mmp_u, camera_fps, frames, trajectories)
    else:
        mmp: float = 0.130
        mmp_u: float = 0.003
        camera_fps: int = 20
        temperature: float = 20.9
        for i, data in enumerate(VIDEO_DATA):
            trajectories, frames = trackedParticletracjectories(data["video_file"], data["feature_size"], data["minmass"], data["frame_limit"], data["frame_limit"]//8)
            generateCSV(data["video_file"], data["video_file"], data["particle_diameter"], data["frame_limit"]//8, data["feature_size"], data["minmass"], temperature, mmp, mmp_u, camera_fps, frames, trajectories)
            print(f"CSV {i+1} / {len(VIDEO_DATA)} generated.")

if __name__ == "__main__":
    '''Module helper.'''
    main()