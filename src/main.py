import typing

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
from pandas import DataFrame, Series
from scipy.constants import Boltzmann
from scipy.optimize import curve_fit

CALIBRATION_FACTOR: float = 0.130                       # [mpp] Determined using Fiji
CALIBRATION_ABSOLUTE_UNCERTAINTY : float= 0.003
CALIBRATION_RELATIVE_UNCERTAINTY: float = CALIBRATION_ABSOLUTE_UNCERTAINTY/CALIBRATION_FACTOR

TRACKPY_RELATIVE_UNCERTAINTY: float = 1/16              # In partice radius

CAMERA_FRAME_RATE: int = 20                             # Frames per second

TEMPERTURE: float = 20.9                                # Temperature in degrees Celsius during video collection
TEMPERATURE_ABSOLUTE_UNCERTAINTY: float = 0.1           # Absolute temperature uncertainty

PARTICLE_DIAMETER_RELATIVE_UNCERTAINTIES: dict[float, float] = {
    2: 0.02,
    3: 0.03,
    4: 0.01,
    6: 0.04,
    8: 0.02
}

VERBOSE: bool = False                                    # Determines whether intermediate steps are printed in console and figures shown

def getViscosityFromDiffusionParameter(temperature: float, particle_radius: float, diffusion_parameter: float) -> float:
    '''Calculates viscosity based on the Stokes-Einstein relation.'''
    return (Boltzmann * temperature) / (6 * np.pi * diffusion_parameter * particle_radius) * 1000

def calculateDiffusionParameterFromData(average_mean_squared_displacement: Series, average_mean_squared_displacement_uncertainty: Series) -> tuple[float, float, float]:
    '''Uses the slope of the average mean squared displacement over time to calculate the Diffusion parameter. To get the slope a linear fit through the data is used.'''
    tau = average_mean_squared_displacement.index.to_numpy()
    msd_values = average_mean_squared_displacement.to_numpy() * (10**(-6))**2
    
    def line(x: float, m : float) -> float:
        return x * m
    
    popt, pcov = curve_fit(line, tau, msd_values, sigma=(average_mean_squared_displacement_uncertainty.to_numpy()* (10**(-6))**2), absolute_sigma=True)
    
    # slope is an array, take the first element
    D: float = popt[0] / 4
    offset = 0.0 # Intercept is strictly forced to 0
    return D, offset, (np.sqrt(np.diag(pcov))[0]/4)

def celsiusToKelvin(temperature: float) -> float:
    '''Converts temperature from degrees Celsius to Kelvins'''
    return temperature + 273.15

def calculateVicosityUncertainty(viscosity: float, diffusion_coeficient: float, diffusion_absolute_uncertainty: float, particle_diameter: float) -> float:
    relative_temperature_uncertainty: float = TEMPERATURE_ABSOLUTE_UNCERTAINTY / celsiusToKelvin(TEMPERTURE)
    relative_diffusion_uncertainty: float = diffusion_absolute_uncertainty / diffusion_coeficient
    relative_particle_radius_uncertainty: float = 0.5 * PARTICLE_DIAMETER_RELATIVE_UNCERTAINTIES[int(particle_diameter * 4)]
    return viscosity * np.sqrt(relative_diffusion_uncertainty**2 + relative_particle_radius_uncertainty**2 + relative_temperature_uncertainty**2)


def viscosityFromTrajectories(drift_corrected_trajectories: DataFrame, number_of_frames: int, temperature: float, particle_diameter: float, show_amsd_plot: bool = False, show_trajectories : bool = False) -> tuple[float, float]:
    '''Wrapping all mean squared dsiplacement, diffucion coefficient and viscosity calculation into one function that returns the calculated viscosity at the end.'''
    amsd: Series = tp.emsd(drift_corrected_trajectories, mpp=CALIBRATION_FACTOR, fps=CAMERA_FRAME_RATE, max_lagtime=number_of_frames//4)
    amsd_uncertainty: Series = amsd*np.sqrt(CALIBRATION_RELATIVE_UNCERTAINTY**2 + (1/16)**2)
    D, offset, D_uncertainty = calculateDiffusionParameterFromData(amsd, amsd_uncertainty)
    if show_amsd_plot:
        plt.figure()
        plt.subplot(121)
        plt.errorbar(amsd.index, amsd.values, amsd_uncertainty, color='blue', label='Data points', linestyle='')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("MSD, time log log plot")
        plt.subplot(122)
        plt.errorbar(amsd.index, amsd.values, amsd_uncertainty, color='green', label='Data points', linestyle='')
        plt.plot(amsd.index, (4*D*amsd.index + offset) * 10**12, color='red', linestyle='--', label=f'Linear fit with resulting D = {D} +- {D_uncertainty}')
        plt.title("MSD, time lienar plot")
        plt.legend()
        plt.show()
    if show_trajectories: plotTrajectories(drift_corrected_trajectories)
    viscosity = getViscosityFromDiffusionParameter(celsiusToKelvin(temperature), particle_diameter* 0.5 * (10**(-6)), D)
    viscosity_uncertainty: float = calculateVicosityUncertainty(viscosity, D, D_uncertainty, particle_diameter)
    return viscosity, viscosity_uncertainty

def calculateAllViscosities(video_data: list[dict[str, typing.Any]]) -> None:
    viscosities: np.ndarray = np.zeros(len(video_data))
    viscosity_uncertainties: np.ndarray = np.zeros(len(video_data))
    radii: np.ndarray = np.zeros(len(video_data))
    for i, data in enumerate(video_data):
        radii[i] = data["particle_diameter"]/2
        if VERBOSE:
            trajectories, frames_n = trackedParticletracjectories(data["video_file"], data["feature_size"], data["minmass"], data["frame_limit"], True, True, True, True)
            if len(trajectories) <= 2:
                viscosities[i] = 0
            else:
                viscosities[i], viscosity_uncertainties[i] = viscosityFromTrajectories(trajectories, frames_n, TEMPERTURE, data["particle_diameter"], True, True)
        else:
            trajectories, frames_n = trackedParticletracjectories(data["video_file"], data["feature_size"], data["minmass"], data["frame_limit"])
            if len(trajectories) <= 2:
                viscosities[i] = 0
            else:
                viscosities[i], viscosity_uncertainties[i] = viscosityFromTrajectories(trajectories, frames_n, TEMPERTURE, data["particle_diameter"])

        print(f'The viscosity of video {data["video_file"]} is {viscosities[i]} +- {viscosity_uncertainties[i]} mPa/s')

    plt.figure()
    plt.errorbar(1/radii, viscosities, viscosity_uncertainties, marker="o", linestyle='')
    #plt.plot([0, 2.5], [0, 2.5])
    plt.title("Viscosity calculations for differently sized particles")
    plt.xlabel("1/(particle radius)")
    plt.ylabel("Viscosity")
    plt.show()

def main() -> None:
    '''Main function.'''
    calculateAllViscosities(VIDEO_PROCESSING_DATA)

if __name__ == "__main__":
    '''Module helper.'''
    main()