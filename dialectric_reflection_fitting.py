import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import *
import scipy as sci
import scipy.optimize as opt

N_AIR = 1
BASE_INTENSITY = 2.0


def load_intensity_data(filename):
    """
    Returns the intensity data with inc_angle in radians.
    Also drops rows with NaN
    """
    data = pd.read_csv(filename)
    data = data.dropna(axis='index')
    data['inc_angle'] = np.radians(data['inc_angle'])
    data = data.astype('float64')
    data = data[data['inc_angle'] <= np.deg2rad(80)]
    return data


def get_trans_angle(inc_angle, n1, n2):
    """
    Returns the transmittance angle in radians. inc_angle should be in radians as well.
    """
    return asin(n1*sin(inc_angle)/n2)


def ortho_reflected_intensity(inc_angle, incidence_intensity, n2):
    """
    Returns the expected reflected intensity for orthogonal polarization
    :param inc_angle: the incidence angle in radians
    :param incidence_intensity: the incidence intensity (usually arbitrary)
    :param n2: the refecttion coefficient of the dielectric medium
    :return: the expected measured intensity.
    """
    trans_angle = np.arcsin(N_AIR * np.sin(inc_angle) / n2)
    r_ortho = ((np.sin(inc_angle - trans_angle))**2)/((np.sin(inc_angle + trans_angle))**2)
    expected_intensity = r_ortho * incidence_intensity + BASE_INTENSITY
    return expected_intensity


def fit_ortho_reflected_intensity(inc_angle_vec, intensity_vec):
    """Gets the fitting parameters and covariance for orthogonal polarization reflection."""
    params, cov = opt.curve_fit(ortho_reflected_intensity, inc_angle_vec, intensity_vec, p0=[100, 1])
    return params, cov


def parallel_reflected_intensity(inc_angle, incidence_intensity, n2):
    """
    Takes a vector of incidence angles as radians, and two scalars, and returns a vector of
    the expected reflectance intensity rate for parallel-polarized beams.
    :param inc_angle: a vector of incidence angles.
    :param incidence_intensity: the intensity of incidence light, as a scalar.
    :param n2: the index of refraction of the medium, as a scalar
    :return: a vector of expected intensities.
    """
    trans_angle = np.arcsin(N_AIR * np.sin(inc_angle) / n2)
    r_parallel = ((np.tan(inc_angle - trans_angle)) ** 2) / ((np.tan(inc_angle + trans_angle)) ** 2)
    expected_intensity = r_parallel * incidence_intensity + BASE_INTENSITY
    return expected_intensity


def fit_parallel_reflected_intensity(inc_angle_vec, intensity_vec):
    """Gets the fitting parameters and covariance for parallel polarization reflection."""
    params, cov = opt.curve_fit(parallel_reflected_intensity, inc_angle_vec, intensity_vec, p0=[100, 1])
    return params, cov


def arbitrary_polarization_reflected_intensity(inc_angle, incidence_intensity, n2, polarization_angle):
    """
    Returns the expected intensity for arbitrary polarization, given the rate of orthogonal intensity in the
    incidence beam.
    :param inc_angle: a vector of incidence angles in radians.
    :param incidence_intensity: the intensity of incidence light.
    :param n2: the index of refraction of the medium.
    :param polarization_angle: the angle of polarization of the incidence light from the optical plane.
    :return: a vector of reflected intensities.
    """
    trans_angle = np.arcsin(N_AIR * np.sin(inc_angle) / n2)
    r_parallel = ((np.tan(inc_angle - trans_angle)) ** 2) / ((np.tan(inc_angle + trans_angle)) ** 2)
    r_ortho = ((np.sin(inc_angle - trans_angle))**2)/((np.sin(inc_angle + trans_angle))**2)
    parallel_rate = np.cos(polarization_angle) ** 2
    expected_intensity = (parallel_rate * r_parallel * incidence_intensity)\
        + ((1 - parallel_rate) * r_ortho * incidence_intensity) + r_ortho + BASE_INTENSITY
    return expected_intensity


def fit_aribtrary_polarization_reflected_intensity(inc_angle_vec, intensity_vec):
    """
    Fits the data into a function and finds n2, I0, and the polarization angle from the optical plane.
    intensity in the incidence light.
    :param inc_angle_vec: a vector of incidence angles in radians.
    :param intensity_vec: a vector of measured intensities.
    :return: the fitting parameters, including polarization angle in radians.
    """
    params, cov = opt.curve_fit(arbitrary_polarization_reflected_intensity, inc_angle_vec, intensity_vec,
                                p0=[100, 1, pi/4])
    return params, cov


def predict_parallel_intensity_using_ortho_params(ortho_params, parallel_data):
    rads_axis = pd.Series(np.arange(0, pi / 2, 0.001))
    expected_intensity = rads_axis.apply(parallel_reflected_intensity, args=(ortho_params[0], ortho_params[1]))
    plt.figure()
    plt.suptitle(
        "Measured and expected intensity for parallel polarization - using n2 of "
        + str(round(ortho_params[1], 4)))
    plt.xlabel("Incidence Angle (rad)")
    plt.ylabel("Intensity (micro Amps)")
    plt.scatter(parallel_data['inc_angle'], parallel_data['intensity'])
    plt.plot(rads_axis, expected_intensity)
    plt.show()


def predict_reflected_intensity(data, parallel_only = False, ortho_only = False):
    """
    Returns the predicted parameters for the fitted reflectance of orthogonally polarized beams,
    and presents a fitted plot.
    :param data: The data for measured intensity per incidence angle.
    :return: the fitting parameters and covariances.
    """
    if parallel_only and not ortho_only:
        params, cov = fit_parallel_reflected_intensity(data['inc_angle'], data['intensity'])
        pol_angle = 0
    elif ortho_only and not parallel_only:
        params, cov = fit_ortho_reflected_intensity(data['inc_angle'], data['intensity'])
        pol_angle = pi/2
    else:
        params, cov = fit_aribtrary_polarization_reflected_intensity(data['inc_angle'], data['intensity'])
        pol_angle = params[2]
    print("Predicted n2 is: " + str(params[1]))
    print("Predicted polarization angle is: " + str(np.rad2deg(pol_angle)))
    print("Covariance matrix is: \n" + str(cov))
    rads_axis = pd.Series(np.arange(0, pi/2, 0.001))
    expected_intensity = rads_axis.apply(arbitrary_polarization_reflected_intensity,
                                         args=(params[0], params[1], pol_angle))
    plt.figure()
    plt.suptitle("Measured and expected intensity - estimated n2 is " \
                 + str(round(params[1], 4)))
    plt.xlabel("Incidence Angle (rad)")
    plt.ylabel("Intensity (micro Amps)")
    plt.scatter(data['inc_angle'], data['intensity'], s=3, c='red')
    plt.plot(rads_axis, expected_intensity)
    plt.show()
    return params


if __name__ == '__main__':
    ortho_data = load_intensity_data('./data/dielectric_reflectance_orthogonal_data_only.csv')
    parallel_data = load_intensity_data('./data/dielectric_reflectance_parallel_data_only.csv')
    ortho_params = predict_reflected_intensity(parallel_data, parallel_only=True)
