import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

BASELINE = 2.3

def create_df(filename):
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


def part2():
    df = pd.read_csv('data/part2.csv')
    df['theta_rad'] = np.radians(df['theta'])

    f = lambda theta, i0, phi: BASELINE + i0*((np.cos(theta+phi))**2)

    p, c = opt.curve_fit(f, df['theta_rad'], df['i_r'], p0=[200, 1])

    x = pd.Series(np.arange(0, 2*np.pi, 0.01))
    y = f(x, p[0], p[1])

    df['y_fit'] = f(df['theta_rad'], p[0], p[1])
    df['y_err'] = 0.05 * df['i_r']

    df['res'] = df['y_fit']-df['i_r']

    chi_sq = 0
    for i in range(len(df['theta_rad'])):
        chi_sq += ((df.iloc[i,1]-df.iloc[i,3])**2)/(df.iloc[i,4]**2)
    chi_sq = chi_sq/len(df.iloc[:,1])


    plt.ion()
    plt.errorbar(df['theta_rad'], df['i_r'], df['y_err'], np.radians(2), fmt='.', capsize=2)
    plt.plot(x, y)
    plt.ylabel('Current [mA]')
    plt.xlabel('Angle [rad]')
    plt.title("Two Polarizers, Malus' Law")
    plt.xlim(0.5, 4.5)
    plt.ioff()
    plt.show()

    plt.scatter(df['theta_rad'], df['res'])
    plt.title('residuals')
    plt.xlabel('Angle [rad]')
    plt.ylabel('Current difference [mA]')
    plt.show()

def part3():
    df = pd.read_csv('data/part3.csv')
    df['theta_rad'] = np.radians(df['theta'])

    f = lambda theta, i0, phi: BASELINE + 0.25*i0*((np.sin(2*theta+phi))**2)

    p, c = opt.curve_fit(f, df['theta_rad'], df['i_r'], p0=[35, 1])

    x = pd.Series(np.arange(0, 2*np.pi, 0.01))
    y = f(x, p[0], p[1])

    df['y_fit'] = f(df['theta_rad'], p[0], p[1])
    df['y_err'] = 0.05 * df['i_r']

    df['res'] = df['y_fit']-df['i_r']

    chi_sq = 0
    for i in range(len(df['theta_rad'])):
        chi_sq += ((df.iloc[i,1]-df.iloc[i,3])**2)/(df.iloc[i,4]**2)
    chi_sq = chi_sq/len(df.iloc[:,1])


    plt.ion()
    plt.errorbar(df['theta_rad'], df['i_r'], df['y_err'], np.radians(2), fmt='.', capsize=2)
    plt.plot(x, y)
    plt.ylabel('Current [mA]')
    plt.xlabel('Angle [rad]')
    plt.title("Three Polarizers, Malus' Law")
    plt.xlim(1, 5.5)
    plt.ioff()
    plt.show()

    plt.scatter(df['theta_rad'], df['res'])
    plt.title('residuals')
    plt.xlabel('Angle [rad]')
    plt.ylabel('Current difference [mA]')
    plt.show()


def part4():
    df = pd.read_csv('data/part4half_wave.csv')
    df['theta_rad'] = np.radians(df['theta'])

    f = lambda theta, i0, phi: BASELINE + 0.25 * i0 * ((np.sin(2 * theta + phi)) ** 2)

    p, c = opt.curve_fit(f, df['theta_rad'], df['i_r'], p0=[35, 1])

    x = pd.Series(np.arange(0, 2 * np.pi, 0.01))
    y = f(x, p[0], p[1])

    df['y_fit'] = f(df['theta_rad'], p[0], p[1])
    df['y_err'] = 0.05 * df['i_r']

    df['res'] = df['y_fit'] - df['i_r']

    chi_sq = 0
    for i in range(len(df['theta_rad'])):
        chi_sq += ((df.iloc[i, 1] - df.iloc[i, 3]) ** 2) / (df.iloc[i, 4] ** 2)
    chi_sq = chi_sq / len(df.iloc[:, 1])

    plt.ion()
    plt.errorbar(df['theta_rad'], df['i_r'], df['y_err'], np.radians(2), fmt='.', capsize=2)
    plt.plot(x, y)
    plt.ylabel('Current [mA]')
    plt.xlabel('Angle of half wave plate[rad]')
    plt.title("Polarizer - Half wave plate - Polarizer")
    plt.xlim(0.5, 5.5)
    plt.ioff()
    plt.show()

    plt.scatter(df['theta_rad'], df['res'])
    plt.title('residuals')
    plt.xlabel('Angle [rad]')
    plt.ylabel('Current difference [mA]')
    plt.show()


def faraday():
    D = 13e-3
    BASELINE = 14.9

    df = pd.read_csv('data/faraday.csv')
    df.drop(df.index[:2], inplace=True)
    # f = lambda b, i0, v, : BASELINE+i0*4*((b-4.86)**2)*(v**2)*(D**2)

    f = lambda b, b0, a, c: a*(b-b0)**2 + c

    p, c = opt.curve_fit(f, df['b'], df['i'])

    x = pd.Series(np.arange(0, 10, 0.01))
    y = f(x, p[0], p[1], p[2])

    df['y_fit'] = f(df['b'], p[0], p[1], p[2])
    df['y_err'] = 0.005 * df['i']

    df['res'] = df['y_fit'] - df['i']

    chi_sq = 0
    for i in range(len(df['b'])):
        chi_sq += ((df.iloc[i,1] - df.iloc[i,2]) ** 2) / (df.iloc[i,3] ** 2)
    chi_sq = chi_sq / len(df['b'])

    plt.ion()
    plt.errorbar(df['b'], df['i'], df['y_err'], fmt='.', capsize=2)
    plt.plot(x, y)
    plt.ylabel('Current [nanoA]')
    plt.xlabel('B[mT]')
    plt.title("Faraday effect : polarizer - magnetic field - polarizer")
    # plt.xlim(0.5, 5.5)
    plt.ioff()
    plt.show()

    plt.scatter(df['b'], df['res'])
    plt.title('residuals')
    plt.xlabel('B [mT]')
    plt.ylabel('Current difference [nanoA]')
    plt.show()


def part_5_parallel():
    BASELINE = 2.3
    df = pd.read_csv('data/part5_parallel.csv')
    df.drop(df.index[20], inplace=True)
    df.drop(df.index[3], inplace=True)

    df['theta_rad'] = np.radians(df['theta'])

    f = lambda theta, i0, n2: BASELINE + i0*(np.tan(theta-np.arcsin(np.sin(theta)/n2)))**2/\
                              ((np.tan(theta+np.arcsin(np.sin(theta)/n2)))**2)

    df['y_err'] = 0.05 * df['i_r']

    p, c = opt.curve_fit(f, df['theta_rad'], df['i_r'], p0=[35, 1])

    x = pd.Series(np.arange(0, np.pi/2, 0.01))
    y = f(x, p[0], p[1])

    y -= BASELINE
    df['i_r'] -= BASELINE

    df['y_fit'] = f(df['theta_rad'], p[0], p[1])


    df['res'] = df['y_fit'] - df['i_r']

    chi_sq = 0
    for i in range(len(df['theta_rad'])):
        chi_sq += ((df.iloc[i, 1] - df.iloc[i, 3]) ** 2) / (df.iloc[i, 4] ** 2)
    chi_sq = chi_sq / len(df.iloc[:, 1])

    plt.ion()
    plt.errorbar(df['theta_rad'], df['i_r'], df['y_err'], np.radians(2), fmt='.', capsize=2)
    plt.plot(x, y)
    plt.ylabel('Current [mA]')
    plt.xlabel('Incidence Angle[rad]')
    plt.title("Dielectric medium reflectance - parallel")
    # plt.xlim(0.5, 5.5)
    plt.ylim(0,100)
    plt.ioff()
    plt.show()

    plt.scatter(df['theta_rad'], df['res'])
    plt.title('residuals')
    plt.xlabel('Incidence Angle [rad]')
    plt.ylabel('Current difference [mA]')
    plt.show()


def part_5_ortho():
    BASELINE = 2.3
    df = pd.read_csv('data/part5_ortho.csv')

    df['theta_rad'] = np.radians(df['theta'])

    df.drop(df.index[17], inplace=True)
    df.drop(df.index[0], inplace=True)

    f = lambda theta, i0, n2: BASELINE + i0*(np.sin(theta-np.arcsin(np.sin(theta)/n2)))**2/\
                              ((np.sin(theta+np.arcsin(np.sin(theta)/n2)))**2)

    df['y_err'] = 0.05 * df['i_r']

    p, c = opt.curve_fit(f, df['theta_rad'], df['i_r'], p0=[35, 1.4])

    x = pd.Series(np.arange(0, np.pi/2, 0.01))
    y = f(x, p[0], p[1])

    df['y_fit'] = f(df['theta_rad'], p[0], p[1])

    df['res'] = df['y_fit'] - df['i_r']

    y -= BASELINE
    df['y_fit'] -= BASELINE
    df['i_r'] -= BASELINE

    chi_sq = 0
    for i in range(len(df['theta_rad'])):
        chi_sq += ((df.iloc[i, 1] - df.iloc[i, 3]) ** 2) / (df.iloc[i, 4] ** 2)
    chi_sq = chi_sq / len(df.iloc[:, 1])

    plt.ion()
    plt.errorbar(df['theta_rad'], df['i_r'], df['y_err'], np.radians(2), fmt='.', capsize=2)
    plt.plot(x, y)
    plt.ylabel('Current [mA]')
    plt.xlabel('Incidence Angle[rad]')
    plt.title("Dielectric medium reflectance - orthogonal")
    plt.ylim(0,125)
    plt.ioff()
    plt.show()

    plt.scatter(df['theta_rad'], df['res'])
    plt.title('residuals')
    plt.xlabel('Incidence Angle [rad]')
    plt.ylabel('Current difference [mA]')
    plt.show()


def get_n_sigma(y1,y2,dy1,dy2=0):
    return (y1-y2)**2/((dy1+dy2)**2)
