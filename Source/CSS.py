
import cv2
import numpy as np
import GeomBase
import Polyline
from scipy.signal import savgol_filter


def get_gaussian_kernel(sigma, M):
    '''
    refer to https://cedar.buffalo.edu/~srihari/CSE555/Normal2.pdf
    '''
    L = int((M - 1) / 2);
    sigma_sq = sigma * sigma;
    sigma_quad = sigma_sq * sigma_sq;
    dg = np.zeros([M])
    d2g = np.zeros([M])
    gaussian = np.zeros([M])

    g = cv2.getGaussianKernel(M, sigma)
    g.reshape(len(g))

    for i in range(2 * L + 1):
        x = i - L
        gaussian[i] = g[i]
        dg[i] = (-x / sigma_sq) * g[i]
        d2g[i] = (-sigma_sq + x * x) / sigma_quad * g[i]
    return gaussian.reshape((1, M)), dg.reshape((1, M)), d2g.reshape((1, M))


def smooth_curve(curve, sigma, is_open=False, m=2):
    '''
    By refer to https://www.sciencedirect.com/science/article/pii/S0031320301000401
    According to the properties of convolution, the derivatives of g(x) * u(x)
    can be calculated easily computed by g'(x) * u(x)
    '''
    M = round((m * sigma + 1.0) / 2.0) * 2 - 1;
    assert (M % 2 == 1)
    g, dg, d2g = get_gaussian_kernel(sigma, M)
    X = curve[:, 0]
    Y = curve[:, 1]

    id_list = list(range(M))
    id_list = list(np.array(id_list) - int(M / 2))
    id_list.reverse()
    xM = np.zeros((M, len(X)))
    yM = np.zeros((M, len(X)))
    row = 0
    for i in id_list:
        xM[row] = np.roll(X, i)
        yM[row] = np.roll(Y, i)
        row += 1
    if (is_open):
        # todo: regulate xM, yM
        i = 0

    gX = np.vstack((np.sum(g.dot(xM), axis=0), np.sum(g.dot(yM), axis=0)))

    dX = np.sum(dg.dot(xM), axis=0)
    dY = np.sum(dg.dot(yM), axis=0)
    dXX = np.sum(d2g.dot(xM), axis=0)
    dYY = np.sum(d2g.dot(yM), axis=0)

    return gX.T, dX, dY, dXX, dYY


def compute_curve_css(curve, sigma, is_open=False, m = 2):
    '''
    example:
       kappa, smooth = compute_curve_css(c,3)
    '''

    smooth, dX, dY, dXX, dYY = smooth_curve(curve, sigma, m)

    kappa = (dX * dYY - dXX * dY) / np.power(dX * dX + dY * dY, 1.5);

    return kappa, smooth


def find_css_point(kappa):
    idx_list = []
    for i in range(len(kappa) - 1):
        if (kappa[i] < 0 and kappa[i + 1] > 0) or (kappa[i] > 0 and kappa[i + 1] < 0):
            idx_list.append(i)
    return idx_list


def polygon_To_Array(polygon):
    list_X = []
    list_Y = []
    for point in polygon.points:
        list_X.append(point.x)
        list_Y.append(point.y)
    X = np.array([list_X])
    Y = np.array([list_Y])
    Curve = np.vstack((X, Y))
    Curve = Curve.T
    return Curve
    #curve = np.

def Array_To_polygon(Curve,z):
    polygon = Polyline.Polyline()
    list_X = list(Curve[:, 0])
    list_Y = list(Curve[:, 1])
    for i in range(len(list_X)):
        polygon.points.append(GeomBase.Point3D(list_X[i], list_Y[i], z))
    return polygon



