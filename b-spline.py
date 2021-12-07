points = [
    [0.957, 0.296],
    [0.625, 0.782],
    [0.075, 0.997],
    [-0.498, 0.867],
    [-0.899, 0.437],
    [-0.988, -0.146],
    [-0.734, -0.678],
    [-0.226, -0.976],
    [0.362, -0.933],
    [0.824, -0.566],
]


import numpy as np
import scipy.interpolate as si


def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree, 1, degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree, 1, count - 1)

    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0 - degree, count + degree + degree - 1, dtype='int')
    else:
        kv = np.concatenate(([0] * degree, np.arange(count - degree + 1), [count - degree] * degree))

    # Calculate query range
    u = np.linspace(periodic, (count - degree), n)

    # Calculate result
    return np.array(si.splev(u, (kv, cv.T, degree))).T


import matplotlib.pyplot as plt

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

cv = np.array(points)

plt.plot(cv[:, 0], cv[:, 1], 'o', label='Control points')

# for d in range(1, 2):
d = 2
p = bspline(cv, n=50, degree=d, periodic=True)
x, y = p.T
plt.plot(x, y, 'k-', label='B-spline', color=colors[d % len(colors)])

plt.minorticks_on()
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
