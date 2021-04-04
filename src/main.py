from spline import pose_to_cv, Spline
import numpy as np
import matplotlib.pyplot as plt


def main():
    cx0, cy0 = pose_to_cv(0, 0, 0, +5.0)
    cx1, cy1 = pose_to_cv(2, 2, 0, -5.0)

    spline = Spline(cx0, cx1, cy0, cy1)

    ts = []
    xs = []
    ys = []
    curvatures = []

    for t in np.linspace(0, 1, 50):
        x, y, _, curvature = spline.at(t)
        ts.append(t)
        xs.append(x)
        ys.append(y)
        curvatures.append(curvature)

    path = plt.figure(1)
    plt.plot(xs, ys)

    curv = plt.figure(2)
    plt.plot(ts, curvatures)

    plt.show()


if __name__ == '__main__':
    main()
