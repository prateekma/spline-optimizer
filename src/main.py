from spline import pose_to_cv, Spline
import numpy as np
import matplotlib.pyplot as plt


def main():
    cx0, cy0 = pose_to_cv(0, 0, 0)
    cx1, cy1 = pose_to_cv(2, 2, 0)

    spline1 = Spline(cx0, cx1, cy0, cy1)

    ts1 = []
    xs1 = []
    ys1 = []
    ks1 = []

    for t in np.linspace(0, 1, 50):
        x1, y1, _, k1, _ = spline1.at(t)

        ts1.append(t)
        xs1.append(x1)
        ys1.append(y1)
        ks1.append(k1)

    plt.figure(1)
    plt.plot(xs1, ys1)

    plt.figure(2)
    plt.plot(ts1, ks1)

    plt.show()


if __name__ == '__main__':
    main()
