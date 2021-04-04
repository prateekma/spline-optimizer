import numpy as np


def pose_to_cv(x, y, rot, curvature=0):
    return np.array([[x, np.cos(rot), curvature]]).transpose(), np.array([[y, np.sin(rot), curvature]]).transpose()


class Spline:
    coefficients = np.zeros([6, 6])

    def __init__(self, cx0, cx1, cy0, cy1):
        basis = np.array(
            [[-06.0, -03.0, -00.5, +06.0, -03.0, +00.5],
             [+15.0, +08.0, +01.5, -15.0, +07.0, -01.0],
             [-10.0, -06.0, -01.5, +10.0, -04.0, +00.5],
             [+00.0, +00.0, +00.5, +00.0, +00.0, +00.0],
             [+00.0, +01.0, +00.0, +00.0, +00.0, +00.0],
             [+01.0, +00.0, +00.0, +00.0, +00.0, +00.0]])

        x = np.concatenate((cx0, cx1))
        y = np.concatenate((cy0, cy1))

        # Build out rows of the coefficient matrix.
        self.coefficients[0, 0:6] = (basis @ x).transpose()
        self.coefficients[1, 0:6] = (basis @ y).transpose()

        # 1st Derivative
        for i in range(6):
            self.coefficients[2][i] = self.coefficients[0][i] * (5 - i)
            self.coefficients[3][i] = self.coefficients[1][i] * (5 - i)

        # 2nd Derivative
        for i in range(5):
            self.coefficients[4][i] = self.coefficients[2][i] * (4 - i)
            self.coefficients[5][i] = self.coefficients[3][i] * (4 - i)

    def at(self, t):
        bases = np.empty([6, 1])
        for i in range(6):
            bases[i][0] = t ** (5 - i)

        # This simply multiplies by the coefficients.
        combined = self.coefficients @ bases

        # If t = 0, all other terms in the equal cancel out to zero. We can use the last x^0 term in the
        # equation.
        if t == 0:
            dx = self.coefficients[2, 4]
            dy = self.coefficients[3, 4]
            ddx = self.coefficients[4, 3]
            ddy = self.coefficients[5, 3]

        else:
            dx = combined[2][0] / t
            dy = combined[3][0] / t
            ddx = combined[4][0] / t / t
            ddy = combined[5][0] / t / t

        # Find curvature.
        curvature = (dx * ddy - ddx * dy) / ((dx * dx + dy * dy) * np.hypot(dx, dy))

        # Return (x, y, rot, curvature)
        return combined[0][0], combined[1][0], np.arctan2(dx, dy), curvature
