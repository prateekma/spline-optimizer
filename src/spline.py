import numpy as np


def pose_to_cv(x, y, rot, curvature=0):
    return np.array([[x, np.cos(rot), curvature]]).transpose(), np.array([[y, np.sin(rot), curvature]]).transpose()


class Spline:
    coefficients = np.zeros([8, 6])

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

        # 3rd Derivative
        for i in range(4):
            self.coefficients[6][i] = self.coefficients[4][i] * (3 - i)
            self.coefficients[7][i] = self.coefficients[5][i] * (3 - i)

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
            dddx = self.coefficients[6, 2]
            dddy = self.coefficients[7, 2]

        else:
            dx = combined[2][0] / t
            dy = combined[3][0] / t
            ddx = combined[4][0] / t / t
            ddy = combined[5][0] / t / t
            dddx = combined[6][0] / t / t / t
            dddy = combined[7][0] / t / t / t

        f = (dx * ddy - dy * ddx)
        g = (dx * dx + dy * dy) ** 1.5

        f_prime = dx * dddy + ddx * ddy - ddy * ddx - dy * dddx
        g_prime = 3 * np.sqrt(dx * dx + dy * dy) * (ddx * dx + ddy * dy)

        curvature = f / g
        dkdt = (g * f_prime - f * g_prime) / (g ** 2)

        # Return (x, y, rot, curvature, dkds)
        return combined[0][0], combined[1][0], np.arctan2(dy, dx), curvature, dkdt


def sum_dkdt_squared(spline):
    integrand = 0
    for t in np.linspace(0, 1, 100):
        _, _, _, _, dkdt = spline.at(t)
        integrand += dkdt ** 2 / 100

    return integrand
