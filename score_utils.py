from typing import Tuple
import numpy as np
import medpy.metric as metric


def calculate_metric_percase(pred, gt):

    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd(pred, gt)
        asd = metric.binary.assd(pred, gt)

        return dice, hd95, asd
    # handling
    elif gt.sum() == 0 and pred.sum() == 0:
        return 1, 0, 0
    else:
        # no information on how to handle missing predictions for hd5 and asd, should not have a maximum value?
        return 0, 0, 0


def find_tangent_lines(
        center: Tuple[float, float],
        semi_axes: Tuple[float, float],
        rotation: float,
        reference_point: Tuple[float, float],
):
    """Find the Ellipse's two tangents that go through a reference point.

    Args:
        center: The center of the ellipse.
        semi_axes: The semi-major and semi-minor axes of the ellipse.
        rotation: The counter-clockwise rotation of the ellipse in radians.
        reference_point: The coordinates of the reference point.

    Returns:
        (m1, h1): Slope and intercept of the first tangent.
        (m2, h2): Slope and intercept of the second tangent.
    """
    x0, y0 = center
    a, b = semi_axes
    s, c = np.sin(rotation), np.cos(rotation)
    p0, q0 = reference_point

    A = (-a ** 2 * s ** 2 - b ** 2 * c ** 2 + (y0 - q0) ** 2)
    B = 2 * (c * s * (a ** 2 - b ** 2) - (x0 - p0) * (y0 - q0))
    C = (-a ** 2 * c ** 2 - b ** 2 * s ** 2 + (x0 - p0) ** 2)

    if B ** 2 - 4 * A * C < 0:
        raise ValueError('Reference point lies inside the ellipse')

    t1, t2 = (
        (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A),
        (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A),
    )
    return (
        (1 / t1, q0 - p0 / t1),
        (1 / t2, q0 - p0 / t2),
    )


def get_major_axis_coord(ellipse):
    (xc, yc), (d1, d2), angle = ellipse

    rmajor = max(d1, d2) / 2

    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    x1 = xc + math.cos(math.radians(angle)) * rmajor
    y1 = yc + math.sin(math.radians(angle)) * rmajor
    x2 = xc + math.cos(math.radians(angle + 180)) * rmajor
    y2 = yc + math.sin(math.radians(angle + 180)) * rmajor
    return x1, y1, x2, y2


def get_minor_axis_coord(ellipse):
    # draw minor axis line in blue
    (xc, yc), (d1, d2), angle = ellipse
    rminor = min(d1, d2) / 2

    x1 = xc + math.cos(math.radians(angle)) * rminor
    y1 = yc + math.sin(math.radians(angle)) * rminor
    x2 = xc + math.cos(math.radians(angle + 180)) * rminor
    y2 = yc + math.sin(math.radians(angle + 180)) * rminor

    return x1, y1, x2, y2


import math


def dot(vA, vB):
    return vA[0] * vB[0] + vA[1] * vB[1]


def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
    vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA) ** 0.5
    magB = dot(vB, vB) ** 0.5
    # Get cosine value
    cos_ = dot_prod / magA / magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod / magB / magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle) % 360

    if ang_deg - 180 >= 0:
        # As in if statement
        return 360 - ang_deg
    else:

        return ang_deg
