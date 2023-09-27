import math
import cv2
import numpy as np
from score_utils import find_tangent_lines, get_minor_axis_coord, get_major_axis_coord, ang
import matplotlib.pyplot as plt


def angle_of_progression_estimation(label, return_vis_img=False):
    tmp_img = np.zeros_like(label).astype(np.uint8)

    contours, _ = cv2.findContours((label == 1).astype(np.uint8), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_i = 0
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_i = idx
            max_area = area

    ellipse = cv2.fitEllipse(contours[max_i])

    if return_vis_img:
        cv2.ellipse(tmp_img, ellipse, (120, 255, 255), 1, cv2.LINE_AA)

    x1, y1, x2, y2 = get_major_axis_coord(ellipse)

    ps_points = [(x1, y1), (x2, y2)] if x1 > x2 else [(x2, y2), (x1, y1)]

    if return_vis_img:
        cv2.line(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), (160, 0, 255), 2)

    contours_fh, _ = cv2.findContours((label == 2).astype(np.uint8), cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_i = 0
    for idx, cnt in enumerate(contours_fh):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_i = idx
            max_area = area

    ellipse_fh = cv2.fitEllipse(contours_fh[max_i])

    if return_vis_img:
        cv2.ellipse(tmp_img, ellipse_fh, (160, 255, 255), 1, cv2.LINE_AA)

    x1_fh, y1_fh, x2_fh, y2_fh = get_minor_axis_coord(ellipse_fh)

    if return_vis_img:
        cv2.line(tmp_img, (int(x1_fh), int(y1_fh)), (int(x2_fh), int(y2_fh)), (160, 0, 255), 2)

    (xc, yc), (d1, d2), angle = ellipse_fh

    CENTER = (xc, yc)

    SEMI_AXES = (d1 / 2, d2 / 2)

    ROTATION = math.radians(angle)

    REFERENCE_POINT = int(ps_points[0][0]), int(ps_points[0][1])

    (m1, h1), (m2, h2) = find_tangent_lines(
        center=CENTER,
        semi_axes=SEMI_AXES,
        rotation=ROTATION,
        reference_point=REFERENCE_POINT,
    )

    tmp_p_disp = 150
    op = ((ps_points[0][1] + tmp_p_disp - h1) / m1)

    if return_vis_img:
        cv2.line(tmp_img, REFERENCE_POINT, (int(op), int(ps_points[0][1] + tmp_p_disp)), (240, 0, 0), 2)

    aop = ang([ps_points[0], ps_points[1]], [REFERENCE_POINT, (op, ps_points[0][1] + tmp_p_disp)])
    if return_vis_img:
        return aop, tmp_img
    else:
        return aop


if __name__ == '__main__':
    sample_label = cv2.imread(
        'train/sample_output/1/epoch_32/labels/label_0.png'
        , 0
    )

    sample_pred = cv2.imread(
        'train/sample_output/1/epoch_32/prediction/out_0.png', 0
    )

    gt_aop, gt_aop_img = angle_of_progression_estimation(sample_label, return_vis_img=True)
    pred_aop, pred_aop_img = angle_of_progression_estimation(sample_pred, return_vis_img=True)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(sample_label)
    axs[0, 0].set_title('Ground Truth')
    axs[0, 1].imshow(sample_pred)
    axs[0, 1].set_title('Predicted')
    axs[1, 0].imshow(gt_aop_img)
    axs[1, 0].set_title(f'Ground Truth AOP: {round(gt_aop, 2)}')
    axs[1, 1].imshow(pred_aop_img)
    axs[1, 1].set_title(f'Predicted AOP: {round(pred_aop, 2)}')
    plt.axis(
        'off'
    )
    for ax in axs.reshape(-1):
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(
        'aop_sample.png'
    )
    plt.show()
