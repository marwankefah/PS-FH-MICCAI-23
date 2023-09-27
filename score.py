import os
import cv2
from monai.transforms import KeepLargestConnectedComponent
import json
from aop import angle_of_progression_estimation
from score_utils import calculate_metric_percase
import numpy as np


def ps_fh_score(data):
    # Extract the values
    dice_fh = data["aggregates"]["dice_fh"]
    dice_ps = data["aggregates"]["dice_ps"]
    dice_all = data["aggregates"]["dice_all"]
    hd_fh = data["aggregates"]["hd_fh"]
    hd_ps = data["aggregates"]["hd_ps"]
    hd_all = data["aggregates"]["hd_all"]
    asd_fh = data["aggregates"]["asd_fh"]
    asd_ps = data["aggregates"]["asd_ps"]
    asd_all = data["aggregates"]["asd_all"]
    aop = data["aggregates"]["aop"]

    # Calculate the score
    S = 0.25 * (dice_fh + dice_ps + dice_all) / 3.0 + \
        0.25 * (
                0.5 * ((1 - hd_fh / 100 + 1 - hd_ps / 100 + 1 - hd_all / 100) / 3.0) + \
                0.5 * ((1 - asd_fh / 100 + 1 - asd_ps / 100 + 1 - asd_all / 100) / 3.0)
        ) + \
        0.5 * (1 - aop / 180)
    return S


if __name__ == '__main__':

    post_processing_monai = KeepLargestConnectedComponent(num_components=1)

    output_path = 'train/sample_output/1/epoch_32'

    label_imgs = os.listdir(os.path.join(output_path, 'labels'))

    dice_1_scores = []
    dice_2_scores = []
    dice_all_scores = []

    hd_1_scores = []
    hd_2_scores = []
    hd_all_scores = []

    asd_1_scores = []
    asd_2_scores = []
    asd_all_scores = []
    aop_gt_score = []
    aop_pred_score = []
    aop_diff_score = []

    aop_scores = []
    for i in label_imgs:
        label = cv2.imread(f'{output_path}/labels/label_{i[6:-4]}.png', 0)
        out = cv2.imread(f'{output_path}/prediction/out_{i[6:-4]}.png', 0)

        met1_1, met2_1, met3_1 = calculate_metric_percase(out == 1, label == 1)
        dice_1_scores.append(met1_1)
        hd_1_scores.append(met2_1)
        asd_1_scores.append(met3_1)

        met1_2, met2_2, met3_2 = calculate_metric_percase(out == 2, label == 2)
        dice_2_scores.append(met1_2)
        hd_2_scores.append(met2_2)
        asd_2_scores.append(met3_2)

        met1_a, met2_a, met3_a = calculate_metric_percase(out > 0, label > 0)

        dice_all_scores.append(met1_a)
        hd_all_scores.append(met2_a)
        asd_all_scores.append(met3_a)

        label = post_processing_monai(label[None, :, :]).numpy()[0]
        out = post_processing_monai(out[None, :, :]).numpy()[0]

        aop_gt = angle_of_progression_estimation(label)

        aop_pred = angle_of_progression_estimation(out)

        aop_gt_score.append(aop_gt)
        aop_pred_score.append(aop_pred)
        aop_diff_score.append(abs(aop_pred - aop_gt))

    data = {}

    data['aggregates'] = {}

    data["aggregates"]["dice_fh"] = np.mean(dice_2_scores)
    data["aggregates"]["dice_ps"] = np.mean(dice_1_scores)
    data["aggregates"]["dice_all"] = np.mean(dice_all_scores)
    data["aggregates"]["hd_fh"] = np.mean(hd_2_scores)
    data["aggregates"]["hd_ps"] = np.mean(hd_1_scores)
    data["aggregates"]["hd_all"] = np.mean(hd_all_scores)
    data["aggregates"]["asd_fh"] = np.mean(asd_2_scores)
    data["aggregates"]["asd_ps"] = np.mean(asd_1_scores)
    data["aggregates"]["asd_all"] = np.mean(asd_all_scores)
    data["aggregates"]["aop"] = np.mean(aop_diff_score)

    score = ps_fh_score(data)

    data['score'] = score
    json_object = json.dumps(data, indent=4)

    print(json_object)

    with open("metrics.json", "w") as outfile:
        json.dump(data, outfile,
                  indent=4, separators=(',', ': '))

    print('Score calculated is', score)
