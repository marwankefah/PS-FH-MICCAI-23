import SimpleITK
from pathlib import Path
import os
import numpy as np
import torch
import torchvision
from typing import Dict
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import evalutils
import json
from segmentation import MySegmentation

execute_in_docker = True


class Nodule_seg:
    def __init__(self):
        # self.input_dir = Path("/opt/algorithm/test/")
        # self.output_dir = Path("/opt/algorithm/output/")

        self.input_dir = Path("/input/images/pelvic-2d-ultrasound/") if execute_in_docker else Path("./test/")
        self.output_dir = Path("/output/images/symphysis-segmentation/") if execute_in_docker else Path("/home/interns/marawan/FH-PS-AOP-grandchallenge/output/")
        # todo Load the trained model
        # print(os.listdir(self.input_dir))
        # print(os.listdir(self.output_dir))

        if execute_in_docker:
            path_model = "/opt/algorithm/model_weights/best_model.pth"
        else:
            path_model = "./model_weights/best_model1.pth"

        self.md = MySegmentation(path_model)
        load_success = self.md.load_model()
        if load_success:
            print("Successfully loaded model.")

    def load_image(self, image_path) -> SimpleITK.Image:
        image = SimpleITK.ReadImage(str(image_path))
        return image

    def predict(self, input_image: SimpleITK.Image):
        # Obtain input image
        image_data = SimpleITK.GetArrayFromImage(input_image)
        with torch.no_grad():
            # Put it into the network for processing
            pred = self.md.process_image(image_data)
            # Post processing and saving of predicted images
            pred = pred.squeeze()
            pred = pred.cpu().numpy().astype(np.uint8)
            # pred = np.argmax(pred, axis=0)
            pred = SimpleITK.GetImageFromArray(pred)

            return pred

    def write_outputs(self, image_name, outputs):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        SimpleITK.WriteImage(outputs, os.path.join(self.output_dir, image_name + '.mha'))
        # print(os.listdir(self.input_dir))
        # print(os.listdir(self.output_dir))

    def process(self):
        image_paths = list(self.input_dir.glob("*"))
        for image_path in image_paths:
            image_name = os.path.basename(image_path).split('.')[0]
            image = self.load_image(image_path)
            result = self.predict(image)
            self.write_outputs(image_name, result)
        print("Success hsiadhfjowiqjeoijfosdj9832049820sahfdi389u4903u409")


if __name__ == "__main__":
    Nodule_seg().process()
