import copy
from importlib import import_module
from segment_anything import sam_model_registry
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandZoomd,
    ScaleIntensityd,
    EnsureTyped,
    Resized,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Rand2DElasticd,
    RandAffined,
    OneOf,
    NormalizeIntensity,
    AsChannelFirstd,
    EnsureType,
    LabelToMaskd,
    HistogramNormalized,
    Resized,
    KeepLargestConnectedComponentd

)
import torch
import logging
from torchvision.transforms import GaussianBlur, RandomHorizontalFlip, RandomVerticalFlip


class MySegmentation:
    def __init__(self, path_model, fold_n=5):
        # network parameters
        # self.model = Resnet34()
        sam, img_embedding_size = sam_model_registry['vit_h'](image_size=512,
                                                              num_classes=2,
                                                              checkpoint='/opt/algorithm/model_weights/sam_vit_h_4b8939.pth',
                                                              pixel_mean=[0, 0, 0],
                                                              pixel_std=[1, 1, 1])
        self.fold_n = fold_n
        self.h_flip = RandomHorizontalFlip(1)
        self.resize_to_256 = Compose([
            Resized(keys=["label"], spatial_size=(1, 256, 256), mode=['nearest']),

            KeepLargestConnectedComponentd(keys=["label"], num_components=1)
        ])

        pkg = import_module('sam_lora_image_encoder')
        model = pkg.LoRA_Sam(sam, 4)

        self.models = [copy.deepcopy(model) for i in range(4)]
        self.path_model = path_model
        self.mean = None
        self.std = None
        self.device = torch.device('cuda')
        self.val_transform = Compose(
            [
                # HistogramNormalized(keys=["image"]),
                ScaleIntensityd(keys=["image"]),

                Resized(keys=["image"], spatial_size=(512, 512), mode=['area']),
                EnsureTyped(keys=["image"])
            ])

    def load_model(self):
        self.mean = torch.FloatTensor([0.7481, 0.5692, 0.7225]).to(self.device)
        self.std = torch.FloatTensor([0.1759, 0.2284, 0.1792]).to(self.device)

        if torch.cuda.is_available():
            print("Model loaded on CUDA")
            [self.models[i - 1].load_lora_parameters(f"/opt/algorithm/model_weights/{i}/epoch_499.pth") for i in
             range(self.fold_n)]
        else:
            print("Model loaded on CPU")
            [self.models[i - 1].load_lora_parameters(f"/opt/algorithm/model_weights/{i}/epoch_499.pth") for i in
             range(self.fold_n)]

        [self.models[i].to(self.device) for i in range(self.fold_n)]

        logging.info("Model loaded. Mean: {} ; Std: {}".format(self.mean, self.std))
        return True

    def process_image(self, input_image):
        device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
        image = self.val_transform({"image": input_image})["image"]
        image = image.to(device=device).unsqueeze(0)
        input_h_flipped = self.h_flip(image)

        final_output = torch.zeros((1, 3, 512, 512), dtype=torch.float32).to(device)

        for i in range(self.fold_n):
            self.models[i].eval()

            # Putting images into the network
            outputs = self.models[i](image, True, 512)
            outputs_h_flip = self.models[i](input_h_flipped, True, 512)

            output_masks_t = (outputs['masks'] + self.h_flip(outputs_h_flip['masks'])) / 2

            final_output += output_masks_t

        output_masks = torch.argmax(torch.softmax(final_output / self.fold_n, dim=1), dim=1, keepdim=True)
        output_masks = self.resize_to_256({"label": output_masks})["label"]

        return output_masks
