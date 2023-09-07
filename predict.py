import argparse
import glob
import numpy as np
import os
import cv2

import json
import torch
from PIL import Image
import SimpleITK as sitk
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.ops import box_iou
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from models.diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from models.diffusiondet.util.model_ema import add_model_ema_configs
from ultralytics import YOLO
from models.unet.utils import load_unet, load_seunet
from util.slconfig import SLConfig

from util.weighted_boxes_fusion import weighted_boxes_fusion
import models.dino.datasets.transforms as DT
from image_list import list_ids_pre, list_ids_final

quadrant_remap = {
    0: 1,
    1: 0,
    2: 2,
    3: 3,
}


class D2DetectionPredictor:
    def __init__(self, cfg, threshold=0.5) -> None:
        self.predictor = DefaultPredictor(cfg)
        self.threshold = threshold

    def predict(self, image):
        predictions = self.predictor(image)
        instances = predictions["instances"]
        return instances[instances.scores > self.threshold]


class DinoDetectionPredictor:
    def __init__(self, args, model_checkpoint_path, score_threshold, cuda=True, mean=None, std=None) -> None:
        model, criterion, postprocessors = build_model_main(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        self.cuda = cuda
        if cuda:
            model = model.cuda()
        model.eval()
        self.model = model
        self.postprocessors = postprocessors

        self.transform = DT.Compose(
            [
                DT.RandomResize([800], max_size=1333),
                DT.ToTensor(),
                DT.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # the origin config, maybe different from the dentex dataset
            ]
        )
        self.score_threshold = score_threshold

    def predict(self, image: np.ndarray):
        image_shape = image.shape[:2]
        image = Image.fromarray(image)
        image, _ = self.transform(image, None)
        if self.cuda:
            image = image.cuda()
        outputs = self.model(image.unsqueeze(0))
        scale = torch.tensor([image_shape])
        if self.cuda:
            scale = scale.cuda()
        output = self.postprocessors["bbox"](outputs, scale)[0]
        scores = output["scores"]
        labels = output["labels"]
        boxes = output["boxes"]
        select_mask = scores > self.score_threshold

        scores = scores[select_mask]
        labels = labels[select_mask]
        boxes = boxes[select_mask]

        return {
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
        }


class SegmentationPredictor:
    def __init__(self, model, mean=0.458, std=0.173, cuda=True) -> None:
        self.model = model
        self.model.eval()
        self.cuda = cuda
        self.mean = mean  # default mean and std here are caculated from dentex dataset
        self.std = std

    def predict(self, image: np.ndarray) -> torch.Tensor:
        origin_shape = image.shape
        image = Image.fromarray(image).resize((256, 256))
        image = F.to_tensor(image)
        image = F.normalize(image, [self.mean], [self.std])
        if self.cuda:
            image = image.cuda()
        image = image.unsqueeze(0)

        predictions = self.model(image)
        predictions = predictions.squeeze(0)
        predictions = torch.argmax(predictions, dim=0, keepdim=True)
        predictions = F.resize(predictions, origin_shape, F.InterpolationMode.NEAREST)
        return predictions.squeeze(0)


def label_mask_to_bbox(mask: np.ndarray):
    """
    convert segmentation mask to bbox, only keep the largest connected component for each label
    mask: (H, W)
    return: bbox_dict, key is label, value is bbox (x, y, w, h)
    """
    bbox_dict = {}
    for label in np.unique(mask):
        if label == 0:
            continue
        label_mask = mask == label
        label_mask = label_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        if len(contours) == 0:
            continue
        bbox = cv2.boundingRect(contours[0])
        bbox_dict[label] = bbox
    return bbox_dict


def build_model_main(args):
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def calc_iou(box1, box2):
    """
    calculate iou between two boxes
        box: (x1, y1, x2, y2)
    """

    box1_tensor = torch.tensor(box1).float()
    box2_tensor = torch.tensor(box2).float()
    return box_iou(box1_tensor.unsqueeze(0), box2_tensor.unsqueeze(0)).item()


def format_result_json_obj(disease_instances):
    """
    format disease_instances to fianl json object
    """
    result = {
        "name": "Regions of interest",
        "type": "Multiple 2D bounding boxes",
        "boxes": [],
        "version": {"major": 1, "minor": 0},
    }
    for disease_instance in disease_instances:
        bbox = disease_instance["bbox"]
        disease_id = disease_instance["disease_id"]
        enumeration_id = disease_instance["enumeration_id"]

        category_str = f"{(enumeration_id - 1) // 8} - {(enumeration_id - 1) % 8} - {disease_id}"
        x1, y1, x2, y2 = bbox
        image_id = int(disease_instance["image_id"])
        corners = [[x1, y1, image_id], [x1, y2, image_id], [x2, y1, image_id], [x2, y2, image_id]]
        box_meta = {
            "name": category_str,
            "corners": corners,
            "probability": disease_instance["score"],
        }
        result["boxes"].append(box_meta)

    return result


def filter_duplicated_disease_instances(disease_instances):
    # only keep the highest score instance for the same label on the same image
    filtered_disease_instances = []
    for disease_instance in disease_instances:
        duplicate = False
        for filtered_disease_instance in filtered_disease_instances:
            if (
                disease_instance["image_id"] == filtered_disease_instance["image_id"]
                and disease_instance["disease_id"] == filtered_disease_instance["disease_id"]
                and disease_instance["enumeration_id"] == filtered_disease_instance["enumeration_id"]
            ):
                duplicate = True
                break

        if not duplicate:
            filtered_disease_instances.append(disease_instance)
        else:
            if disease_instance["score"] > filtered_disease_instance["score"]:
                disease_instance["bbox"] = filtered_disease_instance["bbox"]
                filtered_disease_instances.remove(filtered_disease_instance)
                filtered_disease_instances.append(disease_instance)

    print(f"before: {len(disease_instances)} instances, after: {len(filtered_disease_instances)} instances")
    return filtered_disease_instances


def filter_disease_instances_by_prior(disease_instances):
    # using some prior knowledge to filter disease instances

    filtered_disease_instances = []
    for disease_instance in disease_instances:
        disease_id = disease_instance["disease_id"]
        enumeration_id = disease_instance["enumeration_id"]

        # 1. impacted tooth should only appear on enumeration_id % 8 == 7
        # if it is close to, adjust it, otherwise filter it
        if disease_id == 0:
            if enumeration_id % 8 == 7:
                disease_instance["enumeration_id"] = enumeration_id + 1
                print(f"found an impacted tooth with enumeration_id {enumeration_id}, adjust to {enumeration_id + 1}")
            elif enumeration_id % 8 != 0:
                print(f"found an impacted tooth with enumeration_id {enumeration_id}, filter it")
                continue
            else:
                ...

        # 2. when caries and deep caries appear on the same tooth, only keep the deep caries
        if disease_id in [1, 3]:
            duplicate = False
            for filtered_disease_instance in filtered_disease_instances:
                if not (
                    disease_instance["image_id"] == filtered_disease_instance["image_id"]
                    and disease_instance["enumeration_id"] == filtered_disease_instance["enumeration_id"]
                ):
                    continue

                if filtered_disease_instance["disease_id"] in [1, 3]:
                    filtered_disease_instance["disease_id"] = 3
                    filtered_disease_instance["score"] = max(
                        filtered_disease_instance["score"], disease_instance["score"]
                    )
                    duplicate = True
                    print(
                        f"found a caries and a deep caries on the same tooth (image id: {disease_instance['image_id']}, enumeration id: {disease_instance['enumeration_id']}), only keep the deep caries"
                    )
                    break

            if duplicate:
                continue

        filtered_disease_instances.append(disease_instance)
    return filtered_disease_instances


@torch.no_grad()
def main():
    confidence_threshold = 0.01
    cuda = True
    iou_match_threshold = 0.3
    container = False

    if container:
        input_image_dir = "/input/images/panoramic-dental-xrays"
        output_json_path = "/output/abnormal-teeth-detection.json"
    else:
        input_image_dir = "./"
        output_json_path = "./results/abnormal-teeth-detection.json"

    # load models
    quadrant_cfg = get_cfg()
    add_diffusiondet_config(quadrant_cfg)
    add_model_ema_configs(quadrant_cfg)
    quadrant_cfg.merge_from_file("configs/diffdet/diffdet.dentex.swinbase.quadrant.yaml")
    quadrant_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    quadrant_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    quadrant_cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    quadrant_cfg.freeze()
    quadrant_predictor = D2DetectionPredictor(quadrant_cfg)

    disease_dino_model_config_path = "configs/dino/DINO_4scale_swin_cls4.py"
    disease_dino_model_checkpoint_path = "checkpoints/dino_swin_disease_all_27.pth"
    disease_dino_cfg = SLConfig.fromfile(disease_dino_model_config_path)
    disease_dino_cfg.device = "cuda" if cuda else "cpu"
    disease_dino_predictor = DinoDetectionPredictor(
        disease_dino_cfg, disease_dino_model_checkpoint_path, confidence_threshold, cuda=cuda
    )

    disease_yolo_predictor = YOLO("checkpoints/yolo_disease_all_25.pt")

    enumeration_detector_config_path = "configs/dino/DINO_4scale_cls32.py"
    enumeration_detector_checkpoint_path = "checkpoints/dino_res50_enum_24.pth"
    enumeration_cfg = SLConfig.fromfile(enumeration_detector_config_path)
    enumeration_cfg.device = "cuda" if cuda else "cpu"
    enumeration_predictor = DinoDetectionPredictor(
        enumeration_cfg, enumeration_detector_checkpoint_path, 0.3, cuda=cuda
    )

    enumeration9_segmentation_models = [
        load_unet("checkpoints/unet_3_epoch53.pth", 10, cuda=cuda),  # out_channels include background
        load_unet("checkpoints/unet_3_epoch240.pth", 10, cuda=cuda),
        load_seunet("checkpoints/seunet_3_epoch15.pth", 10, cuda=cuda),
    ]

    enumeration32_segmentation_models = [
        load_unet("checkpoints/unet33_1_epoch175.pth", 33, cuda=cuda),
        load_seunet("checkpoints/seunet33_1_epoch132.pth", 33, cuda=cuda),
        load_seunet("checkpoints/seunet33_1_epoch250.pth", 33, cuda=cuda),
    ]

    enumeration9_segmentation_predictors = [
        SegmentationPredictor(model, cuda=cuda) for model in enumeration9_segmentation_models
    ]

    enumeration32_segmentation_predictors = [
        SegmentationPredictor(model, cuda=cuda) for model in enumeration32_segmentation_models
    ]

    # load image
    file_path = glob.glob(os.path.join(input_image_dir, "*.mha"))[0]
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)

    all_disease_instances = []
    image_count = image_array.shape[2]
    list_ids = list_ids_final if image_count == len(list_ids_final) else list_ids_pre
    if image_count != len(list_ids):
        raise ValueError(f"image count is {image_count}, which maybe not intended.")
    image_type = "test" if image_count == len(list_ids_final) else "val"

    print(f"task type: {image_type}")
    for k in range(image_count):
        image_name = f"{image_type}_{k}.png"

        for input_img in list_ids:
            if input_img["file_name"] == image_name:
                image_id = input_img["id"]
                image_height = input_img["height"]
                image_width = input_img["width"]
        print(f"processing {image_name}, id {image_id}...")

        image_rgb_arr = image_array[:, :, k, :]
        image_rgb_arr = image_rgb_arr[:image_height, :image_width, :]
        image_bgr_arr = image_rgb_arr[:, :, ::-1]
        image_gray_arr = cv2.cvtColor(image_bgr_arr, cv2.COLOR_BGR2GRAY)

        enumeration_bboxes_each_seg_model: list[dict[int, list[int]]] = []  # list[dict[enumeration_id, bbox]]]
        enumeration_result_by_detector = {}

        # 1. detect quadrants
        quadrant_instances = quadrant_predictor.predict(image_bgr_arr)
        # select highest score instance for each quadrant
        quadrant_bboxes_dict = {}
        quadrant_scores_dict = {}
        for quadrant_instance_id in range(len(quadrant_instances)):
            quadrant_bbox = quadrant_instances.pred_boxes.tensor[quadrant_instance_id].tolist()
            quadrant_bbox = [int(x) for x in quadrant_bbox]
            quadrant_id = quadrant_instances.pred_classes[quadrant_instance_id].item()
            quadrant_id = quadrant_remap[quadrant_id]  # 0, 1, 2, 3 -> 1, 0, 2, 3
            quadrant_score = quadrant_instances.scores[quadrant_instance_id].item()
            if quadrant_id not in quadrant_bboxes_dict or quadrant_score > quadrant_scores_dict[quadrant_id]:
                quadrant_bboxes_dict[quadrant_id] = quadrant_bbox
                quadrant_scores_dict[quadrant_id] = quadrant_score

        # 2. crop each quadrant and predict 9 class segmentation
        for quadrant_id, quadrant_bbox in quadrant_bboxes_dict.items():
            quadrant_image_gray_arr = image_gray_arr[
                quadrant_bbox[1] : quadrant_bbox[3], quadrant_bbox[0] : quadrant_bbox[2]
            ]
            for quadrant_enumeration_predictor in enumeration9_segmentation_predictors:
                quadrant_enumeration_prediction_mask = quadrant_enumeration_predictor.predict(quadrant_image_gray_arr)
                quadrant_enumeration_prediction_mask = quadrant_enumeration_prediction_mask.cpu().numpy()
                quadrant_enumeratino_prediction_bbox_dict = label_mask_to_bbox(quadrant_enumeration_prediction_mask)

                enumeration_bboxes_each_seg_model.append({})
                # convert local bbox to global bbox
                for quadrant_enumeration_id, enumeration_bbox in quadrant_enumeratino_prediction_bbox_dict.items():
                    if quadrant_enumeration_id == 9:
                        continue

                    enumeration_bbox = [  # local to global and meanwhile xywh to xyxy
                        enumeration_bbox[0] + quadrant_bbox[0],
                        enumeration_bbox[1] + quadrant_bbox[1],
                        enumeration_bbox[0] + quadrant_bbox[0] + enumeration_bbox[2],
                        enumeration_bbox[1] + quadrant_bbox[1] + enumeration_bbox[3],
                    ]

                    enumeration_id = quadrant_id * 8 + quadrant_enumeration_id
                    enumeration_bboxes_each_seg_model[-1].update({enumeration_id: enumeration_bbox})

        # 3. predict 32 class segmentation
        for enumeration_predictors in enumeration32_segmentation_predictors:
            enumeration_prediction_mask = enumeration_predictors.predict(image_gray_arr)
            enumeration_prediction_mask = enumeration_prediction_mask.cpu().numpy()
            enumeration_prediction_bbox_dict = label_mask_to_bbox(enumeration_prediction_mask)

            enumeration_bboxes_each_seg_model.append({})
            for enumeration_id, enumeration_bbox in enumeration_prediction_bbox_dict.items():
                enumeration_bboxes_each_seg_model[-1].update({enumeration_id: enumeration_bbox})

        enumeration_detector_prediction = enumeration_predictor.predict(image_rgb_arr)
        for enumeration_instance_id in range(len(enumeration_detector_prediction["boxes"])):
            enumeration_bbox = enumeration_detector_prediction["boxes"][enumeration_instance_id].tolist()
            # enumeration_bbox = [int(x) for x in enumeration_bbox]
            enumeration_score = enumeration_detector_prediction["scores"][enumeration_instance_id].item()
            enumeration_label = enumeration_detector_prediction["labels"][enumeration_instance_id].item() + 1

            enumeration_result_by_detector.update(
                {enumeration_label: {"bbox": enumeration_bbox, "score": enumeration_score}}
            )

        # 4. detect disease
        disease_dino_instances = disease_dino_predictor.predict(image_rgb_arr)
        disease_dino_boxes = []
        disease_dino_labels = []
        disease_dino_scores = []

        for disease_instance_id in range(len(disease_dino_instances["boxes"])):
            disease_bbox = disease_dino_instances["boxes"][disease_instance_id].tolist()
            # disease_bbox = [int(x) for x in disease_bbox]
            disease_score = disease_dino_instances["scores"][disease_instance_id].item()
            disease_label = disease_dino_instances["labels"][disease_instance_id].item()

            disease_dino_boxes.append(disease_bbox)
            disease_dino_labels.append(disease_label)
            disease_dino_scores.append(disease_score)

        disease_yolo_instances = disease_yolo_predictor.predict(image_bgr_arr, conf=confidence_threshold)[0].boxes
        disease_yolo_boxes = []
        disease_yolo_labels = []
        disease_yolo_scores = []

        for disease_instance_id in range(len(disease_yolo_instances)):
            disease_bbox = disease_yolo_instances.xyxy[disease_instance_id].tolist()
            # disease_bbox = [int(x) for x in disease_bbox]
            disease_score = disease_yolo_instances.conf[disease_instance_id].item()
            disease_label = int(disease_yolo_instances.cls[disease_instance_id].item())

            disease_yolo_boxes.append(disease_bbox)
            disease_yolo_labels.append(disease_label)
            disease_yolo_scores.append(disease_score)

        # normalize disease bbox for weighted_boxes_fusion
        disease_dino_boxes = np.array(disease_dino_boxes)
        disease_dino_boxes[:, 0] = disease_dino_boxes[:, 0] / image_width
        disease_dino_boxes[:, 1] = disease_dino_boxes[:, 1] / image_height
        disease_dino_boxes[:, 2] = disease_dino_boxes[:, 2] / image_width
        disease_dino_boxes[:, 3] = disease_dino_boxes[:, 3] / image_height
        disease_yolo_boxes = np.array(disease_yolo_boxes)
        disease_yolo_boxes[:, 0] = disease_yolo_boxes[:, 0] / image_width
        disease_yolo_boxes[:, 1] = disease_yolo_boxes[:, 1] / image_height
        disease_yolo_boxes[:, 2] = disease_yolo_boxes[:, 2] / image_width
        disease_yolo_boxes[:, 3] = disease_yolo_boxes[:, 3] / image_height

        disease_dino_boxes = disease_dino_boxes.tolist()
        disease_yolo_boxes = disease_yolo_boxes.tolist()

        # fuse disease bbox
        fuse_results = weighted_boxes_fusion(
            [disease_dino_boxes, disease_yolo_boxes],
            [disease_dino_scores, disease_yolo_scores],
            [disease_dino_labels, disease_yolo_labels],
            weights=[2, 1],
            iou_thr=0.6,
        )
        fused_boxes, fused_scores, fused_labels = fuse_results

        # denormalize disease bbox
        fused_boxes = np.array(fused_boxes)
        fused_boxes[:, 0] = fused_boxes[:, 0] * image_width
        fused_boxes[:, 1] = fused_boxes[:, 1] * image_height
        fused_boxes[:, 2] = fused_boxes[:, 2] * image_width
        fused_boxes[:, 3] = fused_boxes[:, 3] * image_height
        fused_boxes = fused_boxes.tolist()

        disease_instances_list = []
        for disease_instance_id in range(len(fused_boxes)):
            disease_bbox = fused_boxes[disease_instance_id]
            disease_score = fused_scores[disease_instance_id]
            disease_label = int(fused_labels[disease_instance_id])

            disease_instances_list.append(
                {
                    "bbox": disease_bbox,
                    "score": disease_score,
                    "disease_id": disease_label,
                    "image_id": image_id,
                }
            )

        print(f"image {image_id} has {len(disease_instances_list)} disease instances")

        # 5. for each diease instance, match the most possible enumeration_id
        # voting by iou from many segmentation and detection models
        filtered_disease_instances_list = []
        for disease_instance in disease_instances_list:
            disease_bbox = disease_instance["bbox"]
            disease_id = disease_instance["disease_id"]

            # record the accumulated iou for each enumeration_id
            enumeration_id_iou_dict = {i: 0 for i in range(1, 33)}
            for enumeration_bboxes in enumeration_bboxes_each_seg_model:
                for enumeration_id, enumeration_bbox in enumeration_bboxes.items():
                    iou = calc_iou(disease_bbox, enumeration_bbox)
                    if iou > iou_match_threshold:
                        enumeration_id_iou_dict[enumeration_id] += iou

            # higher weight for detection result
            for enumeration_id, enumeration_result in enumeration_result_by_detector.items():
                enumeration_bbox = enumeration_result["bbox"]
                iou = calc_iou(disease_bbox, enumeration_bbox)
                if iou > iou_match_threshold:
                    enumeration_id_iou_dict[enumeration_id] += iou * 3 * enumeration_result["score"]

            # select the enumeration_id with highest accumulated iou
            max_iou = 0
            max_iou_enumeration_id = 0
            for enumeration_id, accumulated_iou in enumeration_id_iou_dict.items():
                if accumulated_iou > max_iou:
                    max_iou = accumulated_iou
                    max_iou_enumeration_id = enumeration_id

            if max_iou_enumeration_id == 0:
                print(
                    f"waring, image {image_id} disease: {disease_id}, bbox: {disease_bbox} has no enumeration_id, ingore it"
                )
            else:
                disease_instance["enumeration_id"] = max_iou_enumeration_id
                filtered_disease_instances_list.append(disease_instance)
        all_disease_instances.extend(filtered_disease_instances_list)

    all_disease_instances = filter_duplicated_disease_instances(all_disease_instances)
    all_disease_instances = filter_disease_instances_by_prior(all_disease_instances)
    # 6. save result
    print("saving result...")
    result_json_obj = format_result_json_obj(all_disease_instances)
    with open(output_json_path, "w") as f:
        json.dump(result_json_obj, f, indent=4)


if __name__ == "__main__":
    main()
