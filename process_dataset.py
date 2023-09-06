import os
import json
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def process_coco_quadrant():
    """
    split quadrant dataset into train and val,
    copy data to coco directory
    """

    dataset_json = load_json("dentex_dataset/origin/quadrant/train_quadrant.json")

    image_ids = [x["id"] for x in dataset_json["images"]]
    random.shuffle(image_ids)
    train_ids = image_ids[: int(len(image_ids) * 0.8)]  # 80% for training

    train_json = {"images": [], "annotations": [], "categories": dataset_json["categories"]}
    val_json = {"images": [], "annotations": [], "categories": dataset_json["categories"]}

    mkdirs("dentex_dataset/coco/quadrant/train2017")
    mkdirs("dentex_dataset/coco/quadrant/val2017")

    for image in dataset_json["images"]:
        image_filename = image["file_name"]
        if image["id"] in train_ids:
            train_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant/xrays/{image_filename}",
                f"dentex_dataset/coco/quadrant/train2017/{image_filename}",
            )
        else:
            val_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant/xrays/{image_filename}",
                f"dentex_dataset/coco/quadrant/val2017/{image_filename}",
            )

    for annotation in dataset_json["annotations"]:
        if annotation["image_id"] in train_ids:
            train_json["annotations"].append(annotation)
        else:
            val_json["annotations"].append(annotation)

    mkdirs("dentex_dataset/coco/quadrant/annotations")
    save_json("dentex_dataset/coco/quadrant/annotations/instances_train2017.json", train_json)
    save_json("dentex_dataset/coco/quadrant/annotations/instances_val2017.json", val_json)


def process_coco_enumeration32():
    """
    convert quadrant_enumeration label to enumeration32 label,
    split dataset into train and val,
    copy data to coco directory
    """

    dataset_json = load_json("dentex_dataset/origin/quadrant_enumeration/train_quadrant_enumeration.json")

    for annotation in dataset_json["annotations"]:
        # convert quadrant_enumeration label to enumeration32 label
        category_id_1 = annotation["category_id_1"]
        category_id_2 = annotation["category_id_2"]

        annotation.pop("category_id_1")
        annotation.pop("category_id_2")

        annotation["category_id"] = category_id_1 * 8 + category_id_2

    image_ids = [x["id"] for x in dataset_json["images"]]
    random.shuffle(image_ids)
    train_ids = image_ids[: int(len(image_ids) * 0.9)]  # 90% for training

    categories = [{"id": i, "name": str(i + 1), "supercategory": str(i + 1)} for i in range(32)]

    train_json = {"images": [], "annotations": [], "categories": categories}
    val_json = {"images": [], "annotations": [], "categories": categories}

    mkdirs("dentex_dataset/coco/enumeration32/train2017")
    mkdirs("dentex_dataset/coco/enumeration32/val2017")

    for image in dataset_json["images"]:
        image_filename = image["file_name"]
        if image["id"] in train_ids:
            train_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_filename}",
                f"dentex_dataset/coco/enumeration32/train2017/{image_filename}",
            )
        else:
            val_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_filename}",
                f"dentex_dataset/coco/enumeration32/val2017/{image_filename}",
            )

    for annotation in dataset_json["annotations"]:
        if annotation["image_id"] in train_ids:
            train_json["annotations"].append(annotation)
        else:
            val_json["annotations"].append(annotation)

    mkdirs("dentex_dataset/coco/enumeration32/annotations")
    save_json("dentex_dataset/coco/enumeration32/annotations/instances_train2017.json", train_json)
    save_json("dentex_dataset/coco/enumeration32/annotations/instances_val2017.json", val_json)


def process_coco_disease():
    """
    extract disease label from quadrant_enumeration_disease label,
    split disease dataset into train and val,
    copy data to coco directory
    """

    dataset_json = load_json(
        "dentex_dataset/origin/quadrant_enumeration_disease/train_quadrant_enumeration_disease.json"
    )

    for annotation in dataset_json["annotations"]:
        # extract disease label from quadrant_enumeration_disease label
        category_id_3 = annotation["category_id_3"]

        annotation.pop("category_id_1")
        annotation.pop("category_id_2")
        annotation.pop("category_id_3")

        annotation["category_id"] = category_id_3

    image_ids = [x["id"] for x in dataset_json["images"]]
    random.shuffle(image_ids)
    train_ids = image_ids[: int(len(image_ids) * 0.8)]  # 80% for training

    categories = [
        {"id": 0, "name": "Impacted", "supercategory": "Impacted"},
        {"id": 1, "name": "Caries", "supercategory": "Caries"},
        {"id": 2, "name": "Periapical Lesion", "supercategory": "Periapical Lesion"},
        {"id": 3, "name": "Deep Caries", "supercategory": "Deep Caries"},
    ]

    train_json = {"images": [], "annotations": [], "categories": categories}
    val_json = {"images": [], "annotations": [], "categories": categories}

    mkdirs("dentex_dataset/coco/disease/train2017")
    mkdirs("dentex_dataset/coco/disease/val2017")

    for image in dataset_json["images"]:
        image_filename = image["file_name"]
        if image["id"] in train_ids:
            train_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant_enumeration_disease/xrays/{image_filename}",
                f"dentex_dataset/coco/disease/train2017/{image_filename}",
            )
        else:
            val_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant_enumeration_disease/xrays/{image_filename}",
                f"dentex_dataset/coco/disease/val2017/{image_filename}",
            )

    for annotation in dataset_json["annotations"]:
        if annotation["image_id"] in train_ids:
            train_json["annotations"].append(annotation)
        else:
            val_json["annotations"].append(annotation)

    mkdirs("dentex_dataset/coco/disease/annotations")
    save_json("dentex_dataset/coco/disease/annotations/instances_train2017.json", train_json)
    save_json("dentex_dataset/coco/disease/annotations/instances_val2017.json", val_json)


def process_coco_disease_all():
    """
    extract disease label from quadrant_enumeration_disease label,
    make all data for both training and validation,
    copy data to coco directory
    """

    dataset_json = load_json(
        "dentex_dataset/origin/quadrant_enumeration_disease/train_quadrant_enumeration_disease.json"
    )

    for annotation in dataset_json["annotations"]:
        # extract disease label from quadrant_enumeration_disease label
        category_id_3 = annotation["category_id_3"]

        annotation.pop("category_id_1")
        annotation.pop("category_id_2")
        annotation.pop("category_id_3")

        annotation["category_id"] = category_id_3

    categories = [
        {"id": 0, "name": "Impacted", "supercategory": "Impacted"},
        {"id": 1, "name": "Caries", "supercategory": "Caries"},
        {"id": 2, "name": "Periapical Lesion", "supercategory": "Periapical Lesion"},
        {"id": 3, "name": "Deep Caries", "supercategory": "Deep Caries"},
    ]

    result_json = {
        "images": dataset_json["images"],
        "annotations": dataset_json["annotations"],
        "categories": categories,
    }

    mkdirs("dentex_dataset/coco/disease_all/train2017")
    mkdirs("dentex_dataset/coco/disease_all/val2017")

    shutil.copytree(
        "dentex_dataset/origin/quadrant_enumeration_disease/xrays",
        "dentex_dataset/coco/disease_all/train2017",
        dirs_exist_ok=True,
    )

    shutil.copytree(
        "dentex_dataset/origin/quadrant_enumeration_disease/xrays",
        "dentex_dataset/coco/disease_all/val2017",
        dirs_exist_ok=True,
    )

    mkdirs("dentex_dataset/coco/disease_all/annotations")
    save_json("dentex_dataset/coco/disease_all/annotations/instances_train2017.json", result_json)
    save_json("dentex_dataset/coco/disease_all/annotations/instances_val2017.json", result_json)


def convert_box_coco_to_yolo(size, box):
    """
    convert box from coco format to yolo format,
    size: (width, height)
    box: (x, y, w, h), unnormalized
    return: (cx, cy, w, h), normalized
    """
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    # rescale to 0~1, and round to 6 decimal places
    cx = round(cx * dw, 6)
    w = round(w * dw, 6)
    cy = round(cy * dh, 6)
    h = round(h * dh, 6)
    return (cx, cy, w, h)


def process_yolo_disease_all():
    """
    convert labels to yolo format,
    copy data to yolo directory.
    this function uses the result of process_coco_disease_all()
    """

    dataset_json = load_json("dentex_dataset/coco/disease_all/annotations/instances_train2017.json")

    coco_image_dir = "dentex_dataset/coco/disease_all/train2017"

    yolo_image_dir_train = "dentex_dataset/yolo/disease_all/images/train2017"
    yolo_image_dir_val = "dentex_dataset/yolo/disease_all/images/val2017"
    yolo_label_dir_train = "dentex_dataset/yolo/disease_all/labels/train2017"
    yolo_label_dir_val = "dentex_dataset/yolo/disease_all/labels/val2017"

    mkdirs(yolo_image_dir_train)
    mkdirs(yolo_image_dir_val)
    mkdirs(yolo_label_dir_train)
    mkdirs(yolo_label_dir_val)

    category_names = [
        "Impacted",
        "Caries",
        "Periapical Lesion",
        "Deep Caries",
    ]

    with open(f"{yolo_label_dir_train}/classes.txt", "w") as f:
        f.write("\n".join(category_names))
    shutil.copy(
        f"{yolo_label_dir_train}/classes.txt",
        f"{yolo_label_dir_val}/classes.txt",
    )

    for image in dataset_json["images"]:
        image_filename = image["file_name"]

        shutil.copy(
            f"{coco_image_dir}/{image_filename}",
            f"{yolo_image_dir_train}/{image_filename}",
        )
        shutil.copy(
            f"{coco_image_dir}/{image_filename}",
            f"{yolo_image_dir_val}/{image_filename}",
        )

        label_filename = image_filename[:-4] + ".txt"

        with open(f"{yolo_label_dir_train}/{label_filename}", "w") as f:
            for annotation in dataset_json["annotations"]:
                if annotation["image_id"] == image["id"]:
                    box = annotation["bbox"]
                    box = convert_box_coco_to_yolo((image["width"], image["height"]), box)
                    f.write(f"{annotation['category_id']} {' '.join(map(str, box))}\n")

        shutil.copy(
            f"{yolo_label_dir_train}/{label_filename}",
            f"{yolo_label_dir_val}/{label_filename}",
        )


def process_yolo_disease():
    """
    split disease dataset into train and val.
    this function uses the result of process_coco_disease() and process_yolo_disease_all()
    """
    train_json = load_json("dentex_dataset/coco/disease/annotations/instances_train2017.json")
    val_json = load_json("dentex_dataset/coco/disease/annotations/instances_val2017.json")

    yolo_image_dir_train = "dentex_dataset/yolo/disease/images/train2017"
    yolo_image_dir_val = "dentex_dataset/yolo/disease/images/val2017"
    yolo_label_dir_train = "dentex_dataset/yolo/disease/labels/train2017"
    yolo_label_dir_val = "dentex_dataset/yolo/disease/labels/val2017"

    mkdirs(yolo_image_dir_train)
    mkdirs(yolo_image_dir_val)
    mkdirs(yolo_label_dir_train)
    mkdirs(yolo_label_dir_val)

    yolo_image_src_dir = "dentex_dataset/yolo/disease_all/images/train2017"
    yolo_label_src_dir = "dentex_dataset/yolo/disease_all/labels/train2017"

    for image in train_json["images"]:
        image_filename = image["file_name"]

        shutil.copy(
            f"{yolo_image_src_dir}/{image_filename}",
            f"{yolo_image_dir_train}/{image_filename}",
        )

        label_filename = image_filename[:-4] + ".txt"

        shutil.copy(
            f"{yolo_label_src_dir}/{label_filename}",
            f"{yolo_label_dir_train}/{label_filename}",
        )

    for image in val_json["images"]:
        image_filename = image["file_name"]

        shutil.copy(
            f"{yolo_image_src_dir}/{image_filename}",
            f"{yolo_image_dir_val}/{image_filename}",
        )

        label_filename = image_filename[:-4] + ".txt"

        shutil.copy(
            f"{yolo_label_src_dir}/{label_filename}",
            f"{yolo_label_dir_val}/{label_filename}",
        )


def process_seg_enumeration32():
    """
    draw segmentation masks for enumeration32
    """
    dataset_json = load_json("dentex_dataset/origin/quadrant_enumeration/train_quadrant_enumeration.json")
    mkdirs("dentex_dataset/segmentation/enumeration32/masks")
    mkdirs("dentex_dataset/segmentation/enumeration32/xrays")

    image_names = []
    for image_info in dataset_json["images"]:
        image_names.append(image_info["file_name"])
        # draw mask for each image
        image = Image.open(f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}")
        mask = Image.new("L", image.size)
        draw = ImageDraw.Draw(mask)

        for annotation in dataset_json["annotations"]:
            if annotation["image_id"] == image_info["id"]:
                points = np.array(annotation["segmentation"]).reshape(-1, 2)
                points = [tuple(point) for point in points]
                # draw polygon, fill with label 1~32
                draw.polygon(points, fill=annotation["category_id_1"] * 8 + annotation["category_id_2"] + 1)

        # save mask and copy image
        mask.save(f"dentex_dataset/segmentation/enumeration32/masks/{image_info['file_name']}")
        shutil.copy(
            f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_info['file_name']}",
            f"dentex_dataset/segmentation/enumeration32/xrays/{image_info['file_name']}",
        )

    save_json("dentex_dataset/segmentation/enumeration32/image_names.json", image_names)


def convert_mask_32_to_9(mask: np.ndarray, quadrant: int) -> np.ndarray:
    """
    convert mask from 32 classes to 9 classes,
    when a foreground label belongs to quadrant, it is converted to 1~8,
    when a foreground label does not belong to quadrant, it is converted to 9
    """
    assert quadrant in [0, 1, 2, 3]

    mask_out = mask.copy()
    mask_out[mask != 0] = 9

    for i in range(1, 9):
        mask_out[mask == (i + quadrant * 8)] = i

    return mask_out


def process_seg_enumeration9(quadrant_prediction_path: str):
    """
    draw segmentation masks for enumeration9
    quadrant_prediction_path: path to quadrant prediction json file, obtained by quadrant detection model
        see results/enumeration_dataset_quadrant_predictions.json for example
    this function uses the result of process_seg_enumeration32()
    """

    quadrant_predictions = load_json(quadrant_prediction_path)
    quadrant_remap = {
        0: 1,
        1: 0,
        2: 2,
        3: 3,
    }  # remap because category names are different between quadrant and quadrant_enumeration/quadrant_enumeration_disease
    mkdirs("dentex_dataset/segmentation/enumeration9/masks")
    mkdirs("dentex_dataset/segmentation/enumeration9/xrays")

    image_names = []
    for prediction_result in quadrant_predictions:
        file_name = prediction_result["file_name"]

        # read image and mask
        image = Image.open(f"dentex_dataset/segmentation/enumeration32/xrays/{file_name}")
        image = np.array(image)
        mask = Image.open(f"dentex_dataset/segmentation/enumeration32/masks/{file_name}")
        mask = np.array(mask)

        # crop image and mask
        for i in range(len(prediction_result["instances"]["classes"])):
            quadrant_id = prediction_result["instances"]["classes"][i]
            quadrant_id = quadrant_remap[quadrant_id]
            bbox = prediction_result["instances"]["boxes"][i]
            bbox = list(map(int, bbox))

            cropped_image_name = f"{file_name[:-4]}_quadrant_{quadrant_id}.png"
            image_names.append(cropped_image_name)

            # crop image and save
            image_crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            Image.fromarray(image_crop).save(f"dentex_dataset/segmentation/enumeration9/xrays/{cropped_image_name}")

            # crop mask and save
            mask_crop = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            mask_crop = convert_mask_32_to_9(mask_crop, quadrant_id)
            Image.fromarray(mask_crop).save(f"dentex_dataset/segmentation/enumeration9/masks/{cropped_image_name}")

    save_json("dentex_dataset/segmentation/enumeration9/image_names.json", image_names)


if __name__ == "__main__":
    # process_coco_quadrant()
    # process_coco_enumeration32()
    # process_coco_disease()
    # process_coco_disease_all()
    # process_yolo_disease_all()
    # process_yolo_disease()
    # process_seg_enumeration32()
    # process_seg_enumeration9("results/enumeration_dataset_quadrant_predictions.json")
    ...
