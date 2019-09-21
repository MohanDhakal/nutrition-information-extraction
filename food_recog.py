import os
import sys
import json
import numpy as np
import skimage.draw



ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, config

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



############################################################
#  Configurations
############################################################


class FoodConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "food"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + food

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.5


class FoodDataset(utils.Dataset):

    def load_food(self, dataset_dir, subset):
        """Load a subset of the Food dataset.
            dataset_dir: Root directory of the dataset.
            subset: Subset to load: train or val
            """
        # Add classes. We have only one class to add.
        # Add classes. We have only n+1 class to add.
        # Add classes
        # food_list = ['momo', "potato fries", "chawmin", "burger"]

        self.add_class("food", 0, "momo")
        self.add_class("food", 1, "potato fries")
        self.add_class("food", 2, "chawmin")
        self.add_class("food", 3, "burger")

        #
        # for n, i in enumerate(food_list):
        #     self.add_class("food", n + 1, i)


        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

                # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "food",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
           Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
            """
        # If not a food dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "food":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "food":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = FoodDataset()
    dataset_train.load_food(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FoodDataset()
    dataset_val.load_food(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='all')


# def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
#     """Runs official COCO evaluation.
#     dataset: A Dataset object with valiadtion data
#     eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
#     limit: if not 0, it's the number of images to use for evaluation
#     """
#     # Pick COCO images from the dataset
#     image_ids = image_ids or dataset.image_ids
#
#     # Limit to a subset
#     if limit:
#         image_ids = image_ids[:limit]
#
#     # Get corresponding COCO image IDs.
#     coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]
#
#     t_prediction = 0
#     t_start = time.time()
#
#     results = []
#     for i, image_id in enumerate(image_ids):
#         # Load image
#         image = dataset.load_image(image_id)
#
#         # Run detection
#         t = time.time()
#         r = model.detect([image], verbose=0)[0]
#         t_prediction += (time.time() - t)
#
#         # Convert results to COCO format
#         # Cast masks to uint8 because COCO tools errors out on bool
#         image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
#                                            r["rois"], r["class_ids"],
#                                            r["scores"],
#                                            r["masks"].astype(np.uint8))
#         results.extend(image_results)
#
#     # Load results. This modifies results with additional attributes.
#     coco_results = coco.loadRes(results)
#
#     # Evaluate
#     cocoEval = COCOeval(coco, coco_results, eval_type)
#     cocoEval.params.imgIds = coco_image_ids
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()
#
#     print("Prediction time: {}. Average {}/image".format(
#         t_prediction, t_prediction / len(image_ids)))
#     print("Total time: ", time.time() - t_start)


# def evalute(model):
#     """validate the model."""
#     # Validation dataset
#     dataset_val = FoodDataset()
#     dataset_val.load_food(args.dataset, "val")
#     dataset_val.prepare()
#
#     # *** This training schedule is an example. Update to your needs ***
#     # Since we're using a very small dataset, and starting from
#     # COCO trained weights, we don't need to train too long. Also,
#     # no need to train all layers, just the heads should do it.
#     print("evaluating network heads")
#     evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
#

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse


    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/food/dataset/",
                        help='datasets/newfood')
    parser.add_argument('--weights', required=True,
                        metavar=r"C:\Users\Mohan Kumar Dhakal\Desktop\Minor Project\Mask_RCNN\mask_rcnn_coco.h5",
                        help=r"C:\Users\Mohan Kumar Dhakal\Desktop\Minor Project\Mask_RCNN\mask_rcnn_coco.h5")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = FoodConfig()
    else:
        class InferenceConfig(FoodConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


    # Train or evaluate
    if args.command == "train":
        train(model)
        model.keras_model.save("mask_rcnn_momo_0005.h5")

    else:
        print("'{}' is not recognized. ""Use 'train' or 'splash'".format(args.command))
