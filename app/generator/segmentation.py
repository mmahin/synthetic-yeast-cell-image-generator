# segmentation.py
import detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def train_model(dataset_name):
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.OUTPUT_DIR = "./output"
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def evaluate_model(dataset_name):
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TEST = (dataset_name,)
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    return results

def predict(image):
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    return outputs