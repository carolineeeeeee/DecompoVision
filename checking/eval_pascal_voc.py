import base64
import io
import yaml
import argparse
import os
from tools.train_net import setup, Trainer, DetectionCheckpointer
from detectron2.evaluation.evaluator import inference_context, ExitStack
from detectron2.engine.defaults import DefaultTrainer
from detectron2.data.datasets import register_pascal_voc
from src.constant import *
import numpy as np
from detectron2.evaluation.pascal_voc_evaluation import *
from detectron2.data import MetadataCatalog
import torch
from torch import nn
import pandas as pd
from PIL import Image
from torchmetrics import JaccardIndex
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy
import pickle

BOX_IOU_THRESHOLD = 0.5
SEG_IOU_THRESHOLD = 0.25
SIGMA = 0.2

all_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def update_yaml_dataset_name(config_file_path, new_dataset_name: str):
    with open(config_file_path, 'r') as file:
        dict_file = yaml.full_load(file)
    if 'DATASETS' not in dict_file:
        dict_file['DATASETS'] = {}
    dict_file['DATASETS']['TEST'] = f'("{new_dataset_name}",)'
    with open(config_file_path, 'w') as file:
        yaml.dump(dict_file, file)
    MetadataCatalog.get(new_dataset_name).set(evaluator_type='pascal_voc')


def process(inputs, outputs, results_file_name): 
    final_output_results = {"image_id": [], "conf_score": [], "class": [], "xmin": [], "ymin": [], "xmax": [], "ymax": [], "masks":[]}
    for i in range(len(inputs)):
        image_id = inputs[i]["image_id"]
        instances = outputs[i]["instances"]
        boxes = instances._fields["pred_boxes"].tensor.cpu()
        conf_scores = instances._fields["scores"].cpu()
        pred_classes = instances._fields["pred_classes"].cpu() 
        
        if "pred_masks" in instances._fields and seg_results_path:
            pred_masks = instances._fields["pred_masks"].cpu()
        for j in range(boxes.shape[0]):         
            cur_box = boxes[j]
            xmin = cur_box[0].item()+1
            ymin = cur_box[1].item()+1
            xmax = cur_box[2].item()
            ymax = cur_box[3].item()
            mask_string = ''
            if int(xmax) > int(xmin) and int(ymax) > int(ymin):
                final_output_results["image_id"].append(image_id)
                final_output_results["conf_score"].append(conf_scores[j].item())
                final_output_results["xmin"].append(xmin)
                final_output_results["ymin"].append(ymin)
                final_output_results["xmax"].append(xmax)
                final_output_results["ymax"].append(ymax)
                final_output_results["class"].append(all_class_names[pred_classes[j].item()])
                if "pred_masks" in instances._fields:                    
                    mask = pred_masks[j,:,:].numpy()
                    mask = (mask != 0)
                    im = Image.fromarray(mask)
                    save_name = "temp.png"
                    im.save(save_name)
                    with open(save_name, "rb") as fid:
                        data = fid.read()

                    b64_bytes = base64.b64encode(data)
                    b64_string = b64_bytes.decode()
                    mask_string = b64_string
                   
                final_output_results["masks"].append(mask_string)

    df = pd.DataFrame(final_output_results)
    df.to_csv(results_file_name, mode='a', header=False)

def load_orig_det_as_gt(new_dataset_name, orig_results):
    class_names = MetadataCatalog.get(new_dataset_name).thing_classes
    all_recs = {}
    npos = {}
    for cls in class_names:
        npos[cls] = 0
    for d in range(len(orig_results)):
        image_id = orig_results['image_id'].iloc[d]
        if image_id not in all_recs:
            all_recs[image_id] = {'bbox': [], 'class': [], 'seg': []}
        bb = [orig_results['xmin'].iloc[d],orig_results['ymin'].iloc[d],orig_results['xmax'].iloc[d],orig_results['ymax'].iloc[d]]
        all_recs[image_id]['bbox'].append(bb)
        detected_class = orig_results['class'].iloc[d]
        all_recs[image_id]['class'].append(detected_class)
        all_recs[image_id]['seg'].append(orig_results['masks'].iloc[d]) 
        npos[detected_class] += 1
    return all_recs, npos


def load_gt(imagesetfile, annopath, class_names):
    imagenames = imagesetfile['image_id'].values.tolist()
    recs = {}
    all_recs = {}
    npos = {}
    for cls in class_names:
        npos[cls] = 0
    for imagename in imagenames:
        gt_mask_file = str(VOC_ROOT) + '/VOC2012/SegmentationObject/'+imagename+ '.png' 

        if os.path.exists(gt_mask_file):
            recs[imagename] = parse_rec(annopath.format(imagename))
        
            R = [obj for obj in recs[imagename]]
            bbox = np.array([x["bbox"] for x in R])

            class_names = [obj['name'] for obj in recs[imagename]]
            for cls in class_names:
                npos[cls] += 1
            
            mask_files = []
            index = 0
            for i in range(len(bbox)):
                gt_mask = np.asarray(Image.open(gt_mask_file))
                gt_mask = (gt_mask == (all_class_names.index(class_names[i])+1))

                save_name = 'temp.png'
                im = Image.fromarray(gt_mask)

                im.save(save_name)

                with open(save_name, "rb") as fid:
                    data = fid.read()

                b64_bytes = base64.b64encode(data)
                b64_string = b64_bytes.decode()
                mask_files.append(b64_string)

                index += 1
            all_recs[imagename] = {"bbox": bbox, 'class': class_names, 'seg': mask_files}

    return all_recs, npos

def voc_eval_l_process(final_output_df_csv_filename, gt, ovthresh=BOX_IOU_THRESHOLD):
    final_results = pd.read_csv(final_output_df_csv_filename) 
    final_results = final_results.sort_values(by='conf_score', ascending=False)
    nd = len(final_results)
    p = np.zeros(nd)
    p_gt = []
    p_seg = []

    for d in range(nd):
        image_id = final_results['image_id'].iloc[d]
        image_id = final_results['image_id'].iloc[d]

        if image_id not in gt:
            p_gt.append([])
            p_seg.append([])
            continue
        
        R = gt[image_id]
        BBGT = np.asarray(R["bbox"]).astype(float)
        ovmax = -np.inf
        bb = (final_results['xmin'].iloc[d],final_results['ymin'].iloc[d],final_results['xmax'].iloc[d],final_results['ymax'].iloc[d])
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            
            gt_classes_matched = [R["class"][i] for i in range(len(overlaps)) if overlaps[i] > ovthresh]
            gt_classes_matched = [x  if isinstance(x, str) else '' for x in gt_classes_matched]
            p_gt.append(','.join(gt_classes_matched))
            seg_matched = [R["seg"][i] for i in range(len(overlaps)) if overlaps[i] > ovthresh]
            seg_matched = [x  if isinstance(x, str) else '' for x in seg_matched]
            p_seg.append(' '.join(seg_matched))
            if ovmax > ovthresh:
                p[d] = 1.0

    final_results['l_IoU_p'] = p
    final_results['IoU_matched_gt'] = p_gt

    final_results = final_results.loc[:, ~final_results.columns.str.contains('^Unnamed')]
    final_results.to_csv(final_output_df_csv_filename, index=False)
    return p, p_gt, p_seg
    
def voc_eval_l_mAP(final_output_df_csv_filename, p, p_gt, cls, npos ):
    final_results = pd.read_csv(final_output_df_csv_filename) 
    nd = len(final_results)

    cd_equals_c = np.zeros(nd) # Precision (total predicted positive)

    # these will be per class
    tp_precision = np.zeros(nd)
    tp_recall = np.zeros(nd)
    for d in range(nd):
        # count cd = c -> Precision
        if final_results['class'].iloc[d] == cls:
            cd_equals_c[d] = 1.0
            if p[d] > 0:
                tp_precision[d] = 1.0
        # count c* = c -> Recall
        matched_gt = p_gt[d]
        if isinstance(matched_gt,str):
            if cls in (p_gt[d]).split(','):
                if p[d] > 0:
                    tp_recall[d] = 1.0

    rec_l = np.nan_to_num(np.cumsum(tp_recall)/npos)

    prec_l = np.nan_to_num(np.cumsum(tp_precision)/np.maximum(np.cumsum(cd_equals_c), np.finfo(np.float64).eps))

    return tp_recall, tp_precision, rec_l, prec_l, cd_equals_c

def voc_eval_c_given_l_mAP(final_output_df_csv_filename, p_gt, tp_recall, tp_precision, cls): 

    # on top of IoU being a p, if cd = c*: count
    final_results = pd.read_csv(final_output_df_csv_filename) 
    nd = len(final_results)
    p = np.zeros(nd)
    for d in range(nd):
        predicted_p = False
        gt_p = False
        if len(p_gt[d]) > 0: # IoU matched
            # count cd = c & IoU is a p -> Precision (previous p's)
            if final_results['class'].iloc[d] == cls:
                predicted_p = True
            # count c* = c & IoU is a p -> Recall (previous p's)
            if cls in (p_gt[d]).split(','):
                gt_p = True
            if predicted_p and gt_p:
                p[d] = 1.0
        
    sum_rec_l = np.cumsum(tp_recall)
    sum_prec_l = np.cumsum(tp_precision)

    tp = np.cumsum(p)
    rec_cl = np.nan_to_num(np.divide(tp, sum_rec_l))

    prec_cl = np.nan_to_num(tp/np.maximum(sum_prec_l, np.finfo(np.float64).eps))
    

    return p, rec_cl, prec_cl


def voc_eval_s_given_cl_mAP(final_output_df_csv_filename, p_gt, p_seg, p, cls, ovthersh=SEG_IOU_THRESHOLD):
    # on top of box IoU being a p and cd = c*, if seg IoU > 0.5, count
    final_results = pd.read_csv(final_output_df_csv_filename) 
    nd = len(final_results)
    tp = np.zeros(nd)

    for d in range(nd): 
        if p[d] > 0: # good detcetion box
            mask_file = final_results['masks'].iloc[d]
            try:
                if isinstance(mask_file, str):
                    mask = Image.open(io.BytesIO(base64.b64decode(mask_file)))
                    mask = np.asarray(mask)[int(final_results['ymin'].iloc[d]-1):int(final_results['ymax'].iloc[d]), int(final_results['xmin'].iloc[d]-1):int(final_results['xmax'].iloc[d])] 
                    
                    gt_mask_file = p_seg[d].split(' ')#'/w/10/users/boyue/VOCdevkit/VOC2012/SegmentationObject/'+image_id+ '.png' #TODO: make generic
                    gt_classes = p_gt[d].split(',')
                    gt_mask_file = gt_mask_file[gt_classes.index(cls)] 
               
                    if isinstance(gt_mask_file, str): 
                        gt_mask = Image.open(io.BytesIO(base64.b64decode(gt_mask_file)))
                        gt_mask = np.asarray(gt_mask)[int(final_results['ymin'].iloc[d]-1):int(final_results['ymax'].iloc[d]), int(final_results['xmin'].iloc[d]-1):int(final_results['xmax'].iloc[d])] 
                 
                        jaccard = JaccardIndex(num_classes=2)
                        iou = jaccard(torch.tensor(mask), torch.tensor(gt_mask))

                        if iou.item() > ovthersh:
                            tp[d] = 1.0
            except:
                pass

    good_detection = np.nan_to_num(p)
    good_detection = np.cumsum(good_detection)
    prec_scl = np.nan_to_num(np.divide(np.cumsum(tp), good_detection))
    rec_scl = np.nan_to_num(np.cumsum(tp)/np.maximum(good_detection, np.finfo(np.float64).eps))

    return rec_scl, prec_scl

def voc_eval_d_mAP(rec_l, prec_l, rec_cl, prec_cl):
    rec = np.multiply(rec_l, rec_cl)
    prec = np.multiply(prec_l, prec_cl)

    ap = voc_ap(rec, prec)
    return ap

def voc_eval_s_mAP(rec_l, prec_l, rec_cl, prec_cl, rec_scl, prec_scl):
    rec = np.multiply(np.multiply(rec_l, rec_cl), rec_scl)
    prec = np.multiply(np.multiply(prec_l, prec_cl), prec_scl)

    ap = voc_ap(rec, prec)

    return ap

def run_eval(args, new_dataset_name, final_output_df_csv_filename):
    cfg = setup(args)
    
    # build model and load model weights
    
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    # build dataloader and evaluator
    dataloader = DefaultTrainer.build_test_loader(cfg, new_dataset_name)
    
    final_output_fields = {"image_id": [], "conf_score": [], "class": [], "xmin": [], "ymin": [], "xmax": [], "ymax": [], "masks": []}
    
    final_output_df = pd.DataFrame(final_output_fields)
    final_output_df.to_csv(final_output_df_csv_filename)

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        for idx, inputs in enumerate(dataloader):
            image_id = inputs[0]['image_id']
            seg_gt = MetadataCatalog.get(new_dataset_name).dirname + '/SegmentationClass/' + image_id+".png"
            if not os.path.exists(seg_gt):
                continue
            outputs = model(inputs)
            process(inputs, outputs, final_output_df_csv_filename)

def estimate_curve_diff(rec_o, prec_o, rec_t, prec_t):
    # correct AP calculation
    # first append sentinel values at the end
    mrec_o = np.concatenate(([0.0], rec_o, [1.0]))
    mpre_o = np.concatenate(([0.0], prec_o, [0.0]))

    mrec_t = np.concatenate(([0.0], rec_t, [1.0]))
    mpre_t = np.concatenate(([0.0], prec_t, [0.0]))
    
    # sort them
    sorted_indices = np.argsort(mrec_o)
    mrec_o = mrec_o[sorted_indices]
    mpre_o = mpre_o[sorted_indices]
    sorted_indices = np.argsort(mrec_t)
    mrec_t = mrec_t[sorted_indices]
    mpre_t = mpre_t[sorted_indices]

    # compute the precision envelope
    for i in range(mpre_o.size - 1, 0, -1):
        mpre_o[i - 1] = np.maximum(mpre_o[i - 1], mpre_o[i])
    for i in range(mpre_t.size - 1, 0, -1):
        mpre_t[i - 1] = np.maximum(mpre_t[i - 1], mpre_t[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    
    i = np.where(mrec_o[1:] != mrec_o[:-1])[0]
    mrec_o = mrec_o[i]
    mpre_o = mpre_o[i]
    i = np.where(mrec_t[1:] != mrec_t[:-1])[0]
    mrec_t = mrec_t[i]
    mpre_t = mpre_t[i]
    

    if len(mrec_o) > 3 and len(mrec_t) > 3:
        tck_o, u  = interpolate.splprep([mrec_o, mpre_o], s=0)
        xi = np.sort(np.asarray(list(set(list(mrec_o) + list(mrec_t)))))
        yi = interpolate.splev(xi, tck_o, der=0) 

        tck_t, u  = interpolate.splprep([mrec_t, mpre_t],s=0)
        yi_t = interpolate.splev(xi, tck_t, der=0)

        difference = np.asarray(yi_t) - np.asarray(yi)
        mu, sigma = scipy.stats.norm.fit(difference)
        distribution = scipy.stats.norm(mu, sigma)
        result = distribution.cdf(0)

        return np.nan_to_num(result), len(xi)
    else:
        return None, None

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transformation", required=True)#, choices=TRANSFORMATIONS)
    parser.add_argument("-r", "--read_only", type=bool, required=False)
    parser.add_argument("-v", "--vision_task", required=True, choices=['D', 'I'])
    parser.add_argument("-th", "--threshold", required=True)
    
    
    args = parser.parse_args()

    if args.vision_task == 'D':
        human_thld = human_thld_D[args.transformation]
    else:
        human_thld = human_thld_I[args.transformation]

    MVC_output_path = ''
    for cls in human_thld.keys():
        for vt in human_thld[cls].keys():
            for req in human_thld[cls][vt].keys():
                if float(args.threshold) in human_thld[cls][vt][req]:
                    MVC_output_path = human_thld[cls][vt][req][float(args.threshold)]
                    break
    assert MVC_output_path != ''

    # check what needs to run
    run = {}
    for cls in human_thld.keys():
        run[cls] = {}
        for req in ['cp', 'pp']:
            if args.vision_task == 'D':
                if float(args.threshold) in human_thld[cls]['L'][req] or float(args.threshold) in human_thld[cls]['C|L'][req] or float(args.threshold) in human_thld[cls]['D'][req]:
                    run[cls][req+'_L'] = True
                else:
                    run[cls][req+'_L'] = False
                if float(args.threshold) in human_thld[cls]['C|L'][req] or float(args.threshold) in human_thld[cls]['D'][req]:
                    run[cls][req+'_C|L'] = True
                else:
                    run[cls][req+'_C|L'] = False
                if float(args.threshold) in human_thld[cls]['D'][req]:
                    run[cls][req+'_D'] = True
                else:
                    run[cls][req+'_D'] = False
            if args.vision_task == 'I':
                if float(args.threshold) in human_thld[cls]['L'][req] or float(args.threshold) in human_thld[cls]['C|L'][req] or float(args.threshold) in human_thld[cls]['S|C,L'][req] or float(args.threshold) in human_thld[cls]['I'][req]:
                    run[cls][req+'_L'] = True
                else:
                    run[cls][req+'_L'] = False
                if float(args.threshold) in human_thld[cls]['C|L'][req] or float(args.threshold) in human_thld[cls]['S|C,L'][req] or float(args.threshold) in human_thld[cls]['I'][req]:
                    run[cls][req+'_C|L'] = True
                else:
                    run[cls][req+'_C|L'] = False
                if float(args.threshold) in human_thld[cls]['S|C,L'][req] or float(args.threshold) in human_thld[cls]['I'][req]:
                    run[cls][req+'_S|C,L'] = True
                else:
                    run[cls][req+'_S|C,L'] = False
                if float(args.threshold) in human_thld[cls]['I'][req]:
                    run[cls][req+'_I'] = True
                else:
                    run[cls][req+'_I'] = False

    args.dist_url = 'tcp://127.0.0.1:50152'
    args.eval_only = True
    args.machine_rank = 0
    args.num_gpus = 1
    args.resume = False
    num_machines = 1
    list_config = ["detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml", 
    "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
    "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
    "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
    "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
    "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
    "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"]
    
    list_weights = ['model_weights/R50-C4_1x_MAXiter50000.pth',
    'model_weights/R50-DC5_1x_MAXiter50000.pth',
    'model_weights/R50-FPN_1x_MAXiter50000.pth',
    'model_weights/R50-C4_3x_MAXiter50000.pth',
    'model_weights/R50-DC5_3x_MAXiter50000.pth',
    'model_weights/R50-FPN_3x_MAXiter50000.pth',
    'model_weights/R101-C4_3x_MAXiter50000.pth',
    'model_weights/R101-DC5_3x_MAXiter50000.pth',
    'model_weights/R101-FPN_3x_MAXiter50000.pth',
    'model_weights/X101-FPN_3x_MAXiter50000.pth']

    assert len(list_config) == len(list_weights)

    all_final = {}
   
    for model_index in range(len(list_config)):
        if model_index not in all_final:
            all_final[model_index] = {}     

        args.config_file = list_config[model_index]
        args.opts = ['MODEL.WEIGHTS',
                list_weights[model_index], 'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5, 'MODEL.ROI_HEADS.NUM_CLASSES', 20, 'SOLVER.BASE_LR', 0.00025]

        bootstrap_data_dir = BOOTSTRAP_DIR / f'bootstrap-{MVC_output_path}'
        if not os.path.exists('results_csv'):
            os.mkdir('results_csv')
        if not os.path.exists('results_csv/bootstrap-'+MVC_output_path):
            os.mkdir('results_csv/bootstrap-'+MVC_output_path)
        if not os.path.exists('results_csv/bootstrap-'+MVC_output_path+ '/model_' + str(model_index)):
            os.mkdir('results_csv/bootstrap-'+MVC_output_path + '/model_' + str(model_index))
        csv_path = 'results_csv/bootstrap-'+MVC_output_path + '/model_' + str(model_index) + '/'

        # run all original images
        new_dataset_name = f"my_voc_orig"+ '_' + str(model_index)
        voc_root = str(VOC_ROOT) + 'VOC2012/'

        update_yaml_dataset_name(args.config_file, new_dataset_name)
        register_pascal_voc(new_dataset_name, dirname=voc_root, split="seg", year=2007)
        orig_final_output_df_csv_filename = csv_path+ '/all_orig.csv'   
        if not args.read_only:     
            run_eval(args, new_dataset_name, orig_final_output_df_csv_filename)
        
        cp_transf_all = []
        cp_orig_all = []
        for i, root in enumerate(bootstrap_data_dir.iterdir()):
            if i not in all_final[model_index]:
                all_final[model_index][i] = {'cp_l_results' : [], 'cp_cl_results' : [], 'cp_scl_results' : [], 'cp_d_results' : [], 'cp_s_results' : [], 'cp_d_mAP' : [], 'cp_s_mAP' : [],
                                            'pp_l_results' : [], 'pp_cl_results' : [], 'pp_scl_results' : [], 'pp_d_results' : [], 'pp_s_results' : [], 'pp_d_mAP' : [], 'pp_s_mAP' : []}
            voc_root = str(bootstrap_data_dir)  + f"/iter{i + 1}" # where the iteration of bootstrap is
            bootstrap_df = pd.read_csv("bootstrap_dfs/bootstrap_df-"+MVC_output_path+ ".csv")
            this_iteration = bootstrap_df.loc[bootstrap_df['iteration_id'] == i+1]
            csv_path = 'results_csv/bootstrap-'+MVC_output_path + '/model_' + str(model_index) + '/'
        
            args.config_file = list_config[model_index]
            args.opts = ['MODEL.WEIGHTS',
                list_weights[model_index], 'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5, 'MODEL.ROI_HEADS.NUM_CLASSES', 20, 'SOLVER.BASE_LR', 0.00025]
            # FOR correctness_preservation
            # run bootstrapped transformed images with real ground truth    
            cp_transf_csv_filename = csv_path +"cp_transf" + "_iter_"+ str(i+1) + ".csv"

            # register custom dataset
            new_dataset_name = f"my_voc_{i}" + '_' +str(model_index)
            update_yaml_dataset_name(args.config_file, new_dataset_name)
            register_pascal_voc(new_dataset_name, dirname=voc_root, split="val", year=2007)
            if not args.read_only:     
                run_eval(args, new_dataset_name, cp_transf_csv_filename)
            # getting all images sampled in this iteration and load their ground truth
            # process them

            # filter threshold: create new file with threshold
            all_transf_results = pd.read_csv(cp_transf_csv_filename)
            less_than_thld = this_iteration.loc[this_iteration['vd_score'] <= float(args.threshold)] # all images < threshold
            # find transformed < sigma
            results_this_iteration = all_transf_results.loc[all_transf_results['image_id'].isin(list(less_than_thld['image_id']))]
            cp_transf_csv_filename = csv_path +"cp_transf" + "_iter_"+ str(i+1) + '_' + args.threshold + ".csv"
            results_this_iteration.to_csv(cp_transf_csv_filename, index=False)


            all_orig = pd.read_csv(orig_final_output_df_csv_filename) 
            results_this_iteration = all_orig.loc[all_orig['image_id'].isin(list(less_than_thld['image_id']))]

            annopath = os.path.join(voc_root, "Annotations", "{}.xml")
            gt, npos_gt = load_gt(less_than_thld, annopath, all_class_names)
            cp_orig_csv_file = csv_path + "cp_orig" + "_iter_"+ str(i+1) + ".csv"
            results_this_iteration.to_csv(cp_orig_csv_file, index=False)
            
            o_pl, o_p_gt, o_p_seg = voc_eval_l_process(cp_orig_csv_file, gt)
            t_pl, t_p_gt, t_p_seg = voc_eval_l_process(cp_transf_csv_filename, gt)
            
            # for prediction-preservation

            # eval transformed with original as gt
            orig_as_gt, npos_orig_as_gt = load_orig_det_as_gt(new_dataset_name, results_this_iteration)
            st_pl, st_p_gt, st_p_seg = voc_eval_l_process(cp_transf_csv_filename, orig_as_gt)
            
            # eval less than sigma as orig

            # find less than sigma, to update original to gt, need to change names to ids
            less_than_sigma = this_iteration.loc[this_iteration['vd_score'] < SIGMA] # all images < sigma
            # find transformed < sigma
            results_this_iteration_l_sigma = all_transf_results.loc[all_transf_results['image_id'].isin(list(less_than_sigma['image_id']))]
            less_than_sigma_csv_file = csv_path + "pp_sigma_" + str(SIGMA) +  "_iter_"+ str(i+1) + ".csv"
            results_this_iteration_l_sigma.to_csv(less_than_sigma_csv_file, index=False)
            # find original < sigma
            orig_results_this_iteration_l_sigma = all_orig.loc[all_orig['image_id'].isin(list(less_than_sigma['image_id']))]
            orig_less_than_sigma_csv_file = csv_path + "pp_sigma_orig_" + str(SIGMA) +  "_iter_"+ str(i+1) + ".csv"
            orig_results_this_iteration_l_sigma.to_csv(orig_less_than_sigma_csv_file, index=False)
            
            sigma_as_gt, npos_sigma_as_gt = load_orig_det_as_gt(new_dataset_name, orig_results_this_iteration_l_sigma)
            s_pl, s_p_gt, s_p_seg = voc_eval_l_process(less_than_sigma_csv_file, sigma_as_gt)


            for cls in run.keys():
                # correctness-preservation
                # calculate PR of l 
                if run[cls]['cp_L']:
                    o_tp_recall, o_tp_precision, o_rec_l, o_prec_l, cd_equals_c_o = voc_eval_l_mAP(cp_orig_csv_file, o_pl, o_p_gt, cls, npos_gt[cls])
                    t_tp_recall, t_tp_precision, t_rec_l, t_prec_l, cd_equals_c_t = voc_eval_l_mAP(cp_transf_csv_filename, t_pl, t_p_gt, cls, npos_gt[cls])

                    try:
                        conf_ninety_five, num_points = estimate_curve_diff(o_rec_l, o_prec_l, t_rec_l, t_prec_l, cd_equals_c_t, npos_gt[cls])
                        if conf_ninety_five:
                            all_final[model_index][i]['cp_l_results'].append((cls, conf_ninety_five, num_points))
                            with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                                    pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass
                
                # calculate PR of c|l  
                if run[cls]['cp_C|L']:
                    o_p, o_rec_cl, o_prec_cl = voc_eval_c_given_l_mAP(cp_orig_csv_file, o_p_gt, o_tp_recall, o_tp_precision, cls)
                    t_p, t_rec_cl, t_prec_cl = voc_eval_c_given_l_mAP(cp_transf_csv_filename, t_p_gt, t_tp_recall, t_tp_precision, cls)
                    
                    try:
                        conf_ninety_five, num_points = estimate_curve_diff(o_rec_cl, o_prec_cl, t_rec_cl, t_prec_cl, t_tp_precision, npos_gt[cls])
                        if conf_ninety_five:
                            all_final[model_index][i]['cp_cl_results'].append((cls, conf_ninety_five, num_points))
                            with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                                pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass
                
                if args.vision_task == 'D' and run[cls]['cp_D']:
                    o_rec_d = np.nan_to_num(np.multiply(o_rec_l, o_rec_cl))
                    o_prec_d = np.nan_to_num(np.multiply(o_prec_l, o_prec_cl))
                    t_rec_d = np.nan_to_num(np.multiply(t_rec_l, t_rec_cl))
                    t_prec_d = np.nan_to_num(np.multiply(t_prec_l, t_prec_cl))
                    
                    try:
                        conf_ninety_five, num_points = estimate_curve_diff(o_rec_d, o_prec_d, t_rec_d, t_prec_d, cd_equals_c_t, npos_gt[cls])
                        if conf_ninety_five:
                            all_final[model_index][i]['cp_d_results'].append((cls, conf_ninety_five, num_points))
                            with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                                pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        mAP_o = voc_ap(o_rec_d, o_prec_d)
                        mAP_t = voc_ap(t_rec_d, t_prec_d)
                        all_final[model_index][i]['cp_d_mAP'].append((cls, mAP_o, mAP_t))
                        with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass


                if args.vision_task == 'I' and run[cls]['cp_S|C,L']:
                    o_rec_scl, o_prec_scl = voc_eval_s_given_cl_mAP(cp_orig_csv_file, o_p_gt, o_p_seg, o_p, cls)
                    t_rec_scl, t_prec_scl = voc_eval_s_given_cl_mAP(cp_transf_csv_filename, t_p_gt, t_p_seg, t_p, cls)
                    
                    try:
                        conf_ninety_five, num_points = estimate_curve_diff(o_rec_scl, o_prec_scl, t_rec_scl, t_prec_scl, t_p, npos_gt[cls])
                        if conf_ninety_five:
                            all_final[model_index][i]['cp_scl_results'].append((cls, conf_ninety_five, num_points))
                            with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                                pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass
                
                if args.vision_task == 'I' and run[cls]['cp_I']:
                    o_rec_s = np.nan_to_num(np.multiply(np.multiply(o_rec_l, o_rec_cl), o_rec_scl))
                    o_prec_s = np.nan_to_num(np.multiply(np.multiply(o_prec_l, o_prec_cl), o_prec_scl))
                    t_rec_s = np.nan_to_num(np.multiply(np.multiply(t_rec_l, t_rec_cl), t_rec_scl))
                    t_prec_s = np.nan_to_num(np.multiply(np.multiply(t_prec_l, t_prec_cl), t_prec_scl))
                    
                    try:
                        conf_ninety_five, num_points = estimate_curve_diff(o_rec_s, o_prec_s, t_rec_s, t_prec_s, cd_equals_c_t, npos_gt[cls])
                        if conf_ninety_five:
                            all_final[model_index][i]['cp_s_results'].append((cls, conf_ninety_five, num_points))
                            with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                                pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass

                    
                    mAP_o = voc_ap(o_rec_s, o_rec_s)
                    mAP_t = voc_ap(t_rec_s, t_prec_s)

                    all_final[model_index][i]['cp_s_mAP'].append((cls, mAP_o, mAP_t))
                    with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                        pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

                
                
                # prediction-preservation
                # calculate PR of l 
                if run[cls]['pp_L']:
                    s_tp_recall, s_tp_precision, s_rec_l, s_prec_l, cd_equals_c_s = voc_eval_l_mAP(less_than_sigma_csv_file, s_pl, s_p_gt, cls, npos_sigma_as_gt[cls])
                    t_tp_recall, t_tp_precision, t_rec_l, t_prec_l, cd_equals_c_t = voc_eval_l_mAP(cp_transf_csv_filename, st_pl, st_p_gt, cls, npos_orig_as_gt[cls])
                    
                    try:
                        conf_ninety_five, num_points = estimate_curve_diff(s_rec_l, s_prec_l, t_rec_l, t_prec_l, cd_equals_c_t, npos_orig_as_gt[cls])
                        if conf_ninety_five:
                            all_final[model_index][i]['pp_l_results'].append((cls, conf_ninety_five, num_points))
                            with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                                pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass
                        
                
                if run[cls]['pp_C|L']:

                    # calculate PR of c|l 
                    s_p, s_rec_cl, s_prec_cl = voc_eval_c_given_l_mAP(less_than_sigma_csv_file, s_p_gt, s_tp_recall, s_tp_precision, cls)
                    t_p, t_rec_cl, t_prec_cl = voc_eval_c_given_l_mAP(cp_transf_csv_filename, st_p_gt, t_tp_recall, t_tp_precision, cls)
                  
                    try:
                        conf_ninety_five, num_points = estimate_curve_diff(s_rec_cl, s_prec_cl, t_rec_cl, t_prec_cl, t_tp_precision, npos_orig_as_gt[cls])
                        
                        if conf_ninety_five:
                            all_final[model_index][i]['pp_cl_results'].append((cls, conf_ninety_five, num_points))
                            with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                                pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass
                
                # calculate mAP of d 
               
                if args.vision_task == 'D' and run[cls]['pp_D']:
                    s_rec_d = np.nan_to_num(np.multiply(s_rec_l, s_rec_cl))
                    s_prec_d = np.nan_to_num(np.multiply(s_prec_l, s_prec_cl))
                    t_rec_d = np.nan_to_num(np.multiply(t_rec_l, t_rec_cl))
                    t_prec_d = np.nan_to_num(np.multiply(t_prec_l, t_prec_cl))
                    try:
                        conf_ninety_five, num_points = estimate_curve_diff(s_rec_d, s_prec_d, t_rec_d, t_prec_d, cd_equals_c_t, npos_orig_as_gt[cls])
                        if conf_ninety_five:
                            all_final[model_index][i]['pp_d_results'].append((cls, conf_ninety_five, num_points))
                            with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                                pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass

                    mAP_s = voc_ap(s_rec_d, s_prec_d)
                    mAP_t = voc_ap(t_rec_d, t_prec_d)
                    all_final[model_index][i]['pp_d_mAP'].append((cls, mAP_s, mAP_t))
                    with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                        pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

                if args.vision_task == 'I' and run[cls]['pp_S|C,L']:
                    s_rec_scl, s_prec_scl = voc_eval_s_given_cl_mAP(less_than_sigma_csv_file, s_p_gt, s_p_seg, s_p, cls)
                    t_rec_scl, t_prec_scl = voc_eval_s_given_cl_mAP(cp_transf_csv_filename, st_p_gt, st_p_seg, t_p, cls)
                    
                    try:
                        conf_ninety_five, num_points = estimate_curve_diff(s_rec_scl, s_prec_scl, t_rec_scl, t_prec_scl, t_p, npos_orig_as_gt[cls])
                        if conf_ninety_five:
                            all_final[model_index][i]['pp_scl_results'].append((cls, conf_ninety_five, num_points))
                            with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) +  '_bootstrap.pickle', 'wb') as handle:
                                pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass
            

                # calculate mAP of s
                if args.vision_task == 'I' and run[cls]['pp_I']:
                    s_rec_s = np.nan_to_num(np.multiply(np.multiply(s_rec_l, s_rec_cl), s_rec_scl))
                    s_prec_s = np.nan_to_num(np.multiply(np.multiply(s_prec_l, s_prec_cl), s_prec_scl))
                    t_rec_s = np.nan_to_num(np.multiply(np.multiply(t_rec_l, t_rec_cl), t_rec_scl))
                    t_prec_s = np.nan_to_num(np.multiply(np.multiply(t_prec_l, t_prec_cl), t_prec_scl))
                    
                    try:
                        conf_ninety_five, num_points = estimate_curve_diff(s_rec_s, s_prec_s, t_rec_s, t_prec_s, cd_equals_c_t, npos_orig_as_gt[cls])
                        if conf_ninety_five:
                            all_final[model_index][i]['pp_s_results'].append((cls, conf_ninety_five, num_points))
                            with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                                pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass
                    
                    mAP_s = voc_ap(s_rec_s, s_prec_s)
                    mAP_t = voc_ap(t_rec_s, t_prec_s)
                    all_final[model_index][i]['pp_s_mAP'].append((cls, mAP_s, mAP_t))
                    with open(args.vision_task + '_' + cls+'_'+args.transformation + '_' + str(args.threshold) + '_bootstrap.pickle', 'wb') as handle:
                        pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
