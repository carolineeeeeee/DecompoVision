# frost 9 is 0.9, frost 7 is 0.7, frost 8 is 0.5 in screen 8, frost 6 is 0.3 in screen 7
import sys
sys.path.insert(0,'/u/boyue/Detection/MVC_reliability_detection/object_detection/detectron2')
import base64
import io
#from sympy import Q
import tracemalloc
import time
import yaml
import argparse
import pathlib2
from pathlib2 import Path
import os
#from detectron2.data.datasets import register_coco_instances
from tools.train_net import setup, Trainer, DetectionCheckpointer
from tools.train_net import build_evaluator
from detectron2.evaluation.evaluator import inference_on_dataset, inference_context, ExitStack
from detectron2.engine.defaults import DefaultTrainer, DefaultPredictor
#from detectron2.evaluation import verify_results
#import detectron2.utils.comm as comm
from detectron2.data.datasets import register_pascal_voc
from src.constant import BOOTSTRAP_DIR, GAUSSIAN_NOISE, TRANSFORMATIONS, ROOT
import numpy as np
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.pascal_voc_evaluation import *
from detectron2.data import MetadataCatalog
import torch
from torch import nn
import pandas as pd
#from pycocotools.cocostuffhelper import pngToCocoResult, cocoSegmentationToPng, cocoSegmentationToSegmentationMap
from PIL import Image
from torchmetrics import JaccardIndex
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats
import scipy
import pickle

#config_file_path = str(pathlib2.Path(
#    __file__).absolute().parent / 'detectron2' / 'configs' / 'PascalVOC-Detection' / 'faster_rcnn_R_50_C4.yaml')

BOX_IOU_THRESHOLD = 0.5
SEG_IOU_THRESHOLD = 0.25
SIGMA = 0.4

all_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def update_yaml_dataset_name(config_file_path, new_dataset_name: str):
    #(config_file_path)
    with open(config_file_path, 'r') as file:
        dict_file = yaml.full_load(file)
    if 'DATASETS' not in dict_file:
        dict_file['DATASETS'] = {}
    dict_file['DATASETS']['TEST'] = f'("{new_dataset_name}",)'
    with open(config_file_path, 'w') as file:
        yaml.dump(dict_file, file)
    MetadataCatalog.get(new_dataset_name).set(evaluator_type='pascal_voc')


def process(inputs, outputs, results_file_name, seg_results_path=None): # collect results
    # parse outputs
    final_output_results = {"image_id": [], "conf_score": [], "class": [], "xmin": [], "ymin": [], "xmax": [], "ymax": [], "masks":[]}
    #for i in range(len(inputs)):
    for i in range(len(inputs)):
        image_id = inputs[i]["image_id"]
        #print(image_id)
        #print(outputs)
        #exit()
        instances = outputs[i]["instances"]
        boxes = instances._fields["pred_boxes"].tensor.cpu()
        conf_scores = instances._fields["scores"].cpu()
        pred_classes = instances._fields["pred_classes"].cpu() # TODO: \emove the ones not under 20
        #print(pred_classes)
        
        if "pred_masks" in instances._fields and seg_results_path:
            #final_output_results["mask"] = []
            pred_masks = instances._fields["pred_masks"].cpu()
        for j in range(boxes.shape[0]):
            if pred_classes[j].item() > 20:
                continue            
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
                if "pred_masks" in instances._fields and seg_results_path:                    
                    #print(cur_box)
                    #print(pred_masks[j,:,:].numpy().shape)
                    mask = pred_masks[j,:,:].numpy()#[ int(cur_box[1].item()):int(cur_box[3].item()), int(cur_box[0].item()):int(cur_box[2].item())] 

                    mask = (mask != 0)
                    im = Image.fromarray(mask)
                    #
                    #im = Image.fromarray(pred_masks[j,:,:].numpy())
                    #im.save(seg_results_path+'/'+image_id + ".png")
                    save_name = "temp.png"
                    im.save(save_name)
                    with open(save_name, "rb") as fid:
                        data = fid.read()

                    b64_bytes = base64.b64encode(data)
                    b64_string = b64_bytes.decode()
                    #print(b64_string)
                    #exit()
                    #if "pred_masks" in instances._fields:
                    mask_string = b64_string
                    #final_output_results["masks"].append(b64_string)
                    #print(final_output_results["masks"])
                    #exit()
                    #break
                #else:
                    #final_output_results["masks"].append('')
                final_output_results["masks"].append(mask_string)
        #exit()

    df = pd.DataFrame(final_output_results)
    #print(results_file_name)
    #print(df)
    df.to_csv(results_file_name, mode='a', header=False)

def load_orig_det_as_gt(new_dataset_name, orig_results):
    #print(orig_results)
    #all_image_id = orig_results['image_id']
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
        #print(all_recs[image_id])
    return all_recs, npos


def load_gt(imagesetfile, annopath, class_names):
    #print('------------load_gt----------')
    # first load gt
    # read list of images

    #with PathManager.open(imagesetfile, "r") as f: 
    #    lines = f.readlines()
    #results = pd.read_csv(results_csv)
    #lines = set(results['image_id'].tolist())
    
    imagenames = imagesetfile['image_id'].values.tolist()#[x.strip() for x in lines]
    #print(imagenames)
    # load annots
    recs = {}
    all_recs = {}
    #class_recs = {}
    #npos = 0
    npos = {}
    for cls in class_names:
        npos[cls] = 0
    for imagename in imagenames:
        gt_mask_file = '/w/10/users/boyue/VOCdevkit/VOC2012/SegmentationObject/'+imagename+ '.png' #TODO: make generic
        
                #mask = np.asarray(mask)[int(final_results['ymin'].iloc[d]-1):int(final_results['ymax'].iloc[d]), int(final_results['xmin'].iloc[d]-1):int(final_results['xmax'].iloc[d])] 
                #print(mask.shape)

        #print(gt_mask_file)
        if os.path.exists(gt_mask_file):
            recs[imagename] = parse_rec(annopath.format(imagename))
        
        #npos = 0
        #for imagename in imagenames:
            R = [obj for obj in recs[imagename]]
            #print(R)
            bbox = np.array([x["bbox"] for x in R])
            #difficult = np.array([x["difficult"] for x in R]).astype(bool)
            # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
            #det = [False] * len(R)
            #npos = npos + sum(~difficult)
            
            class_names = [obj['name'] for obj in recs[imagename]]#tp_gt = gt[image_id][id_ovmax]['name']
            #all_count = all_count + len(class_names)
            for cls in class_names:
                npos[cls] += 1
            
            mask_files = []
            index = 0
            for i in range(len(bbox)):
                bb = bbox[i]

                #print(bb)
                #gt_mask = np.asarray(Image.open(gt_mask_file))
                #print(gt_mask.shape)
                #gt_mask = gt_mask[int(bb[1]-1):int(bb[3]), int(bb[0]-1):int(bb[2])]
                #im = Image.fromarray(gt_mask)
                #if not os.path.exists('/w/10/users/boyue/bootstrap-frost/gt'):
                #    os.mkdir('/w/10/users/boyue/bootstrap-frost/gt')
                #save_name = '/w/10/users/boyue/bootstrap-frost/gt/' + imagename + '-' + str(index) + '.png'
                #im.save(save_name) # TODO: make generic
                #mask_files.append(save_name)

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
                #print(b64_string)

                index += 1
            all_recs[imagename] = {"bbox": bbox, 'class': class_names, 'seg': mask_files}
    #print(npos)
    #exit()
    return all_recs, npos

def voc_eval_l_process(final_output_df_csv_filename, gt, ovthresh=BOX_IOU_THRESHOLD):
    #print('------------l_process----------')
    #print(final_output_df_csv_filename)
    final_results = pd.read_csv(final_output_df_csv_filename) 
    final_results = final_results.sort_values(by='conf_score', ascending=False)
    nd = len(final_results)
    p = np.zeros(nd)
    #fp = np.zeros(nd)
    p_gt = []
    p_seg = []
    # filter detection of this cls first
    #print(final_results)
    #exit()

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
                #print(d, gt_classes_matched, final_results['class'].iloc[d])

    final_results['l_IoU_p'] = p
    final_results['IoU_matched_gt'] = p_gt
    #print(final_results)
    
    #print(final_results)
    final_results = final_results.loc[:, ~final_results.columns.str.contains('^Unnamed')]
    final_results.to_csv(final_output_df_csv_filename, index=False)
    return p, p_gt, p_seg
    
def voc_eval_l_mAP(final_output_df_csv_filename, p, p_gt, cls, npos ):
    #print('-----voc_eval_l_mAP--------')
    #print(cls, cls_index)
    # cumsup of p's then
    final_results = pd.read_csv(final_output_df_csv_filename) 
    nd = len(final_results)
    #print(nd)

    cd_equals_c = np.zeros(nd) # Precision (total predicted positive)

    # these will be per class
    tp_precision = np.zeros(nd)
    tp_recall = np.zeros(nd)
    for d in range(nd):
        image_id = final_results['image_id'].iloc[d]

        #print(final_results['class'].iloc[d], final_results['IoU_matched_gt'].iloc[d], final_results['l_IoU_p'].iloc[d])
        # count cd = c -> Precision
        if final_results['class'].iloc[d] == cls:
            cd_equals_c[d] = 1.0
            if p[d] > 0:
                tp_precision[d] = 1.0
        # count c* = c -> Recall
        #print(final_results['IoU_matched_gt'].iloc[d])
        matched_gt = p_gt[d]
        if isinstance(matched_gt,str):
            if cls in (p_gt[d]).split(','):
                #c_star_equals_c += 1
                if p[d] > 0:
                    tp_recall[d] = 1.0

    #conf_scores = final_results['conf_score'].to_numpy()
    #sorted_ind = np.argsort(-conf_scores)

    #tp_recall = tp_recall[sorted_ind]
    #tp_precision = tp_precision[sorted_ind]
    #cd_equals_c = cd_equals_c[sorted_ind]

    #sum_rec_l = np.cumsum(tp_recall)
    rec_l = np.nan_to_num(np.cumsum(tp_recall)/npos)
    
    #print(rec_l)
    #rec_l = sum_l_recall/npos
    
    #sum_prec_l = np.cumsum(tp_precision)
    #print(sum_prec_l)
    prec_l = np.nan_to_num(np.cumsum(tp_precision)/np.maximum(np.cumsum(cd_equals_c), np.finfo(np.float64).eps))

    #print(rec_l)
    #print(prec_l)
    #print(voc_ap(rec_l, prec_l))
    #print(prec_l)
    #print('------------------------')
    #final_results['sum_rec_l_'+cls] = sum_rec_l
    #final_results['rec_l_'+cls] = rec_l
    #final_results['sum_prec_l_'+cls] = sum_prec_l
    #final_results['prec_l_'+cls] = prec_l
    #final_results = final_results.loc[:, ~final_results.columns.str.contains('^Unnamed')]
    #final_results.to_csv(final_output_df_csv_filename, index=False)
    return tp_recall, tp_precision, rec_l, prec_l, cd_equals_c#sum_rec_l, sum_prec_l, rec_l, prec_l


def voc_eval_c_given_l_mAP(final_output_df_csv_filename, p_gt, tp_recall, tp_precision, cls): 
    #print('-------voc_eval_c_given_l_mAP---------')
    #print(cls, cls_index)
    # on top of IoU being a p, if cd = c*: count
    final_results = pd.read_csv(final_output_df_csv_filename) 
    #print(len(l_p), len(final_results))
    nd = len(final_results)
    #p_r = np.zeros(nd)
    #p_p = np.zeros(nd)
    p = np.zeros(nd)
    #not_empty = [x for x in p_gt if len(x) != 0]
    #print(len(not_empty))
    #exit()
    for d in range(nd):
        image_id = final_results['image_id'].iloc[d]

        predicted_p = False
        gt_p = False
        if len(p_gt[d]) > 0: # IoU matched
            # count cd = c & IoU is a p -> Precision (previous p's)
            #print(cls, final_results['class'].iloc[d])
            if final_results['class'].iloc[d] == cls:
                predicted_p = True
            # count c* = c & IoU is a p -> Recall (previous p's)
            #print(cls, p_gt[d])
            if cls in (p_gt[d]).split(','):
                #c_star_equals_c += 1
                gt_p = True
            if predicted_p and gt_p:
                p[d] = 1.0
                #print("found good box")
        '''
        #print(d, cls, final_results['class'].iloc[d], p_gt[d])
        if tp_recall[d] > 0: # IoU matched and gt matched the class
            # need to check if predicted is this class
            if final_results['class'].iloc[d] == cls:
                p_r[d] = 1.0
        if tp_precision[d] > 0: # IoU matched and predicted this class
            # need to check if the gt is this class
            #print(cls, [gt['class'][i] for i in p_gt[d]])
            if cls in (p_gt[d]).split(','):
                p_p[d] = 1.0
       
    #print(sum(tp_recall), sum(tp_precision))
    sum_rec_l = np.cumsum(tp_recall)
    sum_prec_l = np.cumsum(tp_precision)
    
    #tp_to_return = p # shared by precision and recall
    #cd_equals_c = cd_equals_c[sorted_ind] # for precision
    #print(p)
    #print(sum(p))
    rec_cl = np.nan_to_num(np.divide(np.cumsum(p_r), sum_rec_l))

    prec_cl = np.nan_to_num(np.cumsum(p_p)/np.maximum(sum_prec_l, np.finfo(np.float64).eps))
    print(sum(p_r), sum(p_p))
    #print(rec_cl, prec_cl)
    exit()
    
    for d in range(nd):
        predicted_p = False
        gt_p = False
        
        #print(final_results['class'].iloc[d], final_results['l_IoU_p'].iloc[d], final_results['IoU_matched_gt'].iloc[d])
        # counting given IoU and label
        if p_gt[d] > 0: # IoU matched
            # count cd = c & IoU is a p -> Precision (previous p's)
            if final_results['class'].iloc[d] == cls_index:
                predicted_p = True
            # count c* = c & IoU is a p -> Recall (previous p's)
            if cls in (p_gt[d]).split(','):
                #c_star_equals_c += 1
                gt_p = True
            if predicted_p and gt_p:
                p[d] = 1.0
                #print("found good box")

    #conf_scores = final_results['conf_score'].to_numpy()
    #sorted_ind = np.argsort(-conf_scores)
    
    sum_rec_l = np.cumsum(tp_recall)#final_results['sum_rec_l_' +cls].to_numpy()
    sum_prec_l = np.cumsum(tp_precision)#final_results['sum_prec_l_' +cls].to_numpy()
    '''
    #tp_to_return = p # shared by precision and recall
    #cd_equals_c = cd_equals_c[sorted_ind] # for precision
    #print(p)
    #print(sum(p))
    sum_rec_l = np.cumsum(tp_recall)
    sum_prec_l = np.cumsum(tp_precision)

    tp = np.cumsum(p)
    rec_cl = np.nan_to_num(np.divide(tp, sum_rec_l))

    prec_cl = np.nan_to_num(tp/np.maximum(sum_prec_l, np.finfo(np.float64).eps))
    
    #print(rec_cl)
    #print(prec_cl)
    #print(voc_ap(rec_cl, prec_cl))
    #print(prec_l)
    #print('------------------------')
    
    #final_results['tp_to_return_'+cls] = p
    #final_results['rec_cl_'+cls] = rec_cl
    #final_results['prec_cl_'+cls] = prec_cl
    #final_results = final_results.loc[:, ~final_results.columns.str.contains('^Unnamed')]
    #final_results.to_csv(final_output_df_csv_filename, index=False)

    return p, rec_cl, prec_cl
    #return rec_cl, prec_cl


def voc_eval_s_given_cl_mAP(final_output_df_csv_filename, p_gt, p_seg, p, cls, ovthersh=SEG_IOU_THRESHOLD):
    #print('-------voc_eval_s_given_cl_mAP---------')
    #print("in seg")
    # for each good box from before
    # read gt, seg_results (gt is from seg class) 
    # crop both to this good box
    # compute IoU of this class, record those of > 0.5 (mask_util._iou)

    # on top of box IoU being a p and cd = c*, if seg IoU > 0.5, count
    final_results = pd.read_csv(final_output_df_csv_filename) 
    nd = len(final_results)
    tp = np.zeros(nd)

    for d in range(nd):
        image_id = final_results['image_id'].iloc[d]
 
        if p[d] > 0: # good detcetion box
            #print(d)

            mask_file = final_results['masks'].iloc[d]
            #print(mask_file)
            try:
                if isinstance(mask_file, str):
                    mask = Image.open(io.BytesIO(base64.b64decode(mask_file)))

                        #print(mask.shape)
                    mask = np.asarray(mask)[int(final_results['ymin'].iloc[d]-1):int(final_results['ymax'].iloc[d]), int(final_results['xmin'].iloc[d]-1):int(final_results['xmax'].iloc[d])] 
                    

                    #exit()
            #img = np.asarray(Image.open(mask_file))[int(final_results['ymin'].iloc[d]-1):int(final_results['ymax'].iloc[d]), int(final_results['xmin'].iloc[d]-1):int(final_results['xmax'].iloc[d])]# cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            #mask = (img != 0)

                    gt_mask_file = p_seg[d].split(' ')#'/w/10/users/boyue/VOCdevkit/VOC2012/SegmentationObject/'+image_id+ '.png' #TODO: make generic
                    gt_classes = p_gt[d].split(' ')

                    gt_mask_file = gt_mask_file[gt_classes.index(cls)] 
               

                    if isinstance(gt_mask_file, str): 
                    

                        gt_mask = Image.open(io.BytesIO(base64.b64decode(gt_mask_file)))

    
                        gt_mask = np.asarray(gt_mask)[int(final_results['ymin'].iloc[d]-1):int(final_results['ymax'].iloc[d]), int(final_results['xmin'].iloc[d]-1):int(final_results['xmax'].iloc[d])] 
                    
                #print(gt_mask == mask)
                

                #gt_mask = np.asarray(Image.open(gt_mask_file))[int(final_results['ymin'].iloc[d]-1):int(final_results['ymax'].iloc[d]), int(final_results['xmin'].iloc[d]-1):int(final_results['xmax'].iloc[d])]
                #gt_mask = (gt_mask == all_class_names.index(cls)+1)
                #im = Image.fromarray(gt_mask)
                #im.save('/w/10/users/boyue/bootstrap-frost/my_voc_orig/gt/' + image_id + str(d) + '.png')
                #print('-------')
                #print(mask.shape)
                #print(gt_mask.shape)
                    
                        jaccard = JaccardIndex(num_classes=2)
                        iou = jaccard(torch.tensor(mask), torch.tensor(gt_mask))
                        #print(iou)
                        #print('-------')
                        #exit()
                        if iou.item() > ovthersh:
                            tp[d] = 1.0
            except:
                pass

    #sprint(sum(tp))
    good_detection = np.nan_to_num(p)
    #print(sum(good_detection))
    good_detection = np.cumsum(good_detection)
    prec_scl = np.nan_to_num(np.divide(np.cumsum(tp), good_detection))
    #print(np.isnan(prec_scl).any())
    rec_scl = np.nan_to_num(np.cumsum(tp)/np.maximum(good_detection, np.finfo(np.float64).eps))
    #print(np.isnan(rec_scl).any())
    #print(rec_scl)
    #print(prec_scl)
    #print(voc_ap(rec_scl, prec_scl))
    #print(prec_l)
    #print('------------------------')

    #final_results['good_seg_'+cls] = tp
    #final_results['rec_scl_'+cls] = rec_scl
    #final_results['prec_scl_'+cls] = prec_scl
    #final_results = final_results.loc[:, ~final_results.columns.str.contains('^Unnamed')]
    #final_results.to_csv(final_output_df_csv_filename, index=False)
    return rec_scl, prec_scl

def voc_eval_d_mAP(rec_l, prec_l, rec_cl, prec_cl):
    #final_results = pd.read_csv(final_output_df_csv_filename) 
    #rec_l = np.nan_to_num(final_results['rec_l_' +cls].to_numpy())
    #rec_cl = np.nan_to_num(final_results['rec_cl_' +cls].to_numpy())
    #prec_l = np.nan_to_num(final_results['prec_l_' +cls].to_numpy())
    #prec_cl = np.nan_to_num(final_results['prec_cl_' +cls].to_numpy())

    rec = np.multiply(rec_l, rec_cl)
    prec = np.multiply(prec_l, prec_cl)

    #final_results['rec_d_'+cls] = rec
    #final_results['prec_d_'+cls] = prec
    #final_results = final_results.loc[:, ~final_results.columns.str.contains('^Unnamed')]
    #final_results.to_csv(final_output_df_csv_filename, index=False)

    ap = voc_ap(rec, prec)
    return ap

def voc_eval_s_mAP(rec_l, prec_l, rec_cl, prec_cl, rec_scl, prec_scl):
    #print('--------------voc_eval_s_mAP------------')
    #final_results = pd.read_csv(final_output_df_csv_filename) 
    #rec_l = np.nan_to_num(final_results['rec_l_' +cls].to_numpy())
    #rec_cl = np.nan_to_num(final_results['rec_cl_' +cls].to_numpy())
    #rec_scl = np.nan_to_num(final_results['rec_scl_' +cls].to_numpy())

    #prec_l = np.nan_to_num(final_results['prec_l_' +cls].to_numpy())
    #prec_cl = np.nan_to_num(final_results['prec_cl_' +cls].to_numpy())
    #prec_scl = np.nan_to_num(final_results['prec_scl_' +cls].to_numpy())

    rec = np.multiply(np.multiply(rec_l, rec_cl), rec_scl)
    prec = np.multiply(np.multiply(prec_l, prec_cl), prec_scl)

    #final_results['rec_s_'+cls] = rec
    #final_results['prec_s_'+cls] = prec
    #final_results = final_results.loc[:, ~final_results.columns.str.contains('^Unnamed')]
    #final_results.to_csv(final_output_df_csv_filename, index=False)

    #print(sum(rec))
    #print(sum(prec))

    ap = voc_ap(rec, prec)
    #print(rec)
    #print(prec)
    #print(ap)
    #print('--------------------------')
    return ap

def run_eval(args, new_dataset_name, final_output_df_csv_filename, seg_path=None):
    cfg = setup(args)
    #print(cfg)
    
    # build model and load model weights
    
    model = Trainer.build_model(cfg)
    #model = DefaultPredictor(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    # build dataloader and evaluator
    dataloader = DefaultTrainer.build_test_loader(cfg, new_dataset_name)
    all_size = len(dataloader.dataset)
    # build two csv files for storing results
    #box_proposal_csv_filename = 'results_csv/bootstrap-'+args.transformation +"/box_proposal_records" + "_iter_"+ str(i+1) + ".csv"
    
    
    #box_proposal_fields = {"image_id": [], "logit_score": [], "xmin": [], "ymin": [], "xmax": [], "ymax": []}
    final_output_fields = {"image_id": [], "conf_score": [], "class": [], "xmin": [], "ymin": [], "xmax": [], "ymax": [], "masks": []}
    
    #box_proposal_df = pd.DataFrame(box_proposal_fields)
    #box_proposal_df.to_csv(box_proposal_csv_filename)
    final_output_df = pd.DataFrame(final_output_fields)
    final_output_df.to_csv(final_output_df_csv_filename)

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        for idx, inputs in enumerate(dataloader):
            #print(str(idx) + '/' + str(all_size))
            image_id = inputs[0]['image_id']
            seg_gt = MetadataCatalog.get(new_dataset_name).dirname + '/SegmentationClass/' + image_id+".png"
            #print(seg_gt, os.path.exists(seg_gt))
            #exit()
            if not os.path.exists(seg_gt):
                #print('skipping since no seg gt')
                continue
            #inputs = cv2.imread(MetadataCatalog.get(new_dataset_name).dirname + '/JPEGImages/' + image_id+".jpg")
            #print(inputs)
            outputs = model(inputs)
            #print(outputs)
            #box_proposals = model.proposals
            #print(final_output_df_csv_filename)
            #exit()
            if seg_path:
                process(inputs, outputs, final_output_df_csv_filename, seg_results_path=seg_path)
            else:
                process(inputs, outputs, final_output_df_csv_filename)
            #if idx > 6:
            #    break
            #print('finished processing ' + str(idx) )
    
    
    
    #continue
    # compare with directly computing mAP
    #evaluator = build_evaluator(cfg, new_dataset_name)
    # run inference
    #result = inference_on_dataset(model, dataloader, evaluator)

    #results[i] = result
    #print(result)
    #exit()

def run_metrics(new_dataset_name, voc_root, final_output_df_csv_filename, pp=False, orig_results=None, seg=False):
    # correctness-preservation
    # do computation of metrics
    #print(final_output_df_csv_filename)
    class_names = MetadataCatalog.get(new_dataset_name).thing_classes
    annopath = os.path.join(voc_root, "Annotations", "{}.xml")
    imageset_file = os.path.join(voc_root, "ImageSets/Main", MetadataCatalog.get(new_dataset_name).split+".txt")
    if pp:
        all_recs, npos = load_orig_det_as_gt(new_dataset_name, orig_results)
        #exit()
    else:
        all_recs, npos= load_gt(imageset_file, annopath, class_names)
    voc_eval_l_process(final_output_df_csv_filename, all_recs,  ovthresh=BOX_IOU_THRESHOLD)

    APs = []
    for i in range(len(class_names)):
        cls = class_names[i]
        
        #if cls not in ['aeroplane', 'person']:
        #    continue
        #print(cls)
        # l_AP
        voc_eval_l_mAP(final_output_df_csv_filename, cls, i, npos[cls])
        
        # c_given_l_AP
        voc_eval_c_given_l_mAP(final_output_df_csv_filename, cls, i)
        
        if not seg:
            ap = voc_eval_d_mAP(final_output_df_csv_filename, cls)    
        else:
            voc_eval_s_given_cl_mAP(final_output_df_csv_filename, cls, i, ovthersh=SEG_IOU_THRESHOLD)
            #print(final_results['l_IoU_p'])
            #voc_eval_s_given_cl_mAP()
            ap = voc_eval_s_mAP(final_output_df_csv_filename, cls)    

        APs.append(ap)
        #print("AP: ")
        #print(APs[i])
        
    # mAP of d
    #print(np.average(np.nan_to_num(APs)))
    #print('Our calculation: ' + str(mAP))
    #OrderedDict([('bbox', {'AP': 21.65502060533366, 'AP50': 40.485708435128295, 'AP75': 21.105694335721058})])
    #exit()
    return APs

def test_sat(transformed_csv, orig_csv, dataset_name=''):
    class_names = MetadataCatalog.get(new_dataset_name).thing_classes

    for cls in class_names:
        transformed_results = pd.read_csv(transformed_csv)
        transformed_prec_l = transformed_results['prec_l_'+cls]
        transformed_rec_l = transformed_results['rec_l_'+cls]

        orig_results = pd.read_csv(orig_csv)
        orig_prec_l = orig_results['prec_l_'+cls]
        orig_rec_l = orig_results['rec_l_'+cls]

        # plot

    return 1, 2, 3, 4


def bootstrap_results(csv_path, req_type, classes):
    #print(csv_path)
    all_csv_files = os.listdir(csv_path) 
    num_iter = int(len([x for x in all_csv_files if 'iter' in x])/4)
    #print(num_iter)
    from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    
    for cls in classes:
        if cls not in ['person']:
            continue
        for type in ['l_', 'cl_', 'scl_', '']:
            all_iter_results = []
            for i in range(num_iter):
                # correctness preservation
                if req_type == 'cp':
                    orig_file = csv_path + "cp_orig" + "_iter_"+ str(i+1) + ".csv"
                    orig = pd.read_csv(orig_file)
                    transf_file = csv_path +"cp_transf" + "_iter_"+ str(i+1) + ".csv"
                    transf = pd.read_csv(transf_file)
                # prediction preservation
                if req_type == 'pp':
                    orig_file = csv_path + "pp_sigma_" + str(SIGMA) +  "_iter_"+ str(i+1) + ".csv"
                    orig = pd.read_csv(orig_file)
                    transf_file = csv_path + "pp_all_transf" + "_iter_"+ str(i+1) + ".csv"
                    transf = pd.read_csv(transf_file)
                
                prec_orig = orig['prec_' + type +cls].tolist()
                rec_orig = orig['rec_'+type +cls].tolist()

                prec_transf = transf['prec_' + type+cls].tolist()
                rec_transf = transf['rec_' + type+cls].tolist()

                max_x_value = max(max(rec_orig), max(rec_transf))
                
                x = []
                y = []
                for j in range(len(prec_orig)):
                    if len(x) > 0 and len(y) > 0 and x[-1] == rec_orig[j] and y[-1] >= prec_orig[j]:
                        continue
                    else:
                        x.append(rec_orig[j])
                        y.append(prec_orig[j])
                #print(l_prec_orig)
                #print(l_rec_orig)
                ius = InterpolatedUnivariateSpline(x, y)
                xi = np.linspace(0, max_x_value, 100)
                yi = ius(xi)
                '''
                
                plt.subplot(2, 1, 1)
                plt.plot(x, y, 'bo')
                plt.plot(xi, yi, 'g')
                plt.title('Interpolation using univariate spline')
                plt.show()
                '''

                
                x_t = []
                y_t = []
                for j in range(len(prec_transf)):
                    if len(x_t) > 0 and len(y_t) > 0 and x_t[-1] == rec_transf[j] and y_t[-1] >= prec_transf[j]:
                        continue
                    else:
                        x_t.append(rec_transf[j])
                        y_t.append(prec_transf[j])
                #print(x_t)
                #print(y_t)
                ius_t = InterpolatedUnivariateSpline(x_t, y_t)
                xi_t = np.linspace(0, max_x_value, 100)
                yi_t = ius_t(xi_t)
                
                '''
                plt.subplot(2, 1, 1)
                plt.plot(x_t, y_t, 'bo')
                plt.plot(xi_t, yi_t, 'g')
                plt.title('Interpolation using univariate spline')
                plt.show()
                '''
                difference = sum(yi-yi_t)
                all_iter_results.append(difference)
            #print(type, cls)
            result = stats.ttest_1samp(all_iter_results, 0, alternative='less')
            #print(all_iter_results)
            #print(result)

    return 0

def estimate_curve_diff(rec_o, prec_o, rec_t, prec_t, num_p, npos):
    #print('HERE')
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
    #print(mrec_t)
    #print(mpre_t)
    #print(mrec_o)
    #print(mpre_o)
    

    #exit()
    '''
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'bo')
    plt.plot(xi, yi, 'g')
    plt.title('Interpolation using univariate spline')
    plt.show()
    '''

    '''
    plt.subplot(2, 1, 1)
    plt.plot(x_t, y_t, 'bo')
    plt.plot(xi_t, yi_t, 'g')
    plt.title('Interpolation using univariate spline')
    plt.show()
    '''
    

    
    #difference = sum(yi-yi_t)
    #all_iter_results.append(difference)
    #print(type, cls)
    #result = stats.ttest_1samp(all_iter_results, 0, alternative='less')
    #print(all_iter_results)
    #print(result)


    if len(mrec_o) > 3 and len(mrec_t) > 3:
        #print('-----o------')

        tck_o, u  = interpolate.splprep([mrec_o, mpre_o], s=0)
        #ius = InterpolatedUnivariateSpline(x, y)
        xi = np.sort(np.asarray(list(set(list(mrec_o) + list(mrec_t)))))
        yi = interpolate.splev(xi, tck_o, der=0) #line_fit(xi)

        #print('-----t-----')
        #print(mrec_t, mpre_t)
        #print('-----------')
        tck_t, u  = interpolate.splprep([mrec_t, mpre_t],s=0)
        #ius = InterpolatedUnivariateSpline(x, y)
        #xi_t = np.linspace(0, 1, NUM_samples)
        yi_t = interpolate.splev(xi, tck_t, der=0)

        difference = np.asarray(yi_t) - np.asarray(yi)
        mu, sigma = scipy.stats.norm.fit(difference)
        distribution = scipy.stats.norm(mu, sigma)
        result = distribution.cdf(0)
        #if math.isnan(result):
        #    exit()
        #print(result)
        return np.nan_to_num(result), len(xi)
    else:
        #print('Nah')
        return None, None
        print('Nah') # this should only happen if one of them is almost a constant (I only saw 1 and 0)
        if len(mpre_o) <= 3:
            average_o = sum(mpre_o)/len(mpre_o)
            above_ao = [x for x in mpre_t if x < average_o]
            return len(above_ao)/float(len(mpre_t)), len(list(mrec_t))
        elif len(mrec_t) <= 3:
            average_t = sum(mpre_t)/len(mpre_t)
            above_at = [x for x in mpre_o if x < average_t]
        return len(above_at)/float(len(mpre_o)), len(list(mrec_t))

    

def calculate_confidence(acc_list, req_acc):
    # fitting a normal distribution
    _, bins, _ = plt.hist(acc_list, 20, alpha=0.5, density=True)
    mu, sigma = scipy.stats.norm.fit(acc_list)
    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    print("Estimated mean from bootstrapping: " + str(mu))
    print("Estimated sigma from bootstrapping: " + str(sigma))
    # exit()
    distribution = scipy.stats.norm(mu, sigma)
    result = 1 - distribution.cdf(req_acc)
    print('confidence of satisfication:' + str(result))
    if result >= 0.5:
        print("requirement SATISFIED")
    else:
        print("requirement NOT SATISFIED")
    return result, mu, sigma, result >= 0.5

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--transformation", required=True)#, choices=TRANSFORMATIONS)
    args = parser.parse_args()
    args.dist_url = 'tcp://127.0.0.1:50152'
    args.eval_only = True
    args.machine_rank = 0
    args.num_gpus = 1
    #voc_orig_root = str(pathlib2.Path.home() / 'datasets' / 'PASCAL-VOC' / 'VOCtrainval_06-Nov-2007' / 'VOCdevkit' / 'VOC2007')
    args.resume = False
    num_machines = 1
    #bootstrap_data_dir = pathlib2.Path(__file__).absolute().parent / 'data' / f'bootstrap-{args.transformation}'
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
    
    list_weights = ['/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/R50-C4_1x_MAXiter50000/model_final.pth',
    '/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/R50-DC5_1x_MAXiter50000/model_final.pth',
    '/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/R50-FPN_1x_MAXiter50000/model_final.pth',
    '/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/R50-C4_3x_MAXiter50000/model_final.pth',
    '/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/R50-DC5_3x_MAXiter50000/model_final.pth',
    '/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/R50-FPN_3x_MAXiter50000/model_final.pth',
    '/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/R101-C4_3x_MAXiter50000/model_final.pth',
    '/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/R101-DC5_3x_MAXiter50000/model_final.pth',
    '/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/R101-FPN_3x_MAXiter50000/model_final.pth',
    '/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/X101-FPN_3x_MAXiter50000/model_final.pth']

    #list_config = ["detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"]
    #list_weights = ['/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/R50-FPN_3x_MAXiter20000/model_final.pth', 'detectron2://PascalVOC-Detection/faster_rcnn_R_50_C4/142202221/model_final_b1acc2.pkl']

    #list_weights = ['/w/10/share/Instance-Segmentation-on-Tiny-PASCAL-VOC-Dataset/log/R50-FPN_3x_MAXiter20000/model_final.pth', 'detectron2://PascalVOC-Detection/faster_rcnn_R_50_C4/142202221/model_final_b1acc2.pkl']

    assert len(list_config) == len(list_weights)
    req_checking_results = {}
    # run all original images
    #new_dataset_name = f"my_voc_orig"
    #voc_root = str(pathlib2.Path('/w/10/users/boyue/VOCdevkit/VOC2012/')) # where all original is stored
    #
    #update_yaml_dataset_name(args.config_file, new_dataset_name)
    #register_pascal_voc(new_dataset_name, dirname=voc_root, split="tiny", year=2007)

    
    all_final = {}
    cp_l_results = []
    cp_cl_results = []
    cp_d_results = []

    pp_l_results = []
    pp_cl_results = []
    pp_d_results = []

    for model_index in range(len(list_config)):
        #if model_index != 0:
        #    continue
        if model_index not in all_final:
            all_final[model_index] = {}     
        
        tracemalloc.start()
        #import time
        #tracker = EmissionsTracker()
        #tracker.start()
        start_time = time.time()
        args.config_file = list_config[model_index]
        args.opts = ['MODEL.WEIGHTS',
                list_weights[model_index], 'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5, 'MODEL.ROI_HEADS.NUM_CLASSES', 20, 'SOLVER.BASE_LR', 0.00025]

        req_checking_results[model_index] = {}
        bootstrap_data_dir = pathlib2.Path('/w/10/users/boyue/') / 'data' / f'bootstrap-{args.transformation}'
        #results = {}
        if not os.path.exists('/w/10/users/boyue/results_csv'):
            os.mkdir('/w/10/users/boyue/results_csv')
        if not os.path.exists('/w/10/users/boyue/results_csv/bootstrap-'+args.transformation):
            os.mkdir('/w/10/users/boyue/results_csv/bootstrap-'+args.transformation)
        if not os.path.exists('/w/10/users/boyue/results_csv/bootstrap-'+args.transformation + '/model_' + str(model_index)):
            os.mkdir('/w/10/users/boyue/results_csv/bootstrap-'+args.transformation + '/model_' + str(model_index))
        csv_path = '/w/10/users/boyue/results_csv/bootstrap-'+args.transformation + '/model_' + str(model_index) + '/'

        # run all original images
        new_dataset_name = f"my_voc_orig"+ '_' + str(model_index)
        voc_root = str(pathlib2.Path('/w/10/users/boyue/VOCdevkit/VOC2012/')) # where all original is stored
        #voc_root = str(pathlib2.Path('/w/10/share/ImageCorruption/corrupted_images/frost_c/')) # where all original is stored
        
        seg_results_orig = "/w/10/users/boyue/bootstrap-" + args.transformation + '/model_' + str(model_index) +'/all_orig/'
        Path(seg_results_orig).mkdir(parents=True, exist_ok=True)
        #
        update_yaml_dataset_name(args.config_file, new_dataset_name)
        register_pascal_voc(new_dataset_name, dirname=voc_root, split="seg", year=2007)
        orig_final_output_df_csv_filename = csv_path+ '/all_orig.csv'        
        #run_eval(args, new_dataset_name, orig_final_output_df_csv_filename, seg_path=seg_results_orig)
        #run_metrics(new_dataset_name, voc_root, orig_final_output_df_csv_filename, seg=True)
        
        cp_transf_all = []
        cp_orig_all = []
        for i, root in enumerate(bootstrap_data_dir.iterdir()):
            if i not in all_final[model_index]:
                all_final[model_index][i] = {'cp_l_results' : [], 'cp_cl_results' : [], 'cp_scl_results' : [], 'cp_d_results' : [], 'cp_s_results' : [], 'cp_d_mAP' : [], 'cp_s_mAP' : [],
                                            'pp_l_results' : [], 'pp_cl_results' : [], 'pp_scl_results' : [], 'pp_d_results' : [], 'pp_s_results' : [], 'pp_d_mAP' : [], 'pp_s_mAP' : []}
            #voc_root = str(pathlib2.Path(__file__).absolute().parent / 'data' / f'bootstrap-{args.transformation}' / f"iter{i + 1}")
            voc_root = str(pathlib2.Path('/w/10/users/boyue/') / 'data' / f'bootstrap-{args.transformation}' / f"iter{i + 1}") # where the iteration of bootstrap is
            bootstrap_df = pd.read_csv("bootstrap_dfs/bootstrap_df-"+args.transformation+ ".csv")
            this_iteration = bootstrap_df.loc[bootstrap_df['iteration_id'] == i+1]
            csv_path = '/w/10/users/boyue/results_csv/bootstrap-'+args.transformation + '/model_' + str(model_index) + '/'
        
            args.config_file = list_config[model_index]
            args.opts = ['MODEL.WEIGHTS',
                list_weights[model_index], 'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5, 'MODEL.ROI_HEADS.NUM_CLASSES', 20, 'SOLVER.BASE_LR', 0.00025]
            # FOR correctness_preservation
            # run bootstrapped transformed images with real ground truth    
            cp_transf_csv_filename = csv_path +"cp_transf" + "_iter_"+ str(i+1) + ".csv"
            seg_results_path = "/w/10/users/boyue/bootstrap-" + args.transformation + '/model_' + str(model_index) + "_iter_"+ str(i+1)
            if not os.path.exists("/w/10/users/boyue/bootstrap-" + args.transformation + '/model_' + str(model_index) + "_iter_"+ str(i+1)):
                os.mkdir("/w/10/users/boyue/bootstrap-" + args.transformation + '/model_' + str(model_index) + "_iter_"+ str(i+1))
            
            # register custom dataset
            new_dataset_name = f"my_voc_{i}" + '_' +str(model_index)
            update_yaml_dataset_name(args.config_file, new_dataset_name)
            register_pascal_voc(new_dataset_name, dirname=voc_root, split="val", year=2007)
            #run_eval(args, new_dataset_name, cp_transf_csv_filename, seg_path=seg_results_path)
            #continue
            # getting all images sampled in this iteration and load their ground truth
            # process them
            
            
            all_orig = pd.read_csv(orig_final_output_df_csv_filename)
            results_this_iteration = all_orig.loc[all_orig['image_id'].isin(list(this_iteration['image_id']))]

            annopath = os.path.join(voc_root, "Annotations", "{}.xml")
            gt, npos_gt = load_gt(this_iteration, annopath, all_class_names)
            cp_orig_csv_file = csv_path + "cp_orig" + "_iter_"+ str(i+1) + ".csv"
            results_this_iteration.to_csv(cp_orig_csv_file, index=False)

            #print('l_process')
            o_pl, o_p_gt, o_p_seg = voc_eval_l_process(cp_orig_csv_file, gt)
            #print(sum(o_pl))
            t_pl, t_p_gt, t_p_seg = voc_eval_l_process(cp_transf_csv_filename, gt)
            #print(sum(t_pl))
            #exit()
            
            
            # for prediction-preservation

            # eval transformed with original as gt
            orig_as_gt, npos_orig_as_gt = load_orig_det_as_gt(new_dataset_name, results_this_iteration)
            st_pl, st_p_gt, st_p_seg = voc_eval_l_process(cp_transf_csv_filename, orig_as_gt)
            
            # eval less than sigma as orig

            # find less than sigma, to update original to gt, need to change names to ids
            less_than_sigma = this_iteration.loc[this_iteration['vd_score'] < SIGMA] # all images < sigma
            # find transformed < sigma
            all_transf_results = pd.read_csv(cp_transf_csv_filename)
            results_this_iteration_l_sigma = all_transf_results.loc[all_transf_results['image_id'].isin(list(less_than_sigma['image_id']))]
            less_than_sigma_csv_file = csv_path + "pp_sigma_" + str(SIGMA) +  "_iter_"+ str(i+1) + ".csv"
            results_this_iteration_l_sigma.to_csv(less_than_sigma_csv_file, index=False)
            # find original < sigma
            orig_results_this_iteration_l_sigma = all_orig.loc[all_orig['image_id'].isin(list(less_than_sigma['image_id']))]
            orig_less_than_sigma_csv_file = csv_path + "pp_sigma_orig_" + str(SIGMA) +  "_iter_"+ str(i+1) + ".csv"
            orig_results_this_iteration_l_sigma.to_csv(orig_less_than_sigma_csv_file, index=False)
            
            sigma_as_gt, npos_sigma_as_gt = load_orig_det_as_gt(new_dataset_name, orig_results_this_iteration_l_sigma)
            s_pl, s_p_gt, s_p_seg = voc_eval_l_process(less_than_sigma_csv_file, sigma_as_gt)
            
            

            for cls in ['person', 'bus']:#, 'bird', 'sheep', 'train', 'bus', 'cat', 'dog']:#['person', 'car', 'bird', 'sheep', 'train', 'bus', 'cat', 'dog']:
                
                # correctness-preservation
                # calculate PR of l 
                
                #print(cls)
                #print('l_map')
                o_tp_recall, o_tp_precision, o_rec_l, o_prec_l, cd_equals_c_o = voc_eval_l_mAP(cp_orig_csv_file, o_pl, o_p_gt, cls, npos_gt[cls])
                #print(sum(o_tp_recall), sum(o_tp_precision))
                t_tp_recall, t_tp_precision, t_rec_l, t_prec_l, cd_equals_c_t = voc_eval_l_mAP(cp_transf_csv_filename, t_pl, t_p_gt, cls, npos_gt[cls])
                #print(sum(t_tp_recall), sum(t_tp_precision))
                #print(voc_ap(o_rec_l, o_prec_l), voc_ap(t_rec_l, t_prec_l))
                
                try:
                    conf_ninety_five, num_points = estimate_curve_diff(o_rec_l, o_prec_l, t_rec_l, t_prec_l, cd_equals_c_t, npos_gt[cls])
                    #()
                    #print(t_rec_l, t_prec_l)
                    #ap_o = np.nan_to_num(voc_ap(o_rec_l, o_prec_l))
                    #ap_t = np.nan_to_num(voc_ap(t_rec_l, t_prec_l))
                    #print(ap_t)
                    #if cls in npos_gt and npos_gt[cls] > 0:
                    #    all_final[model_index][i]['cp_l_results'].append((cls, ap_t, ap_o))
                    #    with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                    #            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    if conf_ninety_five:
                    
                    #print(mu_ninety_five)
                    #if cls in npos_gt and npos_gt[cls] > 0:
                    #continue
                        all_final[model_index][i]['cp_l_results'].append((cls, conf_ninety_five, num_points))
                        with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                                pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    pass
                
                # calculate PR of c|l 
                #print(cls)
                #print('c_given_l')
                

                o_p, o_rec_cl, o_prec_cl = voc_eval_c_given_l_mAP(cp_orig_csv_file, o_p_gt, o_tp_recall, o_tp_precision, cls)
                #print(sum(o_p))
                t_p, t_rec_cl, t_prec_cl = voc_eval_c_given_l_mAP(cp_transf_csv_filename, t_p_gt, t_tp_recall, t_tp_precision, cls)
                #print(sum(t_p))
                #print(t_rec_cl, t_prec_cl)
                #print(voc_ap(o_rec_cl, o_prec_cl), voc_ap(t_rec_cl, t_prec_cl))
                
                try:
                    conf_ninety_five, num_points = estimate_curve_diff(o_rec_cl, o_prec_cl, t_rec_cl, t_prec_cl, t_tp_precision, npos_gt[cls])
                    if conf_ninety_five:
                        #print(mu_ninety_five)
                    #if cls in npos_gt and npos_gt[cls] > 0:
                        all_final[model_index][i]['cp_cl_results'].append((cls, conf_ninety_five, num_points))
                        with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    pass
                
                
                #ap_o = np.nan_to_num(voc_ap(o_rec_cl, o_prec_cl))
                #ap_t = np.nan_to_num(voc_ap(t_rec_cl, t_prec_cl))
                #print(ap_t)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #    all_final[model_index][i]['cp_cl_results'].append((cls, ap_t, ap_o))
                #    with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                #            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                
                #print('d')
                o_rec_d = np.nan_to_num(np.multiply(o_rec_l, o_rec_cl))
                o_prec_d = np.nan_to_num(np.multiply(o_prec_l, o_prec_cl))
                t_rec_d = np.nan_to_num(np.multiply(t_rec_l, t_rec_cl))
                t_prec_d = np.nan_to_num(np.multiply(t_prec_l, t_prec_cl))
                #print(voc_ap(o_rec_d, o_prec_d), voc_ap(t_rec_d, t_prec_d))
                
                try:
                    conf_ninety_five, num_points = estimate_curve_diff(o_rec_d, o_prec_d, t_rec_d, t_prec_d, cd_equals_c_t, npos_gt[cls])
                    if conf_ninety_five:
                    #print(mu_ninety_five)
                    #if cls in npos_gt and npos_gt[cls] > 0:
                        all_final[model_index][i]['cp_d_results'].append((cls, conf_ninety_five, num_points))
                        with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    mAP_o = voc_ap(o_rec_d, o_prec_d)
                    mAP_t = voc_ap(t_rec_d, t_prec_d)
                    all_final[model_index][i]['cp_d_mAP'].append((cls, mAP_o, mAP_t))
                    with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                        pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    pass
                # calculate mAP of d 
                #ap_o = np.nan_to_num(voc_eval_d_mAP(o_rec_l, o_prec_l, o_rec_cl, o_prec_cl))
                #ap_t = np.nan_to_num(voc_eval_d_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl))
                #print(ap_t)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #    all_final[model_index][i]['cp_d_results'].append((cls, ap_t, ap_o))
                #    with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                #            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                # calculate PR of s|c,l 
                #print('s')
                o_rec_scl, o_prec_scl = voc_eval_s_given_cl_mAP(cp_orig_csv_file, o_p_gt, o_p_seg, o_p, cls)
                t_rec_scl, t_prec_scl = voc_eval_s_given_cl_mAP(cp_transf_csv_filename, t_p_gt, t_p_seg, t_p, cls)
                #print(t_rec_cl, t_prec_cl)
                #print(voc_ap(o_rec_scl, o_prec_scl), voc_ap(t_rec_scl, t_prec_scl))
                
                try:
                    conf_ninety_five, num_points = estimate_curve_diff(o_rec_scl, o_prec_scl, t_rec_scl, t_prec_scl, t_p, npos_gt[cls])
                    if conf_ninety_five:
                        #print(mu_ninety_five)
                        #if cls in npos_gt and npos_gt[cls] > 0:
                        all_final[model_index][i]['cp_scl_results'].append((cls, conf_ninety_five, num_points))
                        with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    pass
                

                #ap_o = np.nan_to_num(voc_ap(o_rec_scl, o_prec_scl))
                #ap_t = np.nan_to_num(voc_ap(t_rec_scl, t_prec_scl))
                #print(ap_t)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #    all_final[model_index][i]['cp_scl_results'].append((cls, ap_t, ap_o))
                #    with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                #            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # calculate mAP of s
                #ap_o = np.nan_to_num(voc_eval_s_mAP(o_rec_l, o_prec_l, o_rec_cl, o_prec_cl, o_rec_scl, o_prec_scl))
                #ap_t = np.nan_to_num(voc_eval_s_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl, t_rec_scl, t_prec_scl))
                #print(ap_t)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #    all_final[model_index][i]['cp_s_results'].append((cls, ap_t, ap_o))
                #    with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                #            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                o_rec_s = np.nan_to_num(np.multiply(np.multiply(o_rec_l, o_rec_cl), o_rec_scl))
                o_prec_s = np.nan_to_num(np.multiply(np.multiply(o_prec_l, o_prec_cl), o_prec_scl))
                t_rec_s = np.nan_to_num(np.multiply(np.multiply(t_rec_l, t_rec_cl), t_rec_scl))
                t_prec_s = np.nan_to_num(np.multiply(np.multiply(t_prec_l, t_prec_cl), t_prec_scl))
                
                try:
                    conf_ninety_five, num_points = estimate_curve_diff(o_rec_s, o_prec_s, t_rec_s, t_prec_s, cd_equals_c_t, npos_gt[cls])
                    if conf_ninety_five:
                    #print(mu_ninety_five)
                    #if cls in npos_gt and npos_gt[cls] > 0:
                        all_final[model_index][i]['cp_s_results'].append((cls, conf_ninety_five, num_points))
                        with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    pass

                
                mAP_o = voc_ap(o_rec_s, o_rec_s)
                mAP_t = voc_ap(t_rec_s, t_prec_s)

                all_final[model_index][i]['cp_s_mAP'].append((cls, mAP_o, mAP_t))
                with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                    pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

                
                
                # prediction-preservation
                
                # calculate PR of l 
                s_tp_recall, s_tp_precision, s_rec_l, s_prec_l, cd_equals_c_s = voc_eval_l_mAP(less_than_sigma_csv_file, s_pl, s_p_gt, cls, npos_sigma_as_gt[cls])
                #print(transformed_human_results, original_human_results)
                t_tp_recall, t_tp_precision, t_rec_l, t_prec_l, cd_equals_c_t = voc_eval_l_mAP(cp_transf_csv_filename, st_pl, st_p_gt, cls, npos_orig_as_gt[cls])
                #ap_o = np.nan_to_num(voc_ap(s_rec_l, s_prec_l))
                #ap_t = np.nan_to_num(voc_ap(t_rec_l, t_prec_l))
                #print(ap_t)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #    all_final[model_index][i]['pp_l_results'].append((cls, ap_t, ap_o))
                #    with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                #            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                try:
                    conf_ninety_five, num_points = estimate_curve_diff(s_rec_l, s_prec_l, t_rec_l, t_prec_l, cd_equals_c_t, npos_orig_as_gt[cls])
                    if conf_ninety_five:
                    #print(mu_ninety_five)
                    #if cls in npos_orig_as_gt and npos_orig_as_gt[cls] > 0:
                        all_final[model_index][i]['pp_l_results'].append((cls, conf_ninety_five, num_points))
                        with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    pass
                    

                # calculate PR of c|l 
                s_p, s_rec_cl, s_prec_cl = voc_eval_c_given_l_mAP(less_than_sigma_csv_file, s_p_gt, s_tp_recall, s_tp_precision, cls)
                t_p, t_rec_cl, t_prec_cl = voc_eval_c_given_l_mAP(cp_transf_csv_filename, st_p_gt, t_tp_recall, t_tp_precision, cls)
                
                #ap_o = np.nan_to_num(voc_ap(s_rec_cl, s_prec_cl))
                #ap_t = np.nan_to_num(voc_ap(t_rec_cl, t_prec_cl))
                #print(ap_t)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #    all_final[model_index][i]['pp_cl_results'].append((cls, ap_t, ap_o))
                #    with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                #            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                try:
                    conf_ninety_five, num_points = estimate_curve_diff(s_rec_cl, s_prec_cl, t_rec_cl, t_prec_cl, t_tp_precision, npos_orig_as_gt[cls])
                    
                    
                    if conf_ninety_five:
                    #print(mu_ninety_five)
                    #if cls in npos_orig_as_gt and npos_orig_as_gt[cls] > 0:
                        all_final[model_index][i]['pp_cl_results'].append((cls, conf_ninety_five, num_points))
                        with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    pass
                
                # calculate mAP of d 
                #ap_s = np.nan_to_num(voc_eval_d_mAP(s_rec_l, s_prec_l, s_rec_cl, s_prec_cl))
                #ap_t = np.nan_to_num(voc_eval_d_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl))
                #if cls in npos_orig_as_gt and npos_orig_as_gt[cls] > 0:
                #    all_final[model_index][i]['pp_d_results'].append((cls, ap_t, ap_o))
                #    with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                #            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

                s_rec_d = np.nan_to_num(np.multiply(s_rec_l, s_rec_cl))
                s_prec_d = np.nan_to_num(np.multiply(s_prec_l, s_prec_cl))
                t_rec_d = np.nan_to_num(np.multiply(t_rec_l, t_rec_cl))
                t_prec_d = np.nan_to_num(np.multiply(t_prec_l, t_prec_cl))
                try:
                    conf_ninety_five, num_points = estimate_curve_diff(s_rec_d, s_prec_d, t_rec_d, t_prec_d, cd_equals_c_t, npos_orig_as_gt[cls])
                    if conf_ninety_five:
                        #print(mu_ninety_five)
                        #if cls in npos_gt and npos_gt[cls] > 0:
                        all_final[model_index][i]['pp_d_results'].append((cls, conf_ninety_five, num_points))
                        with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    pass

                mAP_s = voc_ap(s_rec_d, s_prec_d)
                mAP_t = voc_ap(t_rec_d, t_prec_d)
                all_final[model_index][i]['pp_d_mAP'].append((cls, mAP_s, mAP_t))
                with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                    pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

                #exit()
                # calculate PR of s|c,l 
                s_rec_scl, s_prec_scl = voc_eval_s_given_cl_mAP(less_than_sigma_csv_file, s_p_gt, s_p_seg, s_p, cls)
                t_rec_scl, t_prec_scl = voc_eval_s_given_cl_mAP(cp_transf_csv_filename, st_p_gt, st_p_seg, t_p, cls)
                #print(t_rec_cl, t_prec_cl)
                #ap_o = np.nan_to_num(voc_ap(s_rec_scl, s_prec_scl))
                #ap_t = np.nan_to_num(voc_ap(t_rec_scl, t_prec_scl))
                #print(ap_t)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #    all_final[model_index][i]['pp_scl_results'].append((cls, ap_t, ap_o))
                #    with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                #            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                try:
                    conf_ninety_five, num_points = estimate_curve_diff(s_rec_scl, s_prec_scl, t_rec_scl, t_prec_scl, t_p, npos_orig_as_gt[cls])
                    if conf_ninety_five:
                    #print(mu_ninety_five)
                    #if cls in npos_gt and npos_gt[cls] > 0:
                        all_final[model_index][i]['pp_scl_results'].append((cls, conf_ninety_five, num_points))
                        with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    pass
                

                # calculate mAP of s
                #ap_s = np.nan_to_num(voc_eval_s_mAP(s_rec_l, s_prec_l, s_rec_cl, s_prec_cl, s_rec_scl, s_prec_scl))
                #ap_t = np.nan_to_num(voc_eval_s_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl, t_rec_scl, t_prec_scl))
                #print(ap_t)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #    all_final[model_index][i]['pp_s_results'].append((cls, ap_t, ap_s))
                #    with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                #        pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #print(all_final)
                
                s_rec_s = np.nan_to_num(np.multiply(np.multiply(s_rec_l, s_rec_cl), s_rec_scl))
                s_prec_s = np.nan_to_num(np.multiply(np.multiply(s_prec_l, s_prec_cl), s_prec_scl))
                t_rec_s = np.nan_to_num(np.multiply(np.multiply(t_rec_l, t_rec_cl), t_rec_scl))
                t_prec_s = np.nan_to_num(np.multiply(np.multiply(t_prec_l, t_prec_cl), t_prec_scl))
                
                try:
                    conf_ninety_five, num_points = estimate_curve_diff(s_rec_s, s_prec_s, t_rec_s, t_prec_s, cd_equals_c_t, npos_orig_as_gt[cls])
                    if conf_ninety_five:
                    #print(mu_ninety_five)
                    #if cls in npos_gt and npos_gt[cls] > 0:
                        all_final[model_index][i]['pp_s_results'].append((cls, conf_ninety_five, num_points))
                        with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                            pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    pass
                
                mAP_s = voc_ap(s_rec_s, s_prec_s)
                mAP_t = voc_ap(t_rec_s, t_prec_s)
                all_final[model_index][i]['pp_s_mAP'].append((cls, mAP_s, mAP_t))
                with open(args.transformation + '_bootstrap.pickle', 'wb') as handle:
                    pickle.dump(all_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    
    
    
    import itertools

    pickle_filename = args.transformation + '_bootstrap.pickle'
    with open(pickle_filename, 'rb') as handle:
        all_final = pickle.load(handle)
        print(all_final)
    for model_index in range(len(list_config)):
        #if model_index not in [1]:
        #    continue
        print('-----------------------------------------')
        print(model_index)
        model_results = all_final[model_index]

        for cls in ['person', 'bus']:#, 'bird', 'sheep', 'train', 'bus', 'cat', 'dog']:#['person', 'car', 'bird', 'sheep', 'train', 'bus', 'cat', 'dog']:
            #print(model_results)
            print('+++++++++++++++++++++++')
            print(cls)
            print('cp_l_results')
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['cp_l_results'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #print(sum(only_numbers)/len(only_numbers))
            #exit()
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
                #calculate_confidence(only_numbers, 0)
            
            print('cp_cl_results')
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['cp_cl_results'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #exit()
            #print(sum(only_numbers)/len(only_numbers))
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
                #calculate_confidence(only_numbers, 0)
            
            print('cp_scl_results')
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['cp_scl_results'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #exit()
            #)
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
                #calculate_confidence(only_numbers, 0)
            
            print('cp_d_results')
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['cp_d_results'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #exit()
            #if len(only_numbers) > 0:
            #print(sum(only_numbers)/len(only_numbers))
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
                #calculate_confidence(only_numbers, 0)

            print('cp_s_results')
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['cp_s_results'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #exit()
            
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
                #calculate_confidence(only_numbers, 0)

            print('cp_d_mAP')
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['cp_d_mAP'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1]-x[2]  for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #print(only_numbers)
            #exit()
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
                #calculate_confidence(only_numbers, 0)
            
            print('cp_s_mAP') 
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['cp_s_mAP'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1]-x[2] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #print(only_numbers)
            #exit()
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
                #calculate_confidence(only_numbers, 0)
            
            
            print('pp_l_results') 
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['pp_l_results'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #exit()
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
            #if len(only_numbers) > 3:
                #calculate_confidence(only_numbers, 0)
            
            print('pp_cl_results')
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['pp_cl_results'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
            #exit()
            #if len(only_numbers) > 3:
                #calculate_confidence(only_numbers, 0)
            
            print('pp_scl_results')  
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['pp_scl_results'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #print(only_numbers)
            #exit()
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
            #if len(only_numbers) > 3:
                
                #calculate_confidence(only_numbers, 0)
            
            print('pp_d_results')
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['pp_d_results'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #print(only_numbers)
            #exit()
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
            #if len(only_numbers) > 3:
                #calculate_confidence(only_numbers, 0)
            
            print('pp_s_results') 
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['pp_s_results'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #print(only_numbers)
            #exit()
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
            #if len(only_numbers) > 3:
                #calculate_confidence(only_numbers, 0)

            print('pp_d_mAP')
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['pp_d_mAP'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1]-x[2]  for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #print(only_numbers)
            #exit()
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
            #if len(only_numbers) > 3:
                #calculate_confidence(only_numbers, 0)
            
            print('pp_s_mAP') 
            all_results = list(itertools.chain.from_iterable([all_final[model_index][i]['pp_s_mAP'] for i in range(50)]))# ()
            #print(all_results)
            only_numbers = [x[1]-x[2] for x in all_results if x[0] == cls if x[1] is not None]
            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
            #print(only_numbers)
            #exit()
            if len(only_numbers) > 0:
                print(sum(only_numbers)/len(only_numbers))
            #if len(only_numbers) > 3:
                #calculate_confidence(only_numbers, 0)
            
            #print('time')
            #print(time.time()-start_time)
            #print('mem')
            #print(tracemalloc.get_traced_memory())
            #print('co2')
            #tracker.stop()
            #tracemalloc.stop()