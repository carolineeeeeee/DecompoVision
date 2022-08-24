from re import I
from tkinter import image_names
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.utils.file_io import PathManager
from yaml import parse
from PIL import Image
import os
import json
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats
import scipy
import math
import base64
import io
from io import BytesIO
from torchmetrics import JaccardIndex
import torch

#from eval_pascal_voc import *

NUM_PARTICIPANTS = 5
CLS = 'person'

IQA_interval = 0.1

Mturk_results_file = '/Users/caroline/Desktop/REforML/projects/Detection/Pilot_exp/500_C|L_brightness.csv'
human_performance = pd.read_csv(Mturk_results_file)
image_IQA_file = '/Users/caroline/Desktop/REforML/projects/Detection/Pilot_exp/images_transformations_info_500_brightness.csv'
image_IQA = pd.read_csv(image_IQA_file)
localization_results_file = '/Users/caroline/Desktop/REforML/projects/Detection/Pilot_exp/500_L_brightness.csv'
localization_results = pd.read_csv(localization_results_file)
segmentation_results_file = '/Users/caroline/Desktop/REforML/projects/Detection/Pilot_exp/500_S_brightness.csv'
segmentation_results = pd.read_csv(segmentation_results_file)

# build image name to scale dict
image_name_to_scale = {}
for i in range(len(localization_results)):
    answers = json.loads(localization_results.iloc[i]['Answer.taskAnswers'])[0]
    for i in range(22):
        if 'bbox-' + str(i) not in answers.keys():
            continue
        #print(answers['bbox-' + str(i)])

        image_url = answers['url-' + str(i)]
        if 'Sanity' in image_url:
            continue
        image_name = image_url.split('/')[-1]
        image_scale = answers['scale-' + str(i)]
        image_name_to_scale[image_name] = float(image_scale)

ORIG_IMAGES = '/Users/caroline/Desktop/REforML/projects/Detection/Pilot_exp/IMG_ORGINAL/frost/'
TRANSF_IMAGES = '/Users/caroline/Desktop/REforML/projects/Detection/Pilot_exp/IMG_TRANSFORMED/frost/'

all_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
gt_annotations = '/Users/caroline/Desktop/REforML/projects/Detection/VOCdevkit/VOC2012/Annotations/'
gt_seg = '/Users/caroline/Desktop/REforML/projects/Detection/VOCdevkit/VOC2012/SegmentationClass/'

BOX_IOU_THRESHOLD = 0.5
SEG_IOU_THRESHOLD = 0.25

NUM_samples = 100

SIGMA = 0.2

def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects

def parse_human_results(human_performance, within_interval):
    #print(within_interval)
    
    #exit()
    original_human_results = {'image_name': [], 'bbox':[], 'class': [], 'conf_score': [], 'time_appeared': [], 'seg': []}
    npos = {}
    # parse human results: transformed
    transformed_human_results = {'image_name': [], 'bbox':[], 'class': [], 'conf_score': [], 'time_appeared': [], 'seg': []}
    for d in range(len(human_performance)):
        box_num = 1
        for i in range(22):
            #image_url = human_performance.iloc[d]['Input.image' + str(i+1)]
            #image_name = image_url.split('/')[-1]
            answers = json.loads(human_performance.iloc[d]['Answer.taskAnswers'])[0]
            image_url = answers['url-' + str(i)]
            if 'Sanity' in image_url:
                continue
            image_name = image_url.split('/')[-1]
            #print(image_name)
            bbox = json.loads(human_performance.iloc[d]['Input.bbox' + str(box_num)])[0]
            box_num += 1
            class_label = json.loads(human_performance.iloc[d]['Answer.taskAnswers'])[0]['class-' + str(i)]
            if class_label not in npos:
                npos[class_label] = 0
            orig = False
            # original
            #print(image_url)

            if 'ORIGINAL' in image_url:
                if image_name not in list(within_interval['original_name']):
                    continue
                #print(image_name)
                dict_to_use = original_human_results
                orig = True
                
            # transformed
            else:
                if image_name not in list(within_interval['img_name']):
                    continue
                #print(image_name)
                dict_to_use = transformed_human_results
            # if not added already
            found = False
            for i in range(len(dict_to_use['image_name'])):
                #print(dict_to_use['image_name'][i], )
                if dict_to_use['image_name'][i] == image_name and dict_to_use['bbox'][i] == [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] and dict_to_use['class'][i] == class_label:
                    found = True
                    dict_to_use['conf_score'][i] += 1
                    dict_to_use['time_appeared'][i] += 1
            if not found:
                dict_to_use['image_name'].append(image_name)
                dict_to_use['class'].append(class_label)
                dict_to_use['conf_score'].append(1)
                dict_to_use['time_appeared'].append(1)
                # find scale
                #box_real_size = [x/image_name_to_scale[image_name] for x in bbox]
                #print(image_name_to_scale[image_name])
                dict_to_use['bbox'].append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]) # minx, miny, w, h
                dict_to_use['seg'].append('')

                if orig:
                    npos[class_label] += 1
                    '''
                    img = Image.open('/Users/caroline/Desktop/REforML/projects/Detection/Pilot_exp/IMG_ORIGINAL/frost/'+image_name)
                    fig, ax = plt.subplots()
                    ax.imshow(img)
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=3, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    
                    plt.show()
                    '''
                    #exit()

    # get conf score
    orig_confs = [original_human_results['conf_score'][i]/NUM_PARTICIPANTS if original_human_results['time_appeared'][i]<=5 else original_human_results['conf_score'][i]/original_human_results['time_appeared'][i] for i in range(len(original_human_results['conf_score'])) ] 
    original_human_results['conf_score'] = orig_confs
    # sort by conf score
    sorted_index = np.argsort([-1*x for x in orig_confs])
    original_human_results['image_name'] = [original_human_results['image_name'][i] for i in sorted_index]
    original_human_results['bbox'] = [original_human_results['bbox'][i] for i in sorted_index]
    original_human_results['class'] = [original_human_results['class'][i] for i in sorted_index]
    original_human_results['conf_score'] = [original_human_results['conf_score'][i] for i in sorted_index]


    transf_confs = [transformed_human_results['conf_score'][i]/NUM_PARTICIPANTS if transformed_human_results['time_appeared'][i]<=5 else transformed_human_results['conf_score'][i]/transformed_human_results['time_appeared'][i] for i in range(len(transformed_human_results['conf_score'])) ] 
    transformed_human_results['conf_score'] = transf_confs

    # sort by conf score
    sorted_index = np.argsort([-1*x for x in transf_confs])
    transformed_human_results['image_name'] = [transformed_human_results['image_name'][i] for i in sorted_index]
    transformed_human_results['bbox'] = [transformed_human_results['bbox'][i] for i in sorted_index]
    transformed_human_results['class'] = [transformed_human_results['class'][i] for i in sorted_index]
    transformed_human_results['conf_score'] = [transformed_human_results['conf_score'][i] for i in sorted_index]

    # load segmentation as binary code
    
    for i in range(len(segmentation_results)):
        row = segmentation_results.iloc[i]
        url = row['Input.image_url']
        #print(url)
        class_label = row['Input.class']
        bbox = json.loads(row['Input.bbox'])
        seg = row['Answer.annotatedResult.labeledImage.pngImageData']
        
        #exit()
        image_name = url.split('/')[-1].split('-')[-1]

        if not 'png' in image_name:
            if image_name not in within_interval['original_name'].tolist():
                continue
            #print(image_name)
            dict_to_use = original_human_results
            orig = True
        else:
            if image_name not in within_interval['img_name'].tolist():
                continue
            #print(image_name)
            dict_to_use = transformed_human_results
        #print(dict_to_use['image_name'])
        #exit()
        for i in range(len(dict_to_use['image_name'])):
            if dict_to_use['image_name'][i] == image_name and dict_to_use['bbox'][i] == bbox and dict_to_use['class'][i] == class_label:
                dict_to_use['seg'][i] = seg
    return original_human_results, transformed_human_results, npos

def load_gt(within_interval):
    all_recs = {'image_name': [], 'bbox':[], 'class': [], 'seg': []}
    npos = {}
    for cls in all_class_names:
        npos[cls] = 0
    for d in range(len(within_interval)):
        orig_img_name = within_interval.iloc[d]['original_name']
        orig_img_id = orig_img_name.split('.')[0].strip()
        anno_file = gt_annotations + orig_img_id + '.xml'
        img_gt = parse_rec(anno_file)
        R = [obj for obj in img_gt]
        bbox = [x["bbox"] for x in R]
        class_names = [obj['name'] for obj in img_gt]
        image_names = [orig_img_id]*len(class_names)
        for cls in class_names:
            npos[cls] += 1
        
        mask_files = [gt_seg +orig_img_id+ '.png'] * len(R)
        
        #mask_files.append(gt_seg +orig_img_id+ '.png')
        # TODO: cut and encode 
        '''
        index = 0
        gt_mask_file = gt_seg +orig_img_id+ '.png'
        for bb in bbox:
            #print(bb)
            gt_mask = np.asarray(Image.open(gt_mask_file))
            #print(gt_mask.shape)
            gt_mask = gt_mask[int(bb[1]-1):int(bb[3]), int(bb[0]-1):int(bb[2])]
            im = Image.fromarray(gt_mask)
            if not os.path.exists('/w/10/users/boyue/bootstrap-frost/gt'):
                os.mkdir('/w/10/users/boyue/bootstrap-frost/gt')
            save_name = '/w/10/users/boyue/bootstrap-frost/gt/' + orig_img_id + '-' + str(index) + '.png'
            im.save(save_name) # TODO: make generic
            mask_files.append(save_name)
            index += 1
        '''
        all_recs['image_name'] += image_names
        all_recs['bbox'] += bbox
        all_recs['class'] += class_names
        all_recs['seg'] += mask_files

    return all_recs, npos


def voc_eval_l_process(human_results, gt, ovthresh=BOX_IOU_THRESHOLD):
    nd = len(human_results['image_name'])
    #print('----------voc_eval_l_process----------------')
    #print(nd)
    
    p = np.zeros(nd)
    p_gt = []

    # TODO: also add the gt seg file for the matched box
    for d in range(nd):
        image_name = human_results['image_name'][d]
        #print(image_name)
        image_id = image_name.split('.')[0]
        
        if image_id not in gt['image_name']:
            #print(image_id)
            #print('*******orig not found********')
            p_gt.append([])
            continue
        #print(image_id)
        #print(human_results['class'][d])
        #print([gt["bbox"][i] for i in range(len(gt["bbox"])) if gt['image_name'][i]==image_id])
        #exit()
        BBGT = np.asarray([gt["bbox"][i] for i in range(len(gt["bbox"])) if gt['image_name'][i]==image_id]).astype(float)
        #print(BBGT)
        ovmax = -np.inf
        bb = human_results['bbox'][d]#(final_results['xmin'].iloc[d],final_results['ymin'].iloc[d],final_results['xmax'].iloc[d],final_results['ymax'].iloc[d])
        #print(bb)
        #exit()

        #bb = [360, 105, 466, 173]
        #BBGT = np.asarray([[375, 96, 490, 208]])
        #print(BBGT)
        #print(bb)
        p_gt.append([])
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            #print(iw, ih)
            inters = iw * ih
            #print(inters)

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )
            #print(uni)
            overlaps = inters / uni
            #print(overlaps)
            #exit()
            ovmax = np.max(overlaps)
            #if ovmax > ovthresh:
            #    print(overlaps)
            
            gt_classes_matched = np.argwhere(overlaps> ovthresh)#[gt["class"][i] for i in range(len(overlaps)) if overlaps[i] > ovthresh]
            gt_classes_matched = gt_classes_matched.reshape(len(gt_classes_matched))
            #print(gt_classes_matched)
            #print([i for i in range(len(gt["bbox"])) if gt['image_name'][i] == image_id])
            real_index = [[i for i in range(len(gt["bbox"])) if gt['image_name'][i] == image_id][x] for x in gt_classes_matched]
            #print([gt['image_name'][i] for i in real_index])
            #print([gt['class'][i] for i in real_index])
            #print(real_index)
            p_gt[d] = real_index

            if ovmax > ovthresh:
                p[d] = 1.0

    #final_results['l_IoU_p'] = p
    #final_results['IoU_matched_gt'] = p_gt
    #print(final_results)
    #final_results = final_results.sort_values(by='conf_score', ascending=False)
    #print(final_results)
    #final_results = final_results.loc[:, ~final_results.columns.str.contains('^Unnamed')]
    #final_results.to_csv(final_output_df_csv_filename, index=False)
    #print(p)
    #print(p_gt)
    #exit()
    
    return p, p_gt
   
def voc_eval_l_mAP(transformed_human_results, gt, p, p_gt, cls, npos):
    #print('-----voc_eval_l_mAP--------')
    #print(cls, cls_index)
    # cumsup of p's then
    #final_results = pd.read_csv(final_output_df_csv_filename) 
    #print(p)
    #print(p_gt)
    nd = len(p)
    #print(nd)

    cd_equals_c = np.zeros(nd) # Precision (total predicted positive)
    

    # these will be per class
    tp_precision = np.zeros(nd)
    tp_recall = np.zeros(nd)
    gt_found = []
    
    for d in range(nd):
        if transformed_human_results['class'][d] == cls:
            cd_equals_c[d] = 1.0
            if p[d] > 0:
                tp_precision[d] = 1.0

        if len(p_gt[d]) > 0: # some ground truth box matched
            #print([gt['class'][i] for i in p_gt[d]])
            #print(transformed_human_results['class'][d])
            #exit()
            if cls in [gt['class'][i] for i in p_gt[d]]:#  (p_gt[d]).split(','):
                #print(p_gt[d])
                
                if p[d] > 0:
                    #gt_found.append()
                    #print(all([i in gt_found for i in p_gt[d] if gt['class'][i] == cls]))
                    #exit()
                    if cls in [gt['class'][i] for i in p_gt[d]]:
                    #if not all([i in gt_found for i in p_gt[d] if gt['class'][i] == cls]):
                        tp_recall[d] = 1.0
                        gt_found += [i for i in p_gt[d] if gt['class'][i] == cls]

    sum_rec_l = np.cumsum(tp_recall)
    rec_l = np.nan_to_num(sum_rec_l/npos[cls])
    
    sum_prec_l = np.cumsum(tp_precision)
    #print(sum_prec_l)
    prec_l = np.nan_to_num(sum_prec_l/np.maximum(np.cumsum(cd_equals_c), np.finfo(np.float64).eps))
    #exit()
    #print(tp_recall, tp_precision)
    #exit()
    return tp_recall, tp_precision, rec_l, prec_l, cd_equals_c

def voc_eval_c_given_l_mAP(transformed_human_results, gt, p_gt, tp_recall, tp_precision, cls): 
    #print('-------voc_eval_c_given_l_mAP---------')
    #print(cls, cls_index)
    # on top of IoU being a p, if cd = c*: count
    #final_results = pd.read_csv(final_output_df_csv_filename) 
    nd = len(tp_recall)
    p_r = np.zeros(nd)
    p_p = np.zeros(nd)
    
    for d in range(nd):
        if tp_recall[d] > 0: # IoU matched and gt matched the class
            # need to check if predicted is this class
            if transformed_human_results['class'][d] == cls:
                p_r[d] = 1.0
        if tp_precision[d] > 0: # IoU matched and predicted this class
            # need to check if the gt is this class
            #print(cls, [gt['class'][i] for i in p_gt[d]])
            if cls in [gt['class'][i] for i in p_gt[d]]:
                p_p[d] = 1.0

        '''
        #print(final_results['class'].iloc[d], final_results['l_IoU_p'].iloc[d], final_results['IoU_matched_gt'].iloc[d])
        # counting given IoU and label
        if l_IoU_p[d] > 0: # IoU matched
            # count cd = c & IoU is a p -> Precision (previous p's)
            if transformed_human_results['class'][d] == cls:
                predicted_p = True
            # count c* = c & IoU is a p -> Recall (previous p's)
            if cls in (IoU_matched_gt[d]).split(','):
                #c_star_equals_c += 1
                gt_p = True
            if predicted_p and gt_p:
                p[d] = 1.0
                #print("found good box")
        '''
    #conf_scores = final_results['conf_score'].to_numpy()
    #sorted_ind = np.argsort(-conf_scores)

    sum_rec_l = np.cumsum(tp_recall)
    sum_prec_l = np.cumsum(tp_precision)

    
    #tp_to_return = p # shared by precision and recall
    #cd_equals_c = cd_equals_c[sorted_ind] # for precision
    #print(p)
    #print(sum(p))
    rec_cl = np.nan_to_num(np.divide(np.cumsum(p_r), sum_rec_l))

    prec_cl = np.nan_to_num(np.cumsum(p_p)/np.maximum(sum_prec_l, np.finfo(np.float64).eps))
    #print(p_r, p_p)
    #print(rec_cl, prec_cl)
    #exit()
    #print(sum(tp_precision))
    #print(sum(p_p))
    #print(prec_cl)
    return p_r, p_p, rec_cl, prec_cl
    #return rec_cl, prec_cl

def voc_eval_s_given_cl_mAP(final_results, gt, p_gt, p, cls, ovthersh=SEG_IOU_THRESHOLD):
    #print('-------voc_eval_s_given_cl_mAP---------')
    #print(len(gt['image_name']))
    #print("in seg")
    # for each good box from before
    # read gt, seg_results (gt is from seg class) 
    # crop both to this good box
    # compute IoU of this class, record those of > 0.5 (mask_util._iou)


    # on top of box IoU being a p and cd = c*, if seg IoU > 0.5, count
    #final_results = pd.read_csv(final_output_df_csv_filename) 
    nd = len(p_gt)
    #print(nd)
    tp = np.zeros(nd)

    for d in range(nd):
        #print(final_results['image_name'][d])
        bbox = final_results['bbox'][d]
        #print(bbox)
        if p[d] > 0: # good detcetion box
            #print(d)
            mask_file = final_results['seg'][d]
            if not mask_file:
                continue
            mask = Image.open(io.BytesIO(base64.b64decode(mask_file)))
            #print(np.asarray(mask).shape)
            mask = np.asarray(mask)[int(bbox[1])-1:int(bbox[3]), int(bbox[0])-1:int(bbox[2])] 
            #print(mask.shape)
            mask = (mask != 0)  
            #Image.fromarray(mask).show()
            #img = np.asarray(Image.open(mask_file))# cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            
            gt_index = p_gt[d]
            #print(gt_index)
            class_labels= [gt['class'][i] for i in gt_index]
            #print(class_labels.index(cls))
            seg_id = gt_index[class_labels.index(cls)]
            #print(seg_id)
            #print(len(gt['seg']))
            gt_mask_file = gt['seg'][seg_id]
            #print(gt_mask_file)
            #gt_mask_file = '/w/10/users/boyue/VOCdevkit/VOC2012/SegmentationObject/'+image_id+ '.png' #TODO: make generic
            if '.png' in gt_mask_file:
                #print(np.asarray(Image.open(gt_mask_file)).shape)               
                #Image.open(gt_mask_file).show()
                gt_mask = np.asarray(Image.open(gt_mask_file))[int(bbox[1])-1:int(bbox[3]), int(bbox[0])-1:int(bbox[2])]
                #print(gt_mask.shape)
                #Image.fromarray(gt_mask).show()
                #print(gt_mask)
                #print(all_class_names.index(cls))
                #Image.fromarray(gt_mask).show()
                gt_mask = (gt_mask == all_class_names.index(cls)+1)
                #print(gt_mask)
                #Image.fromarray(gt_mask).show()

            else:
                gt_mask = Image.open(io.BytesIO(base64.b64decode(gt_mask_file)))
                gt_mask = np.asarray(gt_mask)[int(bbox[1])-1:int(bbox[3]), int(bbox[0])-1:int(bbox[2])] 
                gt_mask = (gt_mask != 0)  

            #print(mask.shape)
            #print(gt_mask.shape)
            #im = Image.fromarray(gt_mask)
            #im.save('/w/10/users/boyue/bootstrap-frost/my_voc_orig/gt/' + image_id + str(d) + '.png')
            #print('-------')
            #print(mask.shape)
            #print(gt_mask.shape)
            jaccard = JaccardIndex(num_classes=2)
            #print(mask.shape, gt_mask.shape)
            if mask.shape[1] == 0 or gt_mask.shape[1] == 0:
                iou = 0
            else:
                iou = jaccard(torch.tensor(mask), torch.tensor(gt_mask))
                #print(iou)
                #print('-------')
                #exit()
                if iou.item() > ovthersh:
                    tp[d] = 1.0

    #sprint(sum(tp))
    good_detection = np.nan_to_num(p)
    #print(sum(good_detection))
    good_detection = np.cumsum(good_detection)
    #print(sum(tp))
    #exit()
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
    return tp, rec_scl, prec_scl


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval_d_mAP(rec_l, prec_l, rec_cl, prec_cl):  
    #print('---------d_mAP--------')
    rec = np.multiply(rec_l, rec_cl)
    prec = np.multiply(prec_l, prec_cl)
    #print(rec, prec)
    #exit()
    ap = voc_ap(rec, prec)
    #print(ap)
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

def estimate_curve_diff(rec_o, prec_o, rec_t, prec_t, num_p, npos):
    if len(rec_t) > 0:
        min_o_rec = rec_o[-1]
        min_o_prec = prec_o[-1]
        min_t_rec = rec_t[-1]
        min_t_prec = prec_t[-1]
        num_for_p = np.cumsum(num_p)[-1]
        #print('-----')
        #print(min_o_rec, min_o_prec, min_t_rec, min_t_prec)
        #print('-----')
        if min_t_rec >= 1:
            min_t_rec = 1
        if min_o_rec >= 1:
            min_o_rec = 1
        results = []
        if npos >= 5:
            #p_rec = stats.binom_test(int(npos*min_t_rec), n=npos, p=min_o_rec, alternative='less')
            results.append((min_t_rec, min_o_rec, npos))
        else:
            results.append((None, None, None))
        if num_for_p >= 5:
            #p_prec = stats.binom_test(int(num_for_p*min_t_prec), n=num_for_p, p=min_o_prec, alternative='less')
            results.append((min_t_prec, min_o_prec, num_for_p))
        else:
            results.append((None, None, None))
        return results
        #if p_rec >= p_prec:
        #    print(p_rec, npos)
        #    return p_rec, npos
        #else:
        #    print(p_prec, num_for_p)
        #    return p_prec, num_for_p
    else:
        return None, None

    #print(prec_t)
    #exit()
    # correct AP calculation
    # first append sentinel values at the end
    mrec_o = np.concatenate(([0.0], rec_o, [1.0]))
    mpre_o = np.concatenate(([0.0], prec_o, [0.0]))

    mrec_t = np.concatenate(([0.0], rec_t, [1.0]))
    mpre_t = np.concatenate(([0.0], prec_t, [0.0]))

    num_obs =  np.cumsum(np.concatenate(([0.0], num_p, [0.0])))

    
    # sort them
    sorted_indices = np.argsort(mrec_o)
    mrec_o = mrec_o[sorted_indices]
    mpre_o = mpre_o[sorted_indices]
    sorted_indices = np.argsort(mrec_t)
    mrec_t = mrec_t[sorted_indices]
    mpre_t = mpre_t[sorted_indices]
    print('*********')
    print(mrec_t, mpre_t)
    print('*********')
    #print(mrec_o, mpre_o)
    #print(mrec_t, mpre_t)
    # compute the precision envelope
    for i in range(mpre_o.size - 1, 0, -1):
        mpre_o[i - 1] = np.maximum(mpre_o[i - 1], mpre_o[i])
    for i in range(mpre_t.size - 1, 0, -1):
        mpre_t[i - 1] = np.maximum(mpre_t[i - 1], mpre_t[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    print(mpre_t)
    i = np.where(mrec_o[1:] != mrec_o[:-1])[0]
    mrec_o = mrec_o[i]
    mpre_o = mpre_o[i]
    i = np.where(mrec_t[1:] != mrec_t[:-1])[0]
    mrec_t = mrec_t[i]
    mpre_t = mpre_t[i]
    num_obs = num_obs[i]
    print(mpre_t)
    


    if len(mrec_t) > 3:
        tck, u = interpolate.splprep([mrec_o, mpre_o], s= 0)
        
        yi = interpolate.splev(mrec_t, tck)

        #new_points = interpolate.splev(np.linspace(0,1,1000), tck)
        #fig, ax = plt.subplots()
        #ax.plot(mrec_o, mpre_o, 'ro')
        #ax.plot(new_points[0], new_points[1], 'b-')
        #plt.show()
        #exit()
        '''
        line_fit_o = interpolate.splprep(mrec_o, mpre_o)
        x = mrec_o
        y = mpre_o
        print(mpre_o)
        print(mrec_o)
        spl = line_fit_o
        x2 = np.linspace(0, 1, 200)
        y2 = interpolate.splev(x2, spl, der=0)
        plt.plot(x, y, 'o', x2, y2)
        plt.show()
        exit()
        '''
        #ius = InterpolatedUnivariateSpline(x, y)
        #xi = np.sort(np.asarray(list(set(list(mrec_o) + list(mrec_t)))))
        #yi = interpolate.splev(mrec_t, line_fit_o, der=0) #line_fit(xi)
        #a = stats.ttest_ind(yi, mpre_t, alternative='less')
        #print(yi)
        #print(len(mpre_t))
        #print(len(num_obs))
        #print(len(mrec_t))
        test_values = []
        
        for j in range(len(yi[1])):
            p = round(yi[1][j],4 )
            if p == 1:
                p = 0.99 # calibration for effective binomial test
            if p >= 0 and p<=1 :
                test_value = stats.binom_test(int(num_obs[j]*mpre_t[j]), n=num_obs[j], p=p, alternative='less')
                test_values.append(test_value)
            #print(round(yi[j][0],4 )>= 0 and round(yi[j][0] , 4)<= 1)
        #test_values = [stats.binom_test(int(num_obs[i]*mpre_t[i]), n=num_obs[i], p=yi[i][0], alternative='less') for i in range(len(mrec_t)) if yi[i][0] >= 0 and yi[i][0]<=1 ]
        #print(test_values)

        tck, u = interpolate.splprep([mpre_o, mrec_o], s= 0)
        
        xi = interpolate.splev(mpre_t, tck)

        for j in range(len(xi[1])):
            p = round(xi[1][j],4 )
            if p == 1:
                p = 0.99 # calibration for effective binomial test
            if p >= 0 and p<=1 :
                test_value = stats.binom_test(int(npos*mrec_t[j]), n=npos, p=p, alternative='less')
                test_values.append(test_value)

        return min(test_values), len(mpre_t)
    else:
        return None, None


    #a = stats.ttest_ind(mrec_o, mrec_t)
    #b = stats.ttest_ind(mpre_o, mpre_t)
    #print(a, b)
    #return 0.5*(a.pvalue+b.pvalue), len(mrec_o)+len(mrec_t)

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
    '''
    if len(mrec_o) > 3 and len(mrec_t) > 3:
        #print('-----o------')
        #print(mrec_o, mpre_o)

        line_fit_o = interpolate.splrep(mrec_o, mpre_o)
        #ius = InterpolatedUnivariateSpline(x, y)
        xi = np.sort(np.asarray(list(set(list(mrec_o) + list(mrec_t)))))
        yi = interpolate.splev(xi, line_fit_o, der=0) #line_fit(xi)

        #print('-----t-----')
        #print(mrec_t, mpre_t)
        #print('-----------')
        line_fit_t = interpolate.splrep(mrec_t, mpre_t)
        #ius = InterpolatedUnivariateSpline(x, y)
        #xi_t = np.linspace(0, 1, NUM_samples)
        yi_t = interpolate.splev(xi, line_fit_t, der=0)

        difference = yi_t - yi
        return len([x for x in difference if x >=0])/len(difference), len(difference)
        print(difference)
        mu, sigma = scipy.stats.norm.fit(difference)
        distribution = scipy.stats.norm(mu, sigma)
        result = distribution.cdf(0)
        print(result)
        #if math.isnan(result):
        #    exit()
        return np.nan_to_num(result), len(xi)
    else:
        return None, None

    '''
if __name__ == '__main__':
    cur_IQA = 0
    # read human csv parse into results csv per IQA interval
    vd_scores = 1 - np.array(image_IQA['IQA'])

    final_results = {'cp_l_results': [], 'cp_cl_results': [], 'cp_d_results': [], 'cp_scl_results': [], 'cp_s_results': [], 'pp_l_results': [], 'pp_cl_results': [], 'pp_d_results': [], 'pp_scl_results': [], 'pp_s_results': []}
    #print(vd_scores)

    
    # correctness-preservation
    # find PR curve for all original
    all_orig, all_transformed, npos_all_orig = parse_human_results(human_performance, image_IQA)
    all_orig_image_ids = [x.split('.')[0] for x in all_orig['image_name']]
    #print(all_orig_image_ids)
    no_orig = []
    for t_img in all_transformed['image_name']:
        image_id = t_img.split('/')[-1].split('.')[0]
        if image_id not in all_orig_image_ids:
            no_orig.append(image_id)
    #print(set(no_orig))
    #print(len(set(no_orig)))
    #has_l = []
    #for x in set(no_orig):
    #    if x + '.jpg' in image_name_to_scale:
    #        has_l.append(x)
    #print(len(has_l))
    #exit()

    gt_all, npos_gt_all = load_gt(image_IQA)
    all_pl, all_p_gt = voc_eval_l_process(all_orig, gt_all)

    cls = CLS
    all_tp_recall, all_tp_precision, all_rec_l, all_prec_l, _ = voc_eval_l_mAP(all_orig, gt_all, all_pl, all_p_gt, cls, npos_gt_all)
    all_p_r, all_p_p, all_rec_cl, all_prec_cl = voc_eval_c_given_l_mAP(all_orig, gt_all, all_p_gt, all_tp_recall, all_tp_precision, cls)
    all_rec_d = np.nan_to_num(np.multiply(all_rec_l, all_rec_cl))
    all_prec_d = np.nan_to_num(np.multiply(all_prec_l, all_prec_cl))
    ap_all = np.nan_to_num(voc_eval_d_mAP(all_rec_l, all_prec_l, all_rec_cl, all_prec_cl))
    
    all_s_tp, all_rec_scl, all_prec_scl = voc_eval_s_given_cl_mAP(all_orig, gt_all, all_p_gt, all_p_p, cls)
    s_ap_all = np.nan_to_num(voc_eval_s_mAP(all_rec_l, all_prec_l, all_rec_cl, all_prec_cl, all_rec_scl, all_prec_scl))
    all_rec_s = np.nan_to_num(np.multiply(all_rec_d, all_rec_scl))
    all_prec_s = np.nan_to_num(np.multiply(all_prec_d, all_prec_scl))
    ap_all = np.nan_to_num(voc_eval_d_mAP(all_rec_l, all_prec_l, all_rec_cl, all_prec_cl))

    
    # prediction-preservation
    # find less than sigma, to update original to gt, need to change names to ids
    indices = np.argwhere((vd_scores <= SIGMA))
    indices = indices.reshape(len(indices))
    less_than_sigma = image_IQA.loc[indices]
    assert(len(less_than_sigma) > 0)
    #print(less_than_sigma)
    #exit()
    sigma_human_orig_results, sigma_human_transf_results, npos_sigma = parse_human_results(human_performance, less_than_sigma)
    #print(npos_sigma)
    image_ids = [x.split('.')[0] for x in sigma_human_orig_results['image_name']]
    sigma_human_orig_results['image_name'] = image_ids # this gives us the 'gt' used for prediction preservation
    # load det as gt
    s_pl, s_p_gt = voc_eval_l_process(sigma_human_transf_results, sigma_human_orig_results)
    s_tp_recall, s_tp_precision, s_rec_l, s_prec_l, _ = voc_eval_l_mAP(sigma_human_transf_results, sigma_human_orig_results, s_pl, s_p_gt, cls, npos_sigma)
    s_p_r, s_p_p, s_rec_cl, s_prec_cl = voc_eval_c_given_l_mAP(sigma_human_transf_results, sigma_human_orig_results, s_p_gt, s_tp_recall, s_tp_precision, cls)
    ap_s = np.nan_to_num(voc_eval_d_mAP(s_rec_l, s_prec_l, s_rec_cl, s_prec_cl))
    s_rec_d = np.nan_to_num(np.multiply(s_rec_l, s_rec_cl))
    s_prec_d = np.nan_to_num(np.multiply(s_prec_l, s_prec_cl))

    s_s_tp, s_rec_scl, s_prec_scl = voc_eval_s_given_cl_mAP(sigma_human_transf_results, sigma_human_orig_results, s_p_gt, s_p_p, cls)
    s_ap_s = np.nan_to_num(voc_eval_s_mAP(s_rec_l, s_prec_l, s_rec_cl, s_prec_cl, s_rec_scl, s_prec_scl))
    s_rec_s = np.nan_to_num(np.multiply(s_rec_d, s_rec_scl))
    s_prec_s = np.nan_to_num(np.multiply(s_prec_d, s_prec_scl))

    
    IQA_to_img = {}
    IQA_to_box = {}
    while cur_IQA < 1:
        print(cur_IQA)
        #if not(0.3<= cur_IQA and cur_IQA < 0.4):
        #    cur_IQA += IQA_interval
        #    continue
        
        if cur_IQA + IQA_interval > 1:
            indices = np.argwhere((vd_scores >cur_IQA) & (vd_scores <=1))
        else:
            indices = np.argwhere((vd_scores >cur_IQA) & (vd_scores <=cur_IQA + IQA_interval))

        indices = indices.reshape(len(indices))
        if len(indices) == 0:
            cur_IQA =  cur_IQA + IQA_interval
            continue
        within_interval = image_IQA.loc[indices]
        if len(within_interval) == 0:
            continue

        print('images in this interval: ' + str(len(within_interval)))
        #exit()
        IQA_to_img[cur_IQA] = len(within_interval)
        # parse human results: original + transformed
        original_human_results, transformed_human_results, npos_orig = parse_human_results(human_performance, within_interval)
        #exit()
        #print(transformed_human_results)
        #cur_IQA =  cur_IQA + IQA_interval
        #continue
        # load gt

        gt, npos_gt = load_gt(within_interval)
        IQA_to_box[cur_IQA] = (len(original_human_results['bbox']), len(transformed_human_results['bbox']), len(gt['bbox']))
        t_pl, t_p_gt = voc_eval_l_process(transformed_human_results, gt)


        # for prediction-preservation
        orig_image_ids = [x.split('.')[0] for x in original_human_results['image_name']]
        original_human_results['image_name'] = orig_image_ids
        st_pl, st_p_gt = voc_eval_l_process(transformed_human_results, original_human_results)
        #exit()

        for cls in [CLS]:
            
            # correctness-preservation
            # calculate PR of l 
            #o_tp_recall, o_tp_precision, o_rec_l, o_prec_l = voc_eval_l_mAP(original_human_results, gt, o_pl, o_p_gt, cls, npos_gt)
            t_tp_recall, t_tp_precision, t_rec_l, t_prec_l, cd_equals_c = voc_eval_l_mAP(transformed_human_results, gt, t_pl, t_p_gt, cls, npos_gt)
            rec_and_prec = estimate_curve_diff(all_rec_l, all_prec_l, t_rec_l, t_prec_l, cd_equals_c, npos_gt[cls])
            #print(t_rec_l, t_prec_l)
            if rec_and_prec[0][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['cp_l_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['cp_l_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            # calculate PR of c|l 
            #o_p_r, o_p_p, o_rec_cl, o_prec_cl = voc_eval_c_given_l_mAP(original_human_results, gt, o_p_gt, o_tp_recall, o_tp_precision, cls)
            t_p_r, t_p_p, t_rec_cl, t_prec_cl = voc_eval_c_given_l_mAP(transformed_human_results, gt, t_p_gt, t_tp_recall, t_tp_precision, cls)
            #print(t_rec_cl, t_prec_cl)
            rec_and_prec = estimate_curve_diff(all_rec_cl, all_prec_cl, t_rec_cl, t_prec_cl, t_tp_precision, npos_gt[cls])
            #print(conf_ninety_five)
            #continue
            if rec_and_prec[0][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['cp_cl_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['cp_cl_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            # calculate mAP of d 
            #ap_o = np.nan_to_num(voc_eval_d_mAP(o_rec_l, o_prec_l, o_rec_cl, o_prec_cl))
            t_rec_d = np.nan_to_num(np.multiply(t_rec_l, t_rec_cl))
            t_prec_d = np.nan_to_num(np.multiply(t_prec_l, t_prec_cl))
            rec_and_prec = estimate_curve_diff(all_rec_d, all_prec_d, t_rec_d, t_prec_d, cd_equals_c, npos_gt[cls])
            ap_t = np.nan_to_num(voc_eval_d_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl))
            #if cls in npos_gt and npos_gt[cls] > 0:
            if rec_and_prec[0][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['cp_d_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['cp_d_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            #exit()

            t_s_tp, t_rec_scl, t_prec_scl = voc_eval_s_given_cl_mAP(transformed_human_results, gt, t_p_gt, t_p_p, cls)
            rec_and_prec = estimate_curve_diff(all_rec_scl, all_prec_scl, t_rec_scl, t_prec_scl, t_p_p, npos_gt[cls])
            #print(conf_ninety_five)
            #continue
            if rec_and_prec[0][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['cp_scl_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['cp_scl_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            s_ap_t = np.nan_to_num(voc_eval_s_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl, t_rec_scl, t_prec_scl))

            t_rec_s = np.nan_to_num(np.multiply(t_rec_d, t_rec_scl))
            t_prec_s = np.nan_to_num(np.multiply(t_prec_d, t_prec_scl))
            rec_and_prec = estimate_curve_diff(all_rec_s, all_prec_s, t_rec_s, t_prec_s, cd_equals_c, npos_gt[cls])

            #if cls in npos_gt and npos_gt[cls] > 0:
            if rec_and_prec[0][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['cp_s_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['cp_s_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))
            
            # prediction-preservation
            # calculate PR of l 
            #s_tp_recall, s_tp_precision, s_rec_l, s_prec_l = voc_eval_l_mAP(sigma_human_transf_results, sigma_human_orig_results, s_pl, s_p_gt, cls, npos_sigma)
            #print(transformed_human_results, original_human_results)
            t_tp_recall, t_tp_precision, t_rec_l, t_prec_l, cd_equals_c = voc_eval_l_mAP(transformed_human_results, original_human_results, st_pl, st_p_gt, cls, npos_orig)
            #if cur_IQA >= 0.3 and cur_IQA < 0.4:
            #    print(original_human_results, transformed_human_results)
            #    print(len(original_human_results['image_name']), len(transformed_human_results['image_name']))
            #    print(sum(st_pl), st_p_gt)
            #    print(sum(t_tp_recall), sum(t_tp_precision))
                #exit()
            rec_and_prec = estimate_curve_diff(s_rec_l, s_prec_l, t_rec_l, t_prec_l, cd_equals_c, npos_orig[cls])
            
            if rec_and_prec[0][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['pp_l_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['pp_l_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            # calculate PR of c|l 
            #s_p_r, s_p_p, s_rec_cl, s_prec_cl = voc_eval_c_given_l_mAP(sigma_human_transf_results, sigma_human_orig_results, s_p_gt, s_tp_recall, s_tp_precision, cls)
            t_p_r, t_p_p, t_rec_cl, t_prec_cl = voc_eval_c_given_l_mAP(transformed_human_results, original_human_results, st_p_gt, t_tp_recall, t_tp_precision, cls)
            
            rec_and_prec = estimate_curve_diff(s_rec_cl, s_prec_cl, t_rec_cl, t_prec_cl, t_tp_precision, npos_orig[cls])
            
            if rec_and_prec[0][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['pp_cl_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['pp_cl_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            # calculate mAP of d 
            #ap_s = np.nan_to_num(voc_eval_d_mAP(s_rec_l, s_prec_l, s_rec_cl, s_prec_cl))
            t_rec_d = np.nan_to_num(np.multiply(t_rec_l, t_rec_cl))
            t_prec_d = np.nan_to_num(np.multiply(t_prec_l, t_prec_cl))

            rec_and_prec = estimate_curve_diff(s_rec_d, s_prec_d, t_rec_d, t_prec_d, cd_equals_c, npos_orig[cls])
            
            if rec_and_prec[0][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['pp_d_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['pp_d_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            ap_t = np.nan_to_num(voc_eval_d_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl))


            t_s_tp, t_rec_scl, t_prec_scl = voc_eval_s_given_cl_mAP(transformed_human_results, original_human_results, st_p_gt, t_p_p, cls)
            rec_and_prec = estimate_curve_diff(s_rec_scl, s_prec_scl, t_rec_scl, t_prec_scl, t_p_p, npos_orig[cls])
            #print(conf_ninety_five)
            #continue
            if rec_and_prec[0][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['pp_scl_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['pp_scl_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            ap_t = np.nan_to_num(voc_eval_d_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl))

            t_rec_s = np.nan_to_num(np.multiply(t_rec_d, t_rec_scl))
            t_prec_s = np.nan_to_num(np.multiply(t_prec_d, t_prec_scl))

            rec_and_prec = estimate_curve_diff(s_rec_s, s_prec_s, t_rec_s, t_rec_s, cd_equals_c, npos_orig[cls])
            
            if rec_and_prec[0][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['pp_s_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                #print(mu_ninety_five)
                #if cls in npos_gt and npos_gt[cls] > 0:
                #continue
                final_results['pp_s_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))


            #s_ap_t = np.nan_to_num(voc_eval_s_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl, t_rec_scl, t_prec_scl))
            #if cls in npos_gt and npos_gt[cls] > 0:
            #    pp_s_results.append((cls, cur_IQA, s_ap_t, s_ap_s))
            
        cur_IQA =  cur_IQA + IQA_interval
    print('number of images in each IQA interval:')
    print(IQA_to_img)
    print('number of boxes in each interval (original, transformed, gt)')
    print(IQA_to_box)

    #exit()
    STATS_THRES = 0.05
    for cls in [CLS]:
        # for cp
        #cp_l_results = []
        #cp_cl_results = []
        #cp_d_results = []
        from csaps import csaps

        for req in ['cp', 'pp']:
            for task in ['l', 'cl', 'd', 'scl', 's']:
                cur_list = req + '_' + task + '_results'
                print(cur_list + ':')
                #print(final_results[cur_list])
                prec_and_rec = []
                for func in ['rec', 'prec']:
                    print(func)
                    IQAs = [i[1] for i in final_results[cur_list] if i[0] == cls and i[2] == func]
                    print(IQAs)
                    probs = [i[3][0] for i in final_results[cur_list] if i[0] == cls and i[2] == func]
                    orig_prob = [i[3][1] for i in final_results[cur_list] if i[0] == cls and i[2] == func]
                    num_points = [i[3][2] for i in final_results[cur_list] if i[0] == cls and i[2] == func]
                    p_values = [round(stats.binom_test(int(probs[i]*num_points[i]), n=num_points[i], p=orig_prob[i], alternative='less'), 2) for i in range(len(IQAs))]
                    print(p_values)
                    #print(orig_prob)
                    #print('-----o------')
                    #print(mrec_o, mpre_o)
                    if len(IQAs) >= 2:
                        
                        #yi = csaps(IQAs, probs, IQAs, weights=[i/sum(num_points) for i in num_points], smooth=0.8)
                        yi = csaps(IQAs, probs, IQAs, smooth=0.8)
                        #yi = line_fit(rec_IQAs)

                        #line_fit_o = interpolate.splrep(rec_IQAs, rec_probs)
                        #ius = InterpolatedUnivariateSpline(x, y)
                        #yi = interpolate.splev(rec_IQAs, line_fit_o, der=1) #line_fit(xi)
                        #print(rec_probs)
                        #print(yi)
                        #p_values = [round(stats.binom_test(int(yi[i]*num_points[i]), n=num_points[i], p=orig_prob[i], alternative='less'),2) for i in range(len(IQAs))]
                        print(p_values)
                        prev = 1
                        index = len(IQAs)-1
                        for i in range(len(IQAs)):
                            cur = p_values[i]
                            if cur <= STATS_THRES and prev <= STATS_THRES:
                                #if prev <= STATS_THRES:
                                index = i-1
                                break
                            #else:
                            #    if prev > STATS_THRES:
                            #        break
                            prev = cur                
                        
                        #if index <= 1: # meaning it has not seen two consecutive positives all the way till the front (fluctuating all the time)
                        #    index = len(IQAs)-1 # in this case we consider performance not dropping
                        prec_and_rec.append(IQAs[index])
                    else:
                        prec_and_rec.append(1-IQA_interval)
                if len(prec_and_rec) > 0:
                    print(min(prec_and_rec))
                #else:
                #    print(IQAs[-1])
                

        '''
        IQAs = [i[1] for i in cp_l_results if i[0] == cls]
        probs = [i[2] for i in cp_l_results if i[0] == cls]
        num_points = [i[3] for i in cp_l_results if i[0] == cls]

        prev = 0
        index = len(IQAs)-1
        for i in range(len(IQAs)):
            if probs[len(IQAs)-1-i] <= STATS_THRES:
                index = len(IQAs)-1-i
            else:
                if prev > STATS_THRES:
                    break
            prev = probs[len(IQAs)-1-i]

        if index <= 1: # meaning it has not seen two consecutive positives all the way till the front (fluctuating all the time)
            index = len(IQAs)-1 # in this case we consider performance not dropping
        print(IQAs[index])

        exit()
        #prev = 1
        #for i in range(len(IQAs)):
        #    if i == len(IQAs) - 1:
        #        print(IQAs[i])
        #        break
        #    if prev <= STATS_THRES and probs[i] <= STATS_THRES:
        #        print(IQAs[i-1])
        #        break
        #    prev = probs[i]
            
            #print(IQAs[i])
            #print(stats.binom_test(int(num_points[i]*probs[i]), n=num_points[i], p=0.95, alternative='less'))

        print('--------------cl----------')
        print(cp_cl_results)
        IQAs = [i[1] for i in cp_cl_results if i[0] == cls]
        probs = [i[2] for i in cp_cl_results if i[0] == cls]
        prev = 0
        index = len(IQAs)-1
        for i in range(len(IQAs)):
            if probs[len(IQAs)-1-i] <= STATS_THRES:
                index = len(IQAs)-1-i
            else:
                if prev > STATS_THRES:
                    break
        if index <= 1: # meaning it has not seen two consecutive positives all the way till the front (fluctuating all the time)
            index = len(IQAs)-1 # in this case we consider performance not dropping
        print(IQAs[index])

        #prev = 1
        #for i in range(len(IQAs)):
        #    if i == len(IQAs) - 1:
        #        print(IQAs[i])
        #        break
        #    if prev <= STATS_THRES and probs[i] <= STATS_THRES:
        #        print(IQAs[i-1])
        #        break
        #    prev = probs[i]
        
        print('--------------d----------')   
        print(cp_d_results)     
        IQAs = [i[1] for i in cp_d_results if i[0] == cls]
        probs = [i[2] for i in cp_d_results if i[0] == cls]
        num_points = [i[3] for i in cp_d_results if i[0] == cls]
        prev = 0
        index = len(IQAs)-1
        for i in range(len(IQAs)):
            if probs[len(IQAs)-1-i] <= STATS_THRES:
                index = len(IQAs)-1-i
            else:
                if prev > STATS_THRES:
                    break
            prev = probs[len(IQAs)-1-i]
        if index <= 1: # meaning it has not seen two consecutive positives all the way till the front (fluctuating all the time)
            index = len(IQAs)-1 # in this case we consider performance not dropping
        print(IQAs[index])

        #prev = 1
        #for i in range(len(IQAs)):
        #    if i == len(IQAs) - 1:
        #        print(IQAs[i])
        #        break
        #    if prev <= STATS_THRES and probs[i] <= STATS_THRES:
        #        print(IQAs[i-1])
        #        break
        #    prev = probs[i]

        print('--------------scl----------')
        print(cp_scl_results)
        IQAs = [i[1] for i in cp_scl_results if i[0] == cls]
        probs = [i[2] for i in cp_scl_results if i[0] == cls]
        num_points = [i[3] for i in cp_scl_results if i[0] == cls]
        prev = 0
        index = len(IQAs)-1
        for i in range(len(IQAs)):
            if probs[len(IQAs)-1-i] <= STATS_THRES:
                index = len(IQAs)-1-i
            else:
                if prev > STATS_THRES:
                    break
            prev = probs[len(IQAs)-1-i]
        if index <= 1: # meaning it has not seen two consecutive positives all the way till the front (fluctuating all the time)
            index = len(IQAs)-1 # in this case we consider performance not dropping
        print(IQAs[index])

        #prev = 1
        #for i in range(len(IQAs)):
        #    print(prev, probs[i])
        #    if i == len(IQAs) - 1:
        #        print(IQAs[i])
        #        break
        #    if prev <= STATS_THRES and probs[i] <= STATS_THRES:
        #        print(IQAs[i-1])
        #        break
        #    prev = probs[i]
        
        print('--------------s----------')   
        print(cp_s_results)     
        IQAs = [i[1] for i in cp_s_results if i[0] == cls]
        probs = [i[2] for i in cp_s_results if i[0] == cls]
        num_points = [i[3] for i in cp_s_results if i[0] == cls]
        prev = 0
        index = len(IQAs)-1
        for i in range(len(IQAs)):
            #print(prev, probs[len(IQAs)-1-i])
            if probs[len(IQAs)-1-i] <= STATS_THRES:
                index = len(IQAs)-1-i
            else:
                if prev > STATS_THRES:
                    break
            prev = probs[len(IQAs)-1-i]
        if index <= 1: # meaning it has not seen two consecutive positives all the way till the front (fluctuating all the time)
            index = len(IQAs)-1 # in this case we consider performance not dropping
        print(IQAs[index])

        #prev = 1
        #for i in range(len(IQAs)):
        #    if i == len(IQAs) - 1:
        #        print(IQAs[i])
        #        break
        #    if prev <= STATS_THRES and probs[i] <= STATS_THRES:
        #        print(IQAs[i-1])
        #        break
        #    prev = probs[i]

        print('++++++++++++++++++++++++++++++++++++++')
        print('--------------l----------')
        print(pp_l_results)
        IQAs = [i[1] for i in pp_l_results if i[0] == cls]
        probs = [i[2] for i in pp_l_results if i[0] == cls]
        num_points = [i[3] for i in pp_l_results if i[0] == cls]
        prev = 0
        index = len(IQAs)-1
        for i in range(len(IQAs)):
            if probs[len(IQAs)-1-i] <= STATS_THRES:
                index = len(IQAs)-1-i
            else:
                if prev > STATS_THRES:
                    break
            prev = probs[len(IQAs)-1-i]
        if index <= 1: # meaning it has not seen two consecutive positives all the way till the front (fluctuating all the time)
            index = len(IQAs)-1 # in this case we consider performance not dropping
        print(IQAs[index])

        #prev = 1
        #for i in range(len(IQAs)):
        #    if i == len(IQAs) - 1:
        #        print(IQAs[i])
        #        break
        #    if prev <= STATS_THRES and probs[i] <= STATS_THRES:
        #        print(IQAs[i-1])
        #        break
        #    prev = probs[i]

        print('--------------cl----------')
        print(pp_cl_results)
        IQAs = [i[1] for i in pp_cl_results if i[0] == cls]
        probs = [i[2] for i in pp_cl_results if i[0] == cls]
        num_points = [i[3] for i in pp_cl_results if i[0] == cls]
        prev = 0
        index = len(IQAs)-1
        for i in range(len(IQAs)):
            if probs[len(IQAs)-1-i] <= STATS_THRES:
                index = len(IQAs)-1-i
            else:
                if prev > STATS_THRES:
                    break
            prev = probs[len(IQAs)-1-i]
        if index <= 1: # meaning it has not seen two consecutive positives all the way till the front (fluctuating all the time)
            index = len(IQAs)-1 # in this case we consider performance not dropping
        print(IQAs[index])

        #prev = 1
        #for i in range(len(IQAs)):
        #    if i == len(IQAs) - 1:
        #        print(IQAs[i])
        #        break
        #    if prev <= STATS_THRES and probs[i] <= STATS_THRES:
        #        print(IQAs[i-1])
        #        break
        #    prev = probs[i]
        
        print('--------------d----------')        
        print(pp_d_results)
        IQAs = [i[1] for i in pp_d_results if i[0] == cls]
        probs = [i[2] for i in pp_d_results if i[0] == cls]
        num_points = [i[3] for i in pp_d_results if i[0] == cls]
        prev = 0
        index = len(IQAs)-1
        for i in range(len(IQAs)):
            if probs[len(IQAs)-1-i] <= STATS_THRES:
                index = len(IQAs)-1-i
            else:
                if prev > STATS_THRES:
                    break
            prev = probs[len(IQAs)-1-i]
        if index <= 1: # meaning it has not seen two consecutive positives all the way till the front (fluctuating all the time)
            index = len(IQAs)-1 # in this case we consider performance not dropping
        print(IQAs[index])

        #prev = 1
        #for i in range(len(IQAs)):
        #    if i == len(IQAs) - 1:
        #        print(IQAs[i])
        #        break
        #    if prev <= STATS_THRES and probs[i] <= STATS_THRES:
        #        print(IQAs[i-1])
        #        break
        #    prev = probs[i]

        print('--------------scl----------')
        print(pp_scl_results)
        IQAs = [i[1] for i in pp_scl_results if i[0] == cls]
        probs = [i[2] for i in pp_scl_results if i[0] == cls]
        num_points = [i[3] for i in pp_scl_results if i[0] == cls]
        prev = 0
        index = len(IQAs)-1
        for i in range(len(IQAs)):
            if probs[len(IQAs)-1-i] <= STATS_THRES:
                index = len(IQAs)-1-i
            else:
                if prev > STATS_THRES:
                    break
            prev = probs[len(IQAs)-1-i]
        if index <= 1: # meaning it has not seen two consecutive positives all the way till the front (fluctuating all the time)
            index = len(IQAs)-1 # in this case we consider performance not dropping
        print(IQAs[index])

        #prev = 1
        #for i in range(len(IQAs)):
        #    if i == len(IQAs) - 1:
        #        print(IQAs[i])
        #        break
        #    if prev <= STATS_THRES and probs[i] <= STATS_THRES:
        #        print(IQAs[i-1])
        #        break
        #    prev = probs[i]
        
        print('--------------s----------')        
        print(pp_s_results)
        IQAs = [i[1] for i in pp_s_results if i[0] == cls]
        probs = [i[2] for i in pp_s_results if i[0] == cls]
        num_points = [i[3] for i in pp_s_results if i[0] == cls]

        prev = 0
        index = len(IQAs)-1
        for i in range(len(IQAs)):
            if probs[len(IQAs)-1-i] <= STATS_THRES:
                index = len(IQAs)-1-i
            else:
                if prev > STATS_THRES:
                    break
            prev = probs[len(IQAs)-1-i]
        if index <= 1: # meaning it has not seen two consecutive positives all the way till the front (fluctuating all the time)
            index = len(IQAs)-1 # in this case we consider performance not dropping
        print(IQAs[index])


        #prev = 1
        #for i in range(len(IQAs)):
        #    if i == len(IQAs) - 1:
        #        print(IQAs[i])
        #        break
        #    if prev <= STATS_THRES and probs[i] <= STATS_THRES:
        #       print(IQAs[i-1])
        #        break
        #    prev = probs[i]

  

        '''