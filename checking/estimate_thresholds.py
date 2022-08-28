import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.utils.file_io import PathManager
from PIL import Image
import json
import scipy.stats as stats
import base64
import io
from torchmetrics import JaccardIndex
import torch
from src.constant import *

NUM_PARTICIPANTS = 5
CLS = 'person'
TRANSFORMATION = 'frost'
IQA_interval = 0.1

Mturk_results_file = '../experiment/experiment_results/500_C|L.csv'
human_performance = pd.read_csv(Mturk_results_file)
image_IQA_file = '../experiment/experiment_results/images_transformations_info_500.csv'
image_IQA = pd.read_csv(image_IQA_file)
localization_results_file = '../experiment/experiment_results/500_L.csv'
localization_results = pd.read_csv(localization_results_file)
segmentation_results_file = '../experiment/experiment_resultsp/500_S.csv'
segmentation_results = pd.read_csv(segmentation_results_file)

# build image name to scale dict
image_name_to_scale = {}
for i in range(len(localization_results)):
    answers = json.loads(localization_results.iloc[i]['Answer.taskAnswers'])[0]
    for i in range(22):
        if 'bbox-' + str(i) not in answers.keys():
            continue
        image_url = answers['url-' + str(i)]
        if 'Sanity' in image_url:
            continue
        image_name = image_url.split('/')[-1]
        image_scale = answers['scale-' + str(i)]
        image_name_to_scale[image_name] = float(image_scale)

all_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
gt_annotations = str(VOC_ROOT / 'VOC2012' / 'Annotations') + '/'
gt_seg = str(VOC_ROOT / 'VOC2012' / 'SegmentationClass') + '/'

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
    original_human_results = {'image_name': [], 'bbox':[], 'class': [], 'conf_score': [], 'time_appeared': [], 'seg': []}
    npos = {}
    # parse human results: transformed
    transformed_human_results = {'image_name': [], 'bbox':[], 'class': [], 'conf_score': [], 'time_appeared': [], 'seg': []}
    for d in range(len(human_performance)):
        box_num = 1
        for i in range(22):
            answers = json.loads(human_performance.iloc[d]['Answer.taskAnswers'])[0]
            image_url = answers['url-' + str(i)]
            if 'Sanity' in image_url:
                continue
            image_name = image_url.split('/')[-1]
            bbox = json.loads(human_performance.iloc[d]['Input.bbox' + str(box_num)])[0]
            box_num += 1
            class_label = json.loads(human_performance.iloc[d]['Answer.taskAnswers'])[0]['class-' + str(i)]
            if class_label not in npos:
                npos[class_label] = 0
            orig = False

            if 'ORIGINAL' in image_url:
                if image_name not in list(within_interval['original_name']):
                    continue
                dict_to_use = original_human_results
                orig = True
                
            # transformed
            else:
                if image_name not in list(within_interval['img_name']):
                    continue
                dict_to_use = transformed_human_results
            # if not added already
            found = False
            for i in range(len(dict_to_use['image_name'])):
                if dict_to_use['image_name'][i] == image_name and dict_to_use['bbox'][i] == [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] and dict_to_use['class'][i] == class_label:
                    found = True
                    dict_to_use['conf_score'][i] += 1
                    dict_to_use['time_appeared'][i] += 1
            if not found:
                dict_to_use['image_name'].append(image_name)
                dict_to_use['class'].append(class_label)
                dict_to_use['conf_score'].append(1)
                dict_to_use['time_appeared'].append(1)
                dict_to_use['bbox'].append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]) # minx, miny, w, h
                dict_to_use['seg'].append('')

                if orig:
                    npos[class_label] += 1
                    

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
        class_label = row['Input.class']
        bbox = json.loads(row['Input.bbox'])
        seg = row['Answer.annotatedResult.labeledImage.pngImageData']
        
        image_name = url.split('/')[-1].split('-')[-1]

        if not 'png' in image_name:
            if image_name not in within_interval['original_name'].tolist():
                continue
            dict_to_use = original_human_results
            orig = True
        else:
            if image_name not in within_interval['img_name'].tolist():
                continue
            dict_to_use = transformed_human_results
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
        
        all_recs['image_name'] += image_names
        all_recs['bbox'] += bbox
        all_recs['class'] += class_names
        all_recs['seg'] += mask_files

    return all_recs, npos


def voc_eval_l_process(human_results, gt, ovthresh=BOX_IOU_THRESHOLD):
    nd = len(human_results['image_name'])

    p = np.zeros(nd)
    p_gt = []

    for d in range(nd):
        image_name = human_results['image_name'][d]
        image_id = image_name.split('.')[0]
        
        if image_id not in gt['image_name']:
            p_gt.append([])
            continue
        BBGT = np.asarray([gt["bbox"][i] for i in range(len(gt["bbox"])) if gt['image_name'][i]==image_id]).astype(float)
        ovmax = -np.inf
        bb = human_results['bbox'][d]
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
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            
            gt_classes_matched = np.argwhere(overlaps> ovthresh)
            gt_classes_matched = gt_classes_matched.reshape(len(gt_classes_matched))
            
            real_index = [[i for i in range(len(gt["bbox"])) if gt['image_name'][i] == image_id][x] for x in gt_classes_matched]

            p_gt[d] = real_index

            if ovmax > ovthresh:
                p[d] = 1.0
    
    return p, p_gt
   
def voc_eval_l_mAP(transformed_human_results, gt, p, p_gt, cls, npos):
    nd = len(p)
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

        if len(p_gt[d]) > 0: 
            if cls in [gt['class'][i] for i in p_gt[d]]:
                if p[d] > 0:
                    if cls in [gt['class'][i] for i in p_gt[d]]:
                        tp_recall[d] = 1.0
                        gt_found += [i for i in p_gt[d] if gt['class'][i] == cls]

    sum_rec_l = np.cumsum(tp_recall)
    rec_l = np.nan_to_num(sum_rec_l/npos[cls])
    
    sum_prec_l = np.cumsum(tp_precision)
    prec_l = np.nan_to_num(sum_prec_l/np.maximum(np.cumsum(cd_equals_c), np.finfo(np.float64).eps))
    return tp_recall, tp_precision, rec_l, prec_l, cd_equals_c

def voc_eval_c_given_l_mAP(transformed_human_results, gt, p_gt, tp_recall, tp_precision, cls): 
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

    sum_rec_l = np.cumsum(tp_recall)
    sum_prec_l = np.cumsum(tp_precision)

    rec_cl = np.nan_to_num(np.divide(np.cumsum(p_r), sum_rec_l))
    prec_cl = np.nan_to_num(np.cumsum(p_p)/np.maximum(sum_prec_l, np.finfo(np.float64).eps))
    
    return p_r, p_p, rec_cl, prec_cl

def voc_eval_s_given_cl_mAP(final_results, gt, p_gt, p, cls, ovthersh=SEG_IOU_THRESHOLD):
    nd = len(p_gt)
    tp = np.zeros(nd)

    for d in range(nd):
        bbox = final_results['bbox'][d]
        if p[d] > 0: # good detcetion box
            mask_file = final_results['seg'][d]
            if not mask_file:
                continue
            mask = Image.open(io.BytesIO(base64.b64decode(mask_file)))
            mask = np.asarray(mask)[int(bbox[1])-1:int(bbox[3]), int(bbox[0])-1:int(bbox[2])] 
            mask = (mask != 0)  

            gt_index = p_gt[d]
            class_labels= [gt['class'][i] for i in gt_index]
            seg_id = gt_index[class_labels.index(cls)]
            gt_mask_file = gt['seg'][seg_id]
            if '.png' in gt_mask_file:
                gt_mask = np.asarray(Image.open(gt_mask_file))[int(bbox[1])-1:int(bbox[3]), int(bbox[0])-1:int(bbox[2])]
                gt_mask = (gt_mask == all_class_names.index(cls)+1)

            else:
                gt_mask = Image.open(io.BytesIO(base64.b64decode(gt_mask_file)))
                gt_mask = np.asarray(gt_mask)[int(bbox[1])-1:int(bbox[3]), int(bbox[0])-1:int(bbox[2])] 
                gt_mask = (gt_mask != 0)  

            jaccard = JaccardIndex(num_classes=2)
            if mask.shape[1] == 0 or gt_mask.shape[1] == 0:
                iou = 0
            else:
                iou = jaccard(torch.tensor(mask), torch.tensor(gt_mask))
                if iou.item() > ovthersh:
                    tp[d] = 1.0

    good_detection = np.nan_to_num(p)
    good_detection = np.cumsum(good_detection)

    prec_scl = np.nan_to_num(np.divide(np.cumsum(tp), good_detection))
    rec_scl = np.nan_to_num(np.cumsum(tp)/np.maximum(good_detection, np.finfo(np.float64).eps))
    
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
    rec = np.multiply(rec_l, rec_cl)
    prec = np.multiply(prec_l, prec_cl)
    ap = voc_ap(rec, prec)
    return ap

def voc_eval_s_mAP(rec_l, prec_l, rec_cl, prec_cl, rec_scl, prec_scl):

    rec = np.multiply(np.multiply(rec_l, rec_cl), rec_scl)
    prec = np.multiply(np.multiply(prec_l, prec_cl), prec_scl)

    ap = voc_ap(rec, prec)
    return ap

def estimate_curve_diff(rec_o, prec_o, rec_t, prec_t, num_p, npos):
    if len(rec_t) > 0:
        min_o_rec = rec_o[-1]
        min_o_prec = prec_o[-1]
        min_t_rec = rec_t[-1]
        min_t_prec = prec_t[-1]
        num_for_p = np.cumsum(num_p)[-1]

        if min_t_rec >= 1:
            min_t_rec = 1
        if min_o_rec >= 1:
            min_o_rec = 1
        results = []
        if npos >= 5:
            results.append((min_t_rec, min_o_rec, npos))
        else:
            results.append((None, None, None))
        if num_for_p >= 5:
            results.append((min_t_prec, min_o_prec, num_for_p))
        else:
            results.append((None, None, None))
        return results

    else:
        return None, None

  
if __name__ == '__main__':
    cur_IQA = 0
    # read human csv parse into results csv per IQA interval
    vd_scores = 1 - np.array(image_IQA['IQA'])

    final_results = {'cp_l_results': [], 'cp_cl_results': [], 'cp_d_results': [], 'cp_scl_results': [], 'cp_s_results': [], 'pp_l_results': [], 'pp_cl_results': [], 'pp_d_results': [], 'pp_scl_results': [], 'pp_s_results': []}
    
    # correctness-preservation
    # find PR curve for all original
    all_orig, all_transformed, npos_all_orig = parse_human_results(human_performance, image_IQA)
    all_orig_image_ids = [x.split('.')[0] for x in all_orig['image_name']]
    no_orig = []
    for t_img in all_transformed['image_name']:
        image_id = t_img.split('/')[-1].split('.')[0]
        if image_id not in all_orig_image_ids:
            no_orig.append(image_id)

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

    sigma_human_orig_results, sigma_human_transf_results, npos_sigma = parse_human_results(human_performance, less_than_sigma)
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
        IQA_to_img[cur_IQA] = len(within_interval)
        # parse human results: original + transformed
        original_human_results, transformed_human_results, npos_orig = parse_human_results(human_performance, within_interval)

        # load gt
        gt, npos_gt = load_gt(within_interval)
        IQA_to_box[cur_IQA] = (len(original_human_results['bbox']), len(transformed_human_results['bbox']), len(gt['bbox']))
        t_pl, t_p_gt = voc_eval_l_process(transformed_human_results, gt)


        # for prediction-preservation
        orig_image_ids = [x.split('.')[0] for x in original_human_results['image_name']]
        original_human_results['image_name'] = orig_image_ids
        st_pl, st_p_gt = voc_eval_l_process(transformed_human_results, original_human_results)

        for cls in [CLS]:
            
            # correctness-preservation
            # calculate PR of l 
            t_tp_recall, t_tp_precision, t_rec_l, t_prec_l, cd_equals_c = voc_eval_l_mAP(transformed_human_results, gt, t_pl, t_p_gt, cls, npos_gt)
            rec_and_prec = estimate_curve_diff(all_rec_l, all_prec_l, t_rec_l, t_prec_l, cd_equals_c, npos_gt[cls])
            if rec_and_prec[0][0] is not None:
                final_results['cp_l_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                final_results['cp_l_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            # calculate PR of c|l 
            t_p_r, t_p_p, t_rec_cl, t_prec_cl = voc_eval_c_given_l_mAP(transformed_human_results, gt, t_p_gt, t_tp_recall, t_tp_precision, cls)
            rec_and_prec = estimate_curve_diff(all_rec_cl, all_prec_cl, t_rec_cl, t_prec_cl, t_tp_precision, npos_gt[cls])
            if rec_and_prec[0][0] is not None:
                final_results['cp_cl_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                final_results['cp_cl_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            # calculate mAP of d 
            t_rec_d = np.nan_to_num(np.multiply(t_rec_l, t_rec_cl))
            t_prec_d = np.nan_to_num(np.multiply(t_prec_l, t_prec_cl))
            rec_and_prec = estimate_curve_diff(all_rec_d, all_prec_d, t_rec_d, t_prec_d, cd_equals_c, npos_gt[cls])
            ap_t = np.nan_to_num(voc_eval_d_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl))
            if rec_and_prec[0][0] is not None:
                final_results['cp_d_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                final_results['cp_d_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            t_s_tp, t_rec_scl, t_prec_scl = voc_eval_s_given_cl_mAP(transformed_human_results, gt, t_p_gt, t_p_p, cls)
            rec_and_prec = estimate_curve_diff(all_rec_scl, all_prec_scl, t_rec_scl, t_prec_scl, t_p_p, npos_gt[cls])

            if rec_and_prec[0][0] is not None:
                final_results['cp_scl_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                final_results['cp_scl_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            s_ap_t = np.nan_to_num(voc_eval_s_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl, t_rec_scl, t_prec_scl))

            t_rec_s = np.nan_to_num(np.multiply(t_rec_d, t_rec_scl))
            t_prec_s = np.nan_to_num(np.multiply(t_prec_d, t_prec_scl))
            rec_and_prec = estimate_curve_diff(all_rec_s, all_prec_s, t_rec_s, t_prec_s, cd_equals_c, npos_gt[cls])

            if rec_and_prec[0][0] is not None:
                final_results['cp_s_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                final_results['cp_s_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))
            
            # prediction-preservation
            # calculate PR of l 
            t_tp_recall, t_tp_precision, t_rec_l, t_prec_l, cd_equals_c = voc_eval_l_mAP(transformed_human_results, original_human_results, st_pl, st_p_gt, cls, npos_orig)
            rec_and_prec = estimate_curve_diff(s_rec_l, s_prec_l, t_rec_l, t_prec_l, cd_equals_c, npos_orig[cls])
            
            if rec_and_prec[0][0] is not None:
                final_results['pp_l_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                final_results['pp_l_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            # calculate PR of c|l 
            t_p_r, t_p_p, t_rec_cl, t_prec_cl = voc_eval_c_given_l_mAP(transformed_human_results, original_human_results, st_p_gt, t_tp_recall, t_tp_precision, cls)
            
            rec_and_prec = estimate_curve_diff(s_rec_cl, s_prec_cl, t_rec_cl, t_prec_cl, t_tp_precision, npos_orig[cls])
            
            if rec_and_prec[0][0] is not None:
                final_results['pp_cl_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                final_results['pp_cl_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            # calculate mAP of d 
            t_rec_d = np.nan_to_num(np.multiply(t_rec_l, t_rec_cl))
            t_prec_d = np.nan_to_num(np.multiply(t_prec_l, t_prec_cl))

            rec_and_prec = estimate_curve_diff(s_rec_d, s_prec_d, t_rec_d, t_prec_d, cd_equals_c, npos_orig[cls])
            
            if rec_and_prec[0][0] is not None:
                final_results['pp_d_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                final_results['pp_d_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            ap_t = np.nan_to_num(voc_eval_d_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl))


            t_s_tp, t_rec_scl, t_prec_scl = voc_eval_s_given_cl_mAP(transformed_human_results, original_human_results, st_p_gt, t_p_p, cls)
            rec_and_prec = estimate_curve_diff(s_rec_scl, s_prec_scl, t_rec_scl, t_prec_scl, t_p_p, npos_orig[cls])
            if rec_and_prec[0][0] is not None:
                final_results['pp_scl_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                final_results['pp_scl_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

            ap_t = np.nan_to_num(voc_eval_d_mAP(t_rec_l, t_prec_l, t_rec_cl, t_prec_cl))

            t_rec_s = np.nan_to_num(np.multiply(t_rec_d, t_rec_scl))
            t_prec_s = np.nan_to_num(np.multiply(t_prec_d, t_prec_scl))

            rec_and_prec = estimate_curve_diff(s_rec_s, s_prec_s, t_rec_s, t_rec_s, cd_equals_c, npos_orig[cls])
            
            if rec_and_prec[0][0] is not None:
                final_results['pp_s_results'].append((cls, cur_IQA, 'rec', rec_and_prec[0]))
            if rec_and_prec[1][0] is not None:
                final_results['pp_s_results'].append((cls, cur_IQA, 'prec', rec_and_prec[1]))

        cur_IQA =  cur_IQA + IQA_interval
    print('number of images in each IQA interval:')
    print(IQA_to_img)
    print('number of boxes in each interval (original, transformed, gt)')
    print(IQA_to_box)

    STATS_THRES = 0.05
    for cls in [CLS]:
        for req in ['cp', 'pp']:
            for task in ['l', 'cl', 'd', 'scl', 's']:
                cur_list = req + '_' + task + '_results'
                print(cur_list + ':')
                prec_and_rec = []
                for func in ['rec', 'prec']:
                    IQAs = [i[1] for i in final_results[cur_list] if i[0] == cls and i[2] == func]
                    probs = [i[3][0] for i in final_results[cur_list] if i[0] == cls and i[2] == func]
                    orig_prob = [i[3][1] for i in final_results[cur_list] if i[0] == cls and i[2] == func]
                    num_points = [i[3][2] for i in final_results[cur_list] if i[0] == cls and i[2] == func]
                    p_values = [round(stats.binom_test(int(probs[i]*num_points[i]), n=num_points[i], p=orig_prob[i], alternative='less'), 2) for i in range(len(IQAs))]
                    if len(IQAs) >= 2:
                        prev = 1
                        index = len(IQAs)-1
                        for i in range(len(IQAs)):
                            cur = p_values[i]
                            if cur <= STATS_THRES and prev <= STATS_THRES:
                                index = i-1
                                break
                            prev = cur                
                        
                        prec_and_rec.append(IQAs[index])
                    else:
                        prec_and_rec.append(1-IQA_interval)
                if len(prec_and_rec) > 0:
                    print(min(prec_and_rec))

    