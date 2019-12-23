import os
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval




def _log_detection_eval_metrics(coco, coco_eval):
    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    
    # import pdb; pdb.set_  trace()
    # IoU_lo_thresh = 0.5
    # IoU_hi_thresh = 0.95  # BORJI changed from 0.5 to 0.95
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95  # BORJI changed from 0.5 to 0.95

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(
        '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(
            IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    


    # for cls_ind in range(1, len(coco.cats)+1):
    #     # minus 1 because of __background__
    #     precision = coco_eval.eval['precision'][
    #         ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
    #     ap = np.mean(precision[precision > -1])
    #     print('{}:{:.1f}'.format(coco.cats[cls_ind]['name'], 100 * ap))
    # print('~~~~ Summary metrics ~~~~')
    # coco_eval.summarize()


    for cls_ind, cat_ind in enumerate(coco.cats, 1): #range(1, len(coco.cats)+1):
        # minus 1 because of __background__
        precision = coco_eval.eval['precision'][
            ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        ap = np.mean(precision[precision > -1])
        print('{}:{:.1f}'.format(coco.cats[cat_ind]['name'], 100 * ap))
    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()



    # # BORJI   % only for COCO dataset
    # valid_ids = [
    #   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
    #   14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
    #   24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
    #   37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
    #   48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
    #   58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
    #   72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
    #   82, 84, 85, 86, 87, 88, 89, 90]
    # cat_ids = {v: i for i, v in enumerate(valid_ids)}

    # for cls_ind in cat_ids:
    #     # minus 1 because of __background__
    #     precision = coco_eval.eval['precision'][
    #         ind_lo:(ind_hi + 1), :, cls_ind, 0, 2]
    #     ap = np.mean(precision[precision > -1])
    #     print('{}:{:.1f}'.format(coco.cats[cat_ids[cls_ind]]['name'], 100 * ap))
    # print('~~~~ Summary metrics ~~~~')
    # coco_eval.summarize()
    # # BORJI



if __name__ == "__main__":

    # coco = COCO("datasets/markable/annotations/markable_test.json")
    # coco_det = coco.loadRes("output/inference/markable_test/bbox.json")

    # coco = COCO("datasets/markable/annotations/markable_val.json")
    # coco_det = coco.loadRes("output/inference/markable_val/bbox.json")

    # coco = COCO("../data/markable/annotations/markable_val.json")
    # coco_det = coco.loadRes("../exp/ctdet/markable_dla/results.json")



    # MARKABEL
    # coco = COCO("./datasets/markable/annotations/markable_test.json")
    # coco_det = coco.loadRes("/home/ali/CenterNet/exp/ctdet/markable_dla/results.json")  # centerNet
    # coco_det = coco.loadRes("./results_ub/markable/markable_ub_new.json")
    # coco_det = coco.loadRes("./output/markable/out_fcos/bbox.json")                      # FCOS

    # coco = COCO("/home/ali/markable-cv-pytorch-detectron/datasets/markable/annotations/markable_test_seg.json")
    # coco_det = coco.loadRes("/home/ali/markable-cv-pytorch-detectron/output/inference/markable_test/bbox.json")                    # Mask RCNN    

    # coco_det = coco.loadRes("./results_ub/markable/markable_ub_extended_new_freq.json")    
    # coco_det = coco.loadRes("./results_ub/markable/markable_ub_extended_new_confidence.json")    
    # coco_det = coco.loadRes("./results_ub/markable/markable_ub_extended_new_freq_orig_classifier.json")    
    # coco_det = coco.loadRes("./results_ub/markable/markable_ub_extended_new_confidence_orig_classifier.json")    




    # PASCAL
    coco = COCO("./datasets/voc/annotations/pascal_test2007.json")
    # coco_det = coco.loadRes("/home/ali/markable-production/object_detection_context/outputs_json/voc_coco_format/faster_rcnn_pascal_cocostyle.json")    
    # coco_det = coco.loadRes("/home/ali/markable-production/object_detection_context/outputs_json/voc_coco_format/fcos_pascal_cocostyle.json")    
    # coco_det = coco.loadRes("/home/ali/markable-production/object_detection_context/outputs_json/voc_coco_format/centernet_pascal_cocostyle.json")            
    # coco_det = coco.loadRes("/home/ali/CenterNet/exp/ctdet/pascal_dla_384/results.json")
    # coco_det = coco.loadRes("/home/ali/CenterNet/exp/ctdet/dla/results.json")
    # coco_det = coco.loadRes("./results_ub/voc/voc_ub.json")
    coco_det = coco.loadRes("./results_ub/voc/voc_ub_new.json")
    # coco_det = coco.loadRes("/home/ali/Downloads/fcos_pacal_remov_cls_error.json")        
    # coco_det = coco.loadRes("./results_ub/voc/voc_ub_extended.json")
    # coco_det = coco.loadRes("./results_ub/voc/voc_ub_extended_new_freq.json")    
    # coco_det = coco.loadRes("./results_ub/voc/voc_ub_extended_new_confidence.json")        
    # # coco_det = coco.loadRes("./results_ub/voc/voc_ub_extended_new_freq_orig_classifier.json")    
    # coco_det = coco.loadRes("./results_ub/voc/voc_ub_extended_new_confidence_orig_classifier.json")        

    # invariance on VOC
    # coco = COCO("./datasets/coco/annotations/instances_val2017_invariance_origBG.json")
    # coco_det = coco.loadRes("./outputs_invariance/fcos_orig.bbox.json")






    # COCO
    # coco = COCO("./datasets/coco/annotations/instances_val2017.json")
    # coco_det = coco.loadRes("./results_ub/coco/coco_ub_new.json")     # new json with ResNet152
    # coco_det = coco.loadRes("./results_ub/coco/coco_ub_extended_new_confidence_orig_classifier.json")
    # coco_det = coco.loadRes("./results_ub/coco/coco_ub_extended_new_freq_orig_classifier.json")    
    # /exp/ctdet/markable_dla/results.json    

    # invariance on COCO
    # coco = COCO("./datasets/coco/annotations/instances_val2017_invariance_crop_origBG.json")
    # coco_det = coco.loadRes("./outputs_invariance/coco/fcos_crop_orig.bbox.json")
    # coco_det = coco.loadRes("./outputs_invariance/coco/crop_MaskRCNN.bbox.json")    

    # coco_det = coco.loadRes("/home/ali/Downloads/score_last_nmsmiss_firsthalf_loc_first_FCOS.json")



    # # OPENIMAGES
    # coco = COCO("./datasets/openImages/detection_annotation/validation_new.json")
    # coco_det = coco.loadRes("./results_ub/openImages/openImages_ub.json")



    coco_eval = COCOeval(coco, coco_det, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    #### coco_eval.params.iouThrs = np.array([0.5]*len(coco_eval.params.iouThrs))
    _log_detection_eval_metrics(coco, coco_eval)




    # # BORJI borrowd from facebook detectron2
    # precision = coco_eval.eval['precision']
    # results_per_category = []
    # for idx, name in enumerate(coco.cats):
    #     # area range index 0: all area ranges
    #     # max dets index -1: typically 100 per image
    #     precision = precisions[:, :, idx, 0, -1]
    #     precision = precision[precision > -1]
    #     ap = np.mean(precision) if precision.size else float("nan")
    #     results_per_category.append(("{}".format(name), float(ap * 100)))    

    # print('~~~~ Mean and per-category AP @ IoU=0.5 to 0.95 ~~~~')
    # print(results_per_category)    

    # print('~~~~ Summary metrics ~~~~')
    # coco_eval.summarize()

