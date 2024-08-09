import cv2
from tqdm import tqdm
import numpy as np
import torch
import torchvision

from segment_anything import SamPredictor, sam_model_registry
from pycocotools import mask as masktool

from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
from lib.pipeline.deva_track import get_deva_tracker, track_with_mask, flush_buffer


if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def video2frames(vidfile, save_folder):
    """ Convert input video to images """
    count = 0
    cap = cv2.VideoCapture(vidfile)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imwrite(f'{save_folder}/{count:04d}.jpg', frame)
            count += 1
        else:
            break
    cap.release()
    return count


def detect_segment_track(imgfiles, out_path, thresh=0.5, min_size=None, 
                         device='cuda', save_vos=True):
    """ A simple pipeline for human detection, segmentation, and tracking. Mainly as input for TRAM.
    Detection: ViTDet.
    Segmentation: SAM.
    Tracking: DEVA-Track-Anything. 
    """
    # ViTDet
    cfg_path = 'data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # SAM
    sam = sam_model_registry["vit_h"](checkpoint="data/pretrain/sam_vit_h_4b8939.pth")
    _ = sam.to(device)
    predictor = SamPredictor(sam)

    # DEVA
    vid_length = len(imgfiles)
    deva, result_saver = get_deva_tracker(vid_length, out_path)

    # Run
    masks_ = []
    boxes_ = []
    for t, imgpath in enumerate(tqdm(imgfiles)):
        img_cv2 = cv2.imread(imgpath)

        ### --- Detection ---
        with torch.no_grad():
            with autocast():
                det_out = detector(img_cv2)
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > thresh)
                boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                confs = det_instances.scores[valid_idx].cpu().numpy()

                boxes = np.hstack([boxes, confs[:, None]])
                boxes = arrange_boxes(boxes, mode='size', min_size=min_size)

        ### --- SAM --- 
        if len(boxes)>0:
            with autocast():
                predictor.set_image(img_cv2, image_format='BGR')

                # multiple boxes
                bb = torch.tensor(boxes[:, :4]).cuda()
                bb = predictor.transform.apply_boxes_torch(bb, img_cv2.shape[:2])  
                masks, scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=bb,
                    multimask_output=False
                )
                scores = scores.cpu()
                masks = masks.cpu().squeeze(1)
                mask = masks.sum(dim=0)
        else:
            mask = np.zeros_like(mask)

        ### --- DEVA ---
        if len(boxes)>0 and (boxes[:, -1] > 0.80).sum()>0:
            track_valid = boxes[:, -1] > 0.80    # only use high-confident
            masks_track = masks[track_valid]
            scores_track = scores[track_valid]
        else:
            masks_track = torch.zeros([1,img_cv2.shape[0],img_cv2.shape[1]])
            scores_track = torch.zeros([1])

        with autocast():
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            track_with_mask(deva, masks_track, scores_track, img_rgb, 
                            imgpath, result_saver, t, save_vos)
            
        ### Record full mask and boxes
        mask_bit = masktool.encode(np.asfortranarray(mask > 0))
        masks_.append(mask_bit)
        boxes_.append(boxes)

    with autocast():
        flush_buffer(deva, result_saver)
    result_saver.end()

    ### --- Adapt tracks data structure ---
    vjson = result_saver.video_json
    ann = vjson['annotations']
    iou_thresh = 0.5
    conf_thresh = 0.5

    tracks = {}
    for frame in ann:         
        seg = frame['segmentations']
        file = frame['file_name']
        frame = int(file.split('.')[0])
        for subj in seg:  
            idx = subj['id']
            msk = subj['rle']
            msk = torch.from_numpy(masktool.decode(msk))[None]
            
            # match tracked segment to detections
            det_boxes = boxes_[frame]
            if len(det_boxes)>0:
                seg_box = torchvision.ops.masks_to_boxes(msk)
                iou = box_iou(det_boxes, seg_box)
                max_iou, max_id = iou.max(), iou.argmax()
                max_conf = det_boxes[max_id, -1]
            else:
                max_iou = max_conf = 0

            if max_iou>iou_thresh and max_conf>conf_thresh:
                det = True
                det_box = det_boxes[[max_id]]
            else:
                det = False
                det_box = np.zeros([1, 5])

            # add fields
            subj['frame'] = frame 
            subj['det'] = det
            subj['det_box'] = det_box
            subj['seg_box'] = seg_box.numpy()
            
            if idx in tracks:
                tracks[idx].append(subj)
            else:
                tracks[idx] = [subj]

    tracks = np.array(tracks, dtype=object)
    masks_ = np.array(masks_, dtype=object)
    boxes_ = np.array(boxes_, dtype=object)

    return boxes_, masks_, tracks


def parse_chunks(frame, boxes, min_len=16):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []
    boxes_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        b_chunk = boxes[start:bk]
        start = bk
        if len(f_chunk)>=min_len:
            frame_chunks.append(f_chunk)
            boxes_chunks.append(b_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            b_chunk = boxes[bk:]
            if len(f_chunk)>=min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)

    return frame_chunks, boxes_chunks


def arrange_boxes(boxes, mode='size', min_size=None):
    """ Helper to re-order boxes """
    # Left2right priority
    if mode == 'left2right':
        cx = (boxes[:,2] - boxes[:,0]) / 2 + boxes[:,0]
        boxes = boxes[np.argsort(cx)]
    # size priority
    elif mode == 'size':
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]
        area = w*h
        boxes = boxes[np.argsort(area)[::-1]]
    # confidence priority
    elif mode == 'conf':  
        conf = boxes[:,4]
        boxes = boxes[np.argsort(conf)[::-1]]
    # filter boxes by size
    if min_size is not None:
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]
        valid = np.stack([w, h]).max(axis=0) > min_size
        boxes = boxes[valid]

    return boxes


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    if type(box1) == np.ndarray:
        box1 = torch.from_numpy(box1)
    if type(box2) == np.ndarray:
        box2 = torch.from_numpy(box2)
    box1 = box1[:, :4]
    box2 = box2[:, :4]

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


