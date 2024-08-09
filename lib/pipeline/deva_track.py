import sys
sys.path.insert(0, 'thirdparty/Tracking-Anything-with-DEVA')

from os import path
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import numpy as np

from deva.model.network import DEVA
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_auto_default_args

from deva.inference.object_info import ObjectInfo
from deva.inference.frame_utils import FrameInfo
from deva.inference.demo_utils import get_input_frame_for_deva

# Some DEVA tracking settings
args = ['--chunk_size', '4', '--amp', '--temporal_setting', 'semionline',
        '--size', '480', '--model', 'data/pretrain/DEVA-propagation.pth',
        '--suppress_small_objects', '--max_long_term_elements', '1000', '--max_num_objects', '50',
        '--detection_every', '5']

parser = ArgumentParser()
add_common_eval_args(parser)
add_ext_eval_args(parser)
add_auto_default_args(parser)

args = parser.parse_args(args)
cfg = vars(args)
cfg['enable_long_term'] = not cfg['disable_long_term']

deva_model = DEVA(cfg).cuda().eval()
model_weights = torch.load(args.model)
_ = deva_model.load_weights(model_weights)


def get_deva_tracker(vid_length, out_path):
    cfg['enable_long_term_count_usage'] = (  # default deva long-term handling
        cfg['enable_long_term']
        and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
                cfg['num_prototypes']) >= cfg['max_long_term_elements'])

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = args.num_voting_frames - 1
    deva.enabled_long_id()
    result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)
    result_saver.json_style = 'burst'
    result_saver.visualize = False

    return deva, result_saver


@torch.inference_mode()
def track_with_mask(deva: DEVAInferenceCore,
                    masks: torch.Tensor,
                    scores: torch.Tensor,
                    image_np: np.ndarray,  #RGB
                    frame_path: str,
                    result_saver: ResultSaver,
                    ti: int,
                    save_vos=True) -> None:
    """ Standard DEVA tracking step"""

    cfg = deva.config
    save_the_mask = save_vos

    h, w = image_np.shape[:2]
    min_side = cfg['size']
    need_resize = min_side > 0
    image = get_input_frame_for_deva(image_np, min_side)

    new_h, new_w = image.shape[1:]
    mask, segments_info = transform_masks(masks, scores, new_h, new_w)
    mask = mask.to('cuda')

    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })

    # Run Semi-online DEVA
    if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
        frame_info.mask = mask
        frame_info.segments_info = segments_info
        frame_info.image_np = image_np  # for visualization only
        deva.add_to_temporary_buffer(frame_info)  # wait for more frames 

        if ti == deva.next_voting_frame:
            # process this clip
            this_image = deva.frame_buffer[0].image
            this_frame_name = deva.frame_buffer[0].name
            this_image_np = deva.frame_buffer[0].image_np

            _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                keyframe_selection='first')
            prob = deva.incorporate_detection(this_image, mask, new_segments_info)
            deva.next_voting_frame += cfg['detection_every']

            result_saver.save_mask(prob,
                                   this_frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=this_image_np,
                                   save_the_mask=save_the_mask)

            for frame_info in deva.frame_buffer[1:]:
                this_image = frame_info.image
                this_frame_name = frame_info.name
                this_image_np = frame_info.image_np
                prob = deva.step(this_image, None, None)
                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np,
                                       save_the_mask=save_the_mask)
            deva.clear_buffer()
    else:
        # standard propagation
        prob = deva.step(image, None, None)
        result_saver.save_mask(prob,
                               frame_name,
                               need_resize=need_resize,
                               shape=(h, w),
                               image_np=image_np,
                               save_the_mask=save_the_mask)


def transform_masks(masks, scores, new_h, new_w):
    """ Convert masks to index-mask format for DEVA """
    area = masks.sum([1,2])
    device = masks.device

    output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=device)
    curr_id = 1
    segments_info = []

    # sort by descending area to preserve the smallest object
    for i in np.flip(np.argsort(area).tolist()):
        mask = masks[i]
        confidence = scores[i].item()
        mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), 
                             (new_h, new_w), 
                             mode='bilinear')[0, 0]
        mask = (mask > 0.5).float()

        if mask.sum() > 0:
            output_mask[mask > 0] = curr_id
            segments_info.append(ObjectInfo(id=curr_id, category_id=None, score=confidence))
            curr_id += 1

    return output_mask, segments_info