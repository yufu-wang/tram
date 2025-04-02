import torch
import logging
from tqdm import tqdm
from lib.core.base_trainer import BaseTrainer
from lib.utils.pose_utils import Evaluator

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):

    def _init_fn(self):
        return

    def train_one_epoch(self, ):
        self.model.train()
        self.model.freeze_modules()
        update_iter = self.cfg.TRAIN.UPDATE_ITER
        crop_size = self.model.crop_size


        for i, batch in enumerate(tqdm(self.train_loader, desc="Computing batch")):

            # Transfer to GPU
            # batch = self.train_loader.batch_normalize_img(batch)
            batch = {k: v.to(self.device).flatten(0, 1) for k, v in batch.items() if type(v)==torch.Tensor}
            batch['beta_weight'] = self.cfg.TRAIN.SMPL_BETA
            batch['smpl'] = self.model.smpl

            # Forward pass
            out, iter_preds = self.model(batch, iters=update_iter)
            try:
                batch['pred_rotmat_0'] = out['pred_rotmat_0']
            except Exception:
                batch['pred_rotmat_0'] = None

            # Loss on full sequence
            rotmat_preds, shape_preds, cam_preds, j3d_preds, j2d_preds = iter_preds
            N = len(rotmat_preds)
            
            gamma = self.cfg.TRAIN.GAMMA
            loss = 0
            for j in range(N):
                batch['pred_rotmat'] = rotmat_preds[j]
                batch['pred_betas'] = shape_preds[j]
                batch['pred_cam'] = cam_preds[j]
                batch['pred_keypoints_3d'] = j3d_preds[j]
                batch['pred_keypoints_2d'] = (j2d_preds[j]-crop_size/2.) / (crop_size/2.) 

                loss_j, losses = self.criterion(batch)
                loss += gamma**(N-j-1) * loss_j
                
            loss *= self.cfg.TRAIN.LOSS_SCALE

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.cfg.TRAIN.CLIP_GRADIENT == True:
                self.clip_gradient_norm(self.model, max_norm=self.cfg.TRAIN.CLIP_NORM)

            self.optimizer.step()
            
            self.global_step += 1
            self.loss_meter.update(losses)
            self.lr_scheduler.step()

            self.check_and_validate(i)

            if self.should_break():
                break

        return 
        

    def check_and_validate(self, batch_id):
        steps = self.global_step

        # Training summary
        if steps % self.cfg.TRAIN.SUMMARY_STEP == 0:
            self.upload_losses(step = steps)
            self.upload_additional(step = steps)
            self.writer.flush()

        # Validation summary
        if steps % self.cfg.TRAIN.VALID_STEP == 0:
            if steps < 5000:
                save_best = False
            else:
                save_best = True
            
            performance = self.validate()
            self.check_performance(performance, batch=batch_id, save_best=save_best)
        else:
            performance = None

        
        # Checkpoint
        if steps % self.cfg.TRAIN.SAVE_STEP == 0:
            self.save_checkpoint(batch=batch_id, performance=performance,
                                index='_{:04d}'.format(steps))
            

    def validate(self,):
        logger.info(f"Epoch {self.epoch}, Step {self.global_step}, validating ...")
        torch.cuda.empty_cache() 

        self.model.eval()
        update_iter = self.cfg.TRAIN.UPDATE_ITER

        model = self.model
        loader = self.test_loader
        device = self.device
        db = loader.dataset

        evaluator = Evaluator(dataset_length=len(db.imgname),
                              seq_len=getattr(model, 'seq_len', None))
        J_regressor = db.J_regressor.to(device)

        for i, batch in enumerate(loader):
            # batch = loader.batch_normalize_img(batch)
            batch = {k: v.to(self.device).flatten(0, 1) for k, v in batch.items() if type(v)==torch.Tensor}

            # gt joints
            gt_keypoints_3d = batch['pose_3d']

            # prediction
            with torch.no_grad():
                out, _ = model(batch, iters=update_iter)
                
                if '3dpw' in db.dataset:
                    mode = '3dpw'
                    smpl_out = model.smpl.query(out)
                    pred_vertices = smpl_out.vertices
                    J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)

                    pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
                    pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
                    pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

                elif 'emdb' in db.dataset:
                    mode = 'emdb'
                    smpl_out = model.smpl.query(out, default_smpl=True)
                    pred_keypoints_3d = smpl_out.joints[:, :24]

                    pred_pelvis = pred_keypoints_3d[:,[1,2],:].mean(dim=1, keepdim=True).clone()
                    pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
                    

            # evaluation
            evaluator(gt_keypoints_3d, pred_keypoints_3d, mode)


        re = evaluator.re[:evaluator.counter].mean()
        mpjpe = evaluator.mpjpe[:evaluator.counter].mean()
        acc = evaluator.acc[:evaluator.counter].mean()

        logger.info(f"Epoch {self.epoch}, Step {self.global_step}, validation re: {re}")
        logger.info(f"Epoch {self.epoch}, Step {self.global_step}, validation mpjpe: {mpjpe}")
        logger.info(f"Epoch {self.epoch}, Step {self.global_step}, validation accel: {acc}")

        self.writer.add_scalar(f"Validation/RE", re, self.global_step)
        self.writer.add_scalar(f"Validation/MPJPE", mpjpe, self.global_step)
        self.writer.add_scalar(f"Validation/ACCEL", acc, self.global_step)
        self.writer.flush()

        self.model.train()
        self.model.freeze_modules()

        self.performance_type = 'min'

        torch.cuda.empty_cache() 

        return re

    def upload_additional(self, step):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("Z/lr", lr, step)
        
        return


    

