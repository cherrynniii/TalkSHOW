import os
import sys
 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())
 
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from evaluation.metrics import LVD
import numpy as np
import smplx as smpl
 
from nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig
from data_utils.get_j import get_joints
from data_utils.lower_body import part2full
 
import torch
from torch.utils import data
 
def init_model(model_name, model_path, args, config):
    generator = eval(model_name)(args, config)
    model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
 
    if 'generator' in model_ckpt:
        generator.load_state_dict(model_ckpt['generator'])
    else:
        generator.load_state_dict(model_ckpt)
 
    return generator
 
def init_dataloader(data_root, speakers, args, config):
    data_base = torch_data(
        data_root=data_root,
        speakers=speakers,
        split='test',
        limbscaling=False,
        normalization=config.Data.pose.normalization,
        norm_method=config.Data.pose.norm_method,
        split_trans_zero=False,
        num_pre_frames=config.Data.pose.pre_pose_length,
        num_generate_length=config.Data.pose.generate_length,
        num_frames=30,
        aud_feat_win_size=config.Data.aud.aud_feat_win_size,
        aud_feat_dim=config.Data.aud.aud_feat_dim,
        feat_method=config.Data.aud.feat_method,
        smplx=True,
        audio_sr=22000,
        convert_to_6d=config.Data.pose.convert_to_6d,
        expression=config.Data.pose.expression,
        config=config
    )
 
    if config.Data.pose.normalization:
        norm_stats_fn = os.path.join(os.path.dirname(args.model_path), "norm_stats.npy")
        norm_stats = np.load(norm_stats_fn, allow_pickle=True)
        data_base.data_mean = norm_stats[0]
        data_base.data_std = norm_stats[1]
    else:
        norm_stats = None
 
    data_base.get_dataset()
    test_loader = data.DataLoader(data_base.all_dataset, batch_size=1, shuffle=False)
 
    return test_loader, norm_stats
 
def compute_loss(gt_joints, pred_joints):
    loss_dict = {}
 
    # LVD
    v_diff = LVD(gt_joints[:, :22, :], pred_joints[:, :22, :], symmetrical=False, weight=False)
    loss_dict['LVD'] = v_diff
 
    # Joint-wise L2 error
    error = (gt_joints - pred_joints).norm(p=2, dim=-1).sum(dim=-1).mean()
    loss_dict['L2_error'] = error
 
    return loss_dict
 
def test(test_loader, generator, smplx_model, args, config):
    print('Start testing...')
    am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
    am_sr = 16000
 
    loss_dict = {}
    with torch.no_grad():
        count = 0
        for bat in tqdm(test_loader, desc="Testing"):
            count += 1
            aud, poses, exp = bat['aud_feat'].to('cuda').to(torch.float32), \
                               bat['poses'].to('cuda').to(torch.float32), \
                               bat['expression'].to('cuda').to(torch.float32)
            id = bat['speaker'].to('cuda') - 20
            betas = bat['betas'][0].to('cuda').to(torch.float64)
            gt_pose = torch.cat([poses, exp], dim=-2).transpose(-1, -2).squeeze()
 
            cur_wav_file = bat['aud_file'][0]
 
            pred = generator.infer_on_audio(
                cur_wav_file,
                initial_pose=None,
                norm_stats=None,
                w_pre=False,
                id=id,
                frame=gt_pose.shape[0],
                am=am,
                am_sr=am_sr
            )
            pred = torch.tensor(pred).to('cuda').squeeze()
 
            if config.Data.pose.convert_to_6d:
                pred_jaw = pred[:, :6].reshape(pred.shape[0], -1, 6)
                from data_utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
                pred_jaw = matrix_to_axis_angle(rotation_6d_to_matrix(pred_jaw)).reshape(pred.shape[0], -1)
                pred = torch.cat([pred_jaw, pred[:, 6:]], dim=-1)
 
            full_pred = part2full(pred, stand=args.stand)
            gt_pose = part2full(gt_pose, stand=args.stand)
 
            gt_joints = get_joints(smplx_model, betas, gt_pose)
            pred_joints = get_joints(smplx_model, betas, full_pred)
 
            bat_loss = compute_loss(gt_joints, pred_joints)
 
            if loss_dict:
                for k in bat_loss:
                    loss_dict[k] += bat_loss[k]
            else:
                loss_dict = bat_loss
 
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / count
            print(f"{k} = {loss_dict[k].item():.4f}")
 
def main():
    parser = parse_args()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)
 
    config = load_JsonConfig(args.config_file)
 
    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path
 
    print('init dataloader...')
    test_loader, norm_stats = init_dataloader(config.Data.data_root, args.speakers, args, config)
 
    print('init model...')
    generator = init_model(args.model_name, args.model_path, args, config)
 
    print('init SMPL-X model...')
    smplx_model = smpl.create(
        model_path='./visualise/',
        model_type='smplx',
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        num_betas=300,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        use_pca=False,
        flat_hand_mean=False,
        create_expression=True,
        num_expression_coeffs=100,
        num_pca_comps=12,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        dtype=torch.float64
    ).to('cuda')
 
    test(test_loader, generator, smplx_model, args, config)
 
if __name__ == '__main__':
    main()