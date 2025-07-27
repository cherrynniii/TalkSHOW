import os
import sys
 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())
 
from transformers import Wav2Vec2Processor
import numpy as np
import smplx as smpl
 
from nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig
 
import torch
import torch.nn.functional as F
from torch.utils import data
from data_utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
from data_utils.lower_body import part2full
from visualise.rendering import RenderTool
 
 
global device
device = 'cpu'
 
def init_model(model_name, model_path, args, config):
    generator = eval(model_name)(args, config)
    model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
 
    if 'generator' in model_ckpt:
        generator.load_state_dict(model_ckpt['generator'])
    else:
        generator.load_state_dict(model_ckpt)
 
    return generator
 
def get_vertices(smplx_model, betas, result_list, exp):
    vertices_list = []
    expression = torch.zeros([1, 50])
 
    for i in result_list:
        vertices = []
        for j in range(i.shape[0]):
            output = smplx_model(
                betas=betas,
                expression=i[j][165:265].unsqueeze(0) if exp else expression,
                jaw_pose=i[j][0:3].unsqueeze(0),
                leye_pose=i[j][3:6].unsqueeze(0),
                reye_pose=i[j][6:9].unsqueeze(0),
                global_orient=i[j][9:12].unsqueeze(0),
                body_pose=i[j][12:75].unsqueeze(0),
                left_hand_pose=i[j][75:120].unsqueeze(0),
                right_hand_pose=i[j][120:165].unsqueeze(0),
                return_verts=True
            )
            vertices.append(output.vertices.detach().cpu().numpy().squeeze())
        vertices = np.asarray(vertices)
        vertices_list.append(vertices)
 
    return vertices_list
 
def infer(g_model, smplx_model, rendertool, config, args):
    betas = torch.zeros([1, 300], dtype=torch.float64).to(device)
    am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
    am_sr = 16000
    cur_wav_file = args.audio_file
    id = torch.tensor([args.id], device=device)
    face = args.only_face
    stand = args.stand
 
    if face:
        body_static = torch.zeros([1, 162], device=device)
        body_static[:, 6:9] = torch.tensor([3.0747, -0.0158, -0.0152])
 
    pred = g_model.infer_on_audio(
        cur_wav_file,
        initial_pose=None,
        norm_stats=None,
        w_pre=False,
        id=id,
        frame=None,
        am=am,
        am_sr=am_sr
    )
    pred = torch.tensor(pred).squeeze().to(device)
 
    if config.Data.pose.convert_to_6d:
        pred_jaw = pred[:, :6].reshape(pred.shape[0], -1, 6)
        pred_jaw = matrix_to_axis_angle(rotation_6d_to_matrix(pred_jaw)).reshape(pred.shape[0], -1)
        pred = torch.cat([pred_jaw, pred[:, 6:]], dim=-1)
 
    pred = part2full(pred, stand)
 
    if face:
        pred = torch.cat([pred[:, :3], body_static.repeat(pred.shape[0], 1), pred[:, -100:]], dim=-1)
 
    vertices_list = get_vertices(smplx_model, betas, [pred], config.Data.pose.expression)
    np.save(f'visualise/video/{config.Log.name}/{os.path.basename(cur_wav_file).split(".")[0]}', pred.cpu().numpy())
    rendertool._render_sequences(cur_wav_file, vertices_list, stand=stand, face=face, whole_body=args.whole_body)
 
def main():
    parser = parse_args()
    args = parser.parse_args()
    config = load_JsonConfig(args.config_file)
 
    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path
 
    print('init model...')
    generator = init_model(args.model_name, args.model_path, args, config)
 
    print('init smplx model...')
    model_params = dict(
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
        dtype=torch.float64,
    )
    smplx_model = smpl.create(**model_params).to(device)
 
    print('init rendertool...')
    rendertool = RenderTool('visualise/video/' + config.Log.name)
    infer(generator, smplx_model, rendertool, config, args)
 
if __name__ == '__main__':
    main()