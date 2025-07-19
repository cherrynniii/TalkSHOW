import os
import sys
 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())
 
import numpy as np
import torch
import smplx as smpl
 
from transformers import Wav2Vec2Processor
from nets import *
from trainer.options import parse_args
from trainer.config import load_JsonConfig
from visualise.rendering import RenderTool
from data_utils.lower_body import part2full
from data_utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
 
device = 'cpu'
 
 
def init_model(model_name, model_path, args, config):
    if model_name == 's2g_simple':
        generator = s2g_simple(args, config)
    else:
        raise NotImplementedError(f"{model_name}은 이 스크립트에서는 지원되지 않습니다.")
 
    model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    if 'generator' in model_ckpt:
        generator.load_state_dict(model_ckpt['generator'])
    else:
        generator.load_state_dict({'generator': model_ckpt})
 
    return generator
 
 
def get_vertices(smplx_model, betas, result_list, exp):
    vertices_list = []
    expression = torch.zeros([1, 50])
    for seq in result_list:
        vertices = []
        for frame in seq:
            output = smplx_model(
                betas=betas,
                expression=frame[165:265].unsqueeze(0) if exp else expression,
                jaw_pose=frame[0:3].unsqueeze(0),
                leye_pose=frame[3:6].unsqueeze(0),
                reye_pose=frame[6:9].unsqueeze(0),
                global_orient=frame[9:12].unsqueeze(0),
                body_pose=frame[12:75].unsqueeze(0),
                left_hand_pose=frame[75:120].unsqueeze(0),
                right_hand_pose=frame[120:165].unsqueeze(0),
                return_verts=True
            )
            vertices.append(output.vertices.detach().cpu().numpy().squeeze())
        vertices_list.append(np.asarray(vertices))
    return vertices_list
 
 
def infer(generator, smplx_model, rendertool, config, args):
    betas = torch.zeros([1, 300], dtype=torch.float64).to(device)
    am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
    cur_wav_file = args.audio_file
    id = torch.tensor([args.id], device=device)
 
    result_list = []
 
    pred_res = generator.infer_on_audio(
        cur_wav_file,
        initial_pose=None,
        norm_stats=None,
        txgfile=None,
        id=id,
        var=None,
        fps=30,
        w_pre=False
    )
    pred = torch.tensor(pred_res).squeeze().to(device)
 
    if config.Data.pose.convert_to_6d:
        pred = pred.reshape(pred.shape[0], -1, 6)
        pred = matrix_to_axis_angle(rotation_6d_to_matrix(pred)).reshape(pred.shape[0], -1)
 
    pred = part2full(pred, args.stand)
    result_list.append(pred)
 
    vertices_list = get_vertices(smplx_model, betas, result_list, config.Data.pose.expression)
    dict = np.concatenate(result_list, axis=0)
    file_name = f'visualise/video/{config.Log.name}/{os.path.basename(cur_wav_file).split(".")[0]}'
    np.save(file_name, dict)
 
    rendertool._render_sequences(cur_wav_file, vertices_list, stand=args.stand, face=False, whole_body=args.whole_body)
 
 
def main():
    parser = parse_args()
    args = parser.parse_args()
    config = load_JsonConfig(args.config_file)
 
    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path
 
    print("init model...")
    generator = init_model(args.body_model_name, args.body_model_path, args, config)
 
    print("init smplx model...")
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
 
    print("init rendertool...")
    rendertool = RenderTool('visualise/video/' + config.Log.name)
 
    infer(generator, smplx_model, rendertool, config, args)
 
 
if __name__ == '__main__':
    main()