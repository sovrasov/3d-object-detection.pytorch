from subprocess import run, DEVNULL, CalledProcessError
import argparse

import torch
import os

from torchdet3d.builders import build_model
from torchdet3d.utils import load_pretrained_weights, read_py_config


def export_onnx(model, snapshot_path, img_size=(128,128), save_path='model.onnx'):
    # input to inference model
    dummy_input = torch.rand(size=(1,3,*img_size))
    dummy_cat = torch.zeros(1, dtype=torch.long)
    # load checkpoint from config
    load_pretrained_weights(model, snapshot_path)
    # convert model to onnx
    input_names = ["data"]
    output_names = ["cls_bbox"]
    with torch.no_grad():
        model.eval()
        torch.onnx.export(model, args=dummy_input, f=save_path, verbose=True,
                      input_names=input_names, output_names=output_names)

def export_mo(onnx_model_path, mean_values, scale_values, save_path):
    command_line = (f'mo.py --input_model="{onnx_model_path}" '
                   f'--mean_values="{mean_values}" '
                   f'--scale_values="{scale_values}" '
                   f'--output_dir="{save_path}" '
                   f'--reverse_input_channels ')

    try:
        run('mo.py -h', stdout=DEVNULL, stderr=DEVNULL, shell=True, check=True)
    except CalledProcessError as _:
        print('OpenVINO Model Optimizer not found, please source '
            'openvino/bin/setupvars.sh before running this script.')
        return

    run(command_line, shell=True, check=True)

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='converting model to onnx/mo')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='path to configuration file')
    parser.add_argument('--model_onnx_path', type=str, default='./converted_models/model.onnx', required=False,
                        help='path where to save the model in onnx format')
    parser.add_argument('--model_torch_path', type=str, required=False,
                        help='path where to get the model in .pth format.'
                             'By default the model will be obtained from config, the lastest epoch')
    parser.add_argument('--model_mo_path', type=str, default='./converted_models', required=False,
                        help='path where to save the model in IR format')
    parser.add_argument('--convert_mo', action='store_true',
                        help='argument defines whether or not to convert to IR format')

    args = parser.parse_args()
    # read config
    cfg = read_py_config(args.config)
    if not args.model_torch_path:
        x = os.listdir(cfg.output_dir)
        snap=sorted(x, key=lambda z: int(z[5:-4]))[-1]
        snapshot_path = os.path.join(cfg.output_dir, snap)
    else:
        snapshot_path = args.model_torch_path
    model = build_model(cfg, export_mode=True)

    mean_values = str([s*255 for s in cfg.data.normalization.mean])
    scale_values = str([s*255 for s in cfg.data.normalization.std])
    export_onnx(model, snapshot_path, cfg.data.resize, args.model_onnx_path)
    if args.convert_mo:
        export_mo(args.model_onnx_path, mean_values, scale_values, args.model_mo_path)

if __name__ == "__main__":
    main()
