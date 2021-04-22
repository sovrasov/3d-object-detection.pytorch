import argparse

from torchdet3d.builders import build_model
from torchdet3d.utils import read_py_config
from ptflops import get_model_complexity_info


def main():
    parser = argparse.ArgumentParser(description='Estimating model complexity')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='path to configuration file')
    args = parser.parse_args()

    cfg = read_py_config(args.config)
    model = build_model(cfg, export_mode=True)
    model.eval()
    macs, params = get_model_complexity_info(model, (3, *cfg.data.resize),
                                             verbose=False, print_per_layer_stat=True)
    print(f'{"Input shape: ":<30}  {str((1, 3, *cfg.data.resize)):<8}')
    print(f'{"Computational complexity: ":<30}  {macs:<8}')
    print(f'{"Number of parameters: ":<30}  {params:<8}')

if __name__ == "__main__":
    main()
