# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument('ffdm_output', type=str, help='destination file name')
    parser.add_argument('us_output', type=str, help='destination file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.ffdm_output.endswith('.pth')
    assert args.us_output.endswith('.pth')
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author='MMSelfSup')
    has_target = False
    for key, value in ck['state_dict'].items():
        if key.startswith('locality_atten'):
            output_dict['state_dict'][key] = value
            has_target = True
        if key.startswith('ffdm_backbone'):
            output_dict['state_dict'][key[len('ffdm')+1:]] = value
            has_target = True
    if not has_target:
        raise Exception('Cannot find locality_atten module in the checkpoint.')
    torch.save(output_dict, args.ffdm_output)

    output_dict = dict(state_dict=dict(), author='MMSelfSup')
    has_target = False
    for key, value in ck['state_dict'].items():
        if key.startswith('mil_attn'):
            output_dict['state_dict'][key] = value
            has_target = True
        if key.startswith('us_backbone'):
            output_dict['state_dict'][key[len('us')+1:]] = value
            has_target = True
    if not has_target:
        raise Exception('Cannot find locality_atten module in the checkpoint.')
    torch.save(output_dict, args.us_output)


if __name__ == '__main__':
    main()
