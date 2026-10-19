#!/usr/bin/env python3
"""DRA-SAM training, prediction and evaluation."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def absolute(value):
    return str(Path(value).expanduser().resolve())


def run(command, env, cwd, dry_run):
    print(json.dumps({'cwd': str(cwd), 'command': shlex.join(command),
                      'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES']}, indent=2), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='task', required=True)

    def common(name, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--gpu', type=int, default=0, help='Physical CUDA device exposed as cuda:0')
        p.add_argument('--dry-run', action='store_true')
        return p

    p = common('train-backbone', 'Fine-tune SAM3-I backbone stages')
    p.add_argument('--stage', choices=['stage1_2', 'stage1_2_continue', 'stage3'], required=True)
    p.add_argument('--annotations', required=True, help='Folder containing sam3i_train/val.json')
    p.add_argument('--images', required=True)
    p.add_argument('--init', help='Initial checkpoint; Stage3 defaults to the trained Stage1_2 checkpoint')

    p = common('global', 'Generate candidate predictions and segmentation metrics')
    p.add_argument('--ann', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--sam-checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=8)

    p = common('cache', 'Cache training observations and create the internal validation split')
    p.add_argument('--annotations', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--sam-checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=8)

    p = common('train-evidence', 'Train the lightweight EvidenceScorer from cached observations')
    p.add_argument('--train-cache', required=True, help='Path to cache/train/records')
    p.add_argument('--val-cache', required=True, help='Path to cache/val/records (internal validation)')
    p.add_argument('--out', required=True)
    p.add_argument('--config', help='Override the dataset EvidenceScorer YAML')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--resume')

    p = common('evaluate', 'Evaluate DRA-SAM segmentation predictions')
    p.add_argument('--ann', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--sam-checkpoint', required=True)
    p.add_argument('--evidence-checkpoint', required=True)
    p.add_argument('--baseline', required=True, help='Global inference global_all.json')
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--tau', type=float)
    p.add_argument('--unknown-weight', type=float)
    p.add_argument('--save-masks', action='store_true')

    args = parser.parse_args()
    dataset = 'rislad'
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    if hasattr(args, 'batch_size') and args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.task == 'evaluate' and args.tau is not None and args.tau <= 0:
        parser.error('temperature must be positive')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               PYTHONUNBUFFERED='1', DRA_ROOT=str(ROOT))
    # Load the model runtime from this repository.
    paths = [str(ROOT / 'sam3'), str(ROOT / 'tools'), str(ROOT / 'scripts')]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    py = sys.executable

    def execute(command, cwd=ROOT):
        run(command, env, cwd, args.dry_run)

    if args.task == 'train-backbone':
        env.update(DRA_ANNOTATIONS=absolute(args.annotations), DRA_IMAGES=absolute(args.images))
        # Select the initialization checkpoint for this training stage.
        env.pop('DRA_INIT_CHECKPOINT', None)
        if args.init:
            env['DRA_INIT_CHECKPOINT'] = absolute(args.init)
        execute([py, 'sam3/train/train.py', '-c',
                 f'configs/dra_sam/{dataset}_{args.stage}', '--use-cluster', '0',
                 '--num-nodes', '1', '--num-gpus', '1'], ROOT / 'sam3')
        if args.stage == 'stage1_2':
            env.pop('DRA_INIT_CHECKPOINT', None)
            execute([py, 'sam3/train/train.py', '-c',
                     'configs/dra_sam/rislad_stage1_2_continue', '--use-cluster', '0',
                     '--num-nodes', '1', '--num-gpus', '1'], ROOT / 'sam3')

    elif args.task == 'global':
        out = Path(absolute(args.out))
        if not args.dry_run:
            out.mkdir(parents=True, exist_ok=True)
        pred = str(out / 'global_all.json')
        execute([py, 'scripts/inference.py', '--json_path', absolute(args.ann),
                 '--image_root', absolute(args.images), '--checkpoint_path', absolute(args.sam_checkpoint),
                 '--output_path', pred, '--batch_size', str(args.batch_size), '--num_workers', '0',
                 '--gpus', '0', '--categories', 'complex', '--detection_threshold', '0.5'])
        execute([py, 'tools/metrics.py', '--gt', absolute(args.ann),
                 '--pred', pred, '--out', str(out / 'metrics.json')])

    elif args.task == 'cache':
        out = Path(absolute(args.out))
        internal = out / 'internal_split'
        execute([py, 'tools/prepare_internal_split.py', '--ann',
                 str(Path(absolute(args.annotations)) / 'sam3i_train.json'),
                 '--out', str(internal)])
        for split in ('train', 'val'):
            execute([py, 'tools/build_cache.py', '--ann', str(internal / f'sam3i_internal_{split}.json'),
                     '--image-root', absolute(args.images), '--checkpoint', absolute(args.sam_checkpoint),
                     '--out', str(out / split), '--device', '0', '--batch-size', str(args.batch_size),
                     '--num-workers', '2', '--scales', '0.30,0.45,0.60,0.75',
                     '--random-crops', '2', '--threshold', '0.5', '--feature-grid', '24', '--resume'])

    elif args.task == 'train-evidence':
        import yaml
        config_file = Path(args.config).resolve() if args.config else ROOT / 'configs' / f'{dataset}.yaml'
        cfg = yaml.safe_load(config_file.read_text())
        out = Path(absolute(args.out))
        cfg['paths'] = {'project_root': str(ROOT), 'output_dir': str(out)}
        cfg['train']['batch_size'] = args.batch_size
        cfg['runtime']['device'] = 'cuda:0'
        config_path = out / 'config.yaml'
        if args.dry_run:
            print(json.dumps({'generated_config': str(config_path), 'configuration': cfg}, indent=2))
        else:
            out.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        command = [py, 'tools/train_evidence.py', '--config', str(config_path),
                   '--train-cache', absolute(args.train_cache), '--val-cache', absolute(args.val_cache)]
        if args.resume:
            command += ['--resume', absolute(args.resume)]
        execute(command)

    elif args.task == 'evaluate':
        import yaml
        defaults = yaml.safe_load((ROOT / 'configs' / f'{dataset}.yaml').read_text())['inference']
        tau = defaults['posterior_temperature'] if args.tau is None else args.tau
        uw = defaults['unknown_weight'] if args.unknown_weight is None else args.unknown_weight
        command = [py, 'tools/evaluate.py',
                   '--sam-checkpoint', absolute(args.sam_checkpoint),
                   '--checkpoint', absolute(args.evidence_checkpoint),
                   '--image-root', absolute(args.images), '--ann', absolute(args.ann),
                   '--baseline', absolute(args.baseline), '--subset-name', dataset,
                   '--device', '0', '--batch-size', str(args.batch_size),
                   '--unknown-weight', str(uw),
                   '--posterior-temperature', str(tau), '--out', absolute(args.out)]
        if args.save_masks:
            command.append('--save-visual-masks')
        execute(command)


if __name__ == '__main__':
    main()
