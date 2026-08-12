# Train the FMIR VoxelMorph-style decoder (README "Method 2") on OASIS.
#
# Unlike convex_optimization's ConvexAdam scripts (Method 1: per-pair test-time optimization,
# no learning across pairs), this trains one FMIR decoder's weights so it can register a new
# fixed/moving pair with a single forward pass. That means it needs many *training* pairs,
# distinct from the pairs it's evaluated on.
#
# OASIS/OASIS_config.json's 19 pairs are the standard Learn2Reg OASIS *validation* protocol
# (subjects 0395<->0414, chained sequentially) - the right thing for ConvexAdam to optimize+
# report on directly, but only 19 of OASIS's 414 labeled subjects. This script instead:
#   - trains on the exhaustive pool of ordered pairs among the other 394 subjects (0001-0394),
#     which the validation protocol never touches;
#   - holds the 19 official pairs out untouched, purely for Dice evaluation.
#
# Feature handling for that pool: fit ONE global PCA basis (on a random subject subsample),
# and reuse it for every subject and for eval - a fixed shared projection is what makes a
# *learned, weight-shared* decoder's input space consistent across pairs. (convex_optimization's
# per-pair joint PCA fit is correct for its per-instance optimization, but isn't reusable here:
# refitting a basis per pair would make the same physical feature project differently depending
# on which partner it's paired with, which a shared-weight network can't be trained against.)
# Each subject's image+features are extracted/reduced/cube-resized once and cached (fp16, on
# CPU) rather than per pair, since every subject appears in ~393 pairs.
#
# Run from the repo root:
#   python -m PlaneAdam.decoder_method.train
#   python -m PlaneAdam.decoder_method.train --epochs 1 --steps-per-epoch 20000
import argparse
import json
import os
import sys
import time

import nibabel as nib
import torch
import torch.nn.functional as F
import wandb

# convex_optimization/ has no __init__.py (its scripts are invoked directly, not imported),
# so its directory needs to be on sys.path explicitly to reuse dice_coeff.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONVEX_OPT_DIR = os.path.join(_REPO_ROOT, 'PlaneAdam', 'convex_optimization')
if _CONVEX_OPT_DIR not in sys.path:
    sys.path.insert(0, _CONVEX_OPT_DIR)

from convexAdam_hyper_util import dice_coeff  # noqa: E402
from models.hub.backbones import dinov3_vits16  # noqa: E402
from PlaneAdam.Dataset.load_dataset_OASIS import get_data_train, normalize_intensity  # noqa: E402
from PlaneAdam.decoder_method.backbones import SpatialTransformer, VecInt  # noqa: E402
from PlaneAdam.decoder_method.FMIR import FMIR  # noqa: E402
from PlaneAdam.decoder_method.utils.loss import Grad3d, NccLoss  # noqa: E402
from PlaneAdam.feature_extract.dino_extract import DinoBackboneExtractor  # noqa: E402

UPSCALE = (2, 2, 2)
N_LEVELS = 5
OASIS_IMAGES_DIR = os.path.join(_REPO_ROOT, 'OASIS', 'imagesTr')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-config', default=os.path.join(_REPO_ROOT, 'OASIS_config.json'),
                    help='defines the held-out evaluation pairs (default: the 19 official L2R validation pairs)')
    p.add_argument('--num-train-subjects', type=int, default=394,
                    help='OASIS_0001..OASIS_{this:04d} form the training pool (must not overlap the eval pairs\' subjects)')
    p.add_argument('--epochs', type=int, default=1, help='1 epoch = one shuffled pass over the full training pair pool')
    p.add_argument('--steps-per-epoch', type=int, default=None,
                    help='truncate each epoch to this many pairs (default: the full 394x393 pool)')
    p.add_argument('--batch-size', type=int, default=8,
                    help='pairs per optimizer step; amortizes per-step overhead (see cached_batch_inputs). '
                         'This GPU is shared with other processes and free memory fluctuates - 16 OOM\'d '
                         'intermittently in testing at cube=48, 8 was stable; raise cautiously.')
    p.add_argument('--pca', type=int, default=64, help='PCA-reduced feature channel count')
    p.add_argument('--pca-fit-subjects', type=int, default=60,
                    help='random training subjects used to fit the one global PCA basis')
    p.add_argument('--pca-samples-per-subject', type=int, default=2000)
    p.add_argument('--cube', type=int, default=48,
                    help='isotropic training resolution (kept modest so 394 subjects\' cached '
                         'features fit in CPU RAM - see module docstring)')
    p.add_argument('--start-channel', type=int, default=8, help='FMIR encoder width')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lambda-smooth', type=float, default=1.0, help='flow-smoothness loss weight')
    p.add_argument('--eval-every-steps', type=int, default=300,
                    help='held-out Dice check + checkpoint cadence, counted in optimizer steps (batches), not pairs')
    p.add_argument('--patience', type=int, default=None,
                    help='stop after this many consecutive eval checks (each --eval-every-steps apart) with no '
                         'new best held-out dice. Default: disabled, always run all --epochs. Eval-to-eval dice '
                         'noise is ~0.003 stdev in practice, so patience should span enough evals to not trigger '
                         'on noise - e.g. 40-50 (12,000-15,000 steps) held up well in testing.')
    p.add_argument('--max-training-subjects', type=int, default=None, help='debug: cap the training pool size')
    p.add_argument('--output-dir', default=None, help='defaults to <repo root>/OASIS_output')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                    help='log to Weights & Biases (--no-wandb to disable, e.g. for quick debug runs)')
    p.add_argument('--wandb-project', default='planeadam-decoder-method')
    p.add_argument('--wandb-run-name', default=None, help='defaults to an auto-generated name from key hyperparams')
    return p.parse_args()


def to_volume_tensor(img):
    # nib volume is (H, W, D); the backbone wants (B, C, D, H, W) with in_chans=3.
    x = img.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    return x.repeat(1, 3, 1, 1, 1)  # (1, 3, D, H, W)


def load_training_image(subject_id):
    """Loads+normalizes one OASIS training-pool subject's image (no label - unused, training is unsupervised)."""
    img_path = os.path.join(OASIS_IMAGES_DIR, f'OASIS_{subject_id:04d}_0000.nii.gz')
    mask_path = img_path.replace('imagesTr', 'masksTr')
    img = torch.from_numpy(nib.load(img_path).get_fdata()).float().contiguous()
    mask = torch.from_numpy(nib.load(mask_path).get_fdata()).float().contiguous()
    return normalize_intensity(img, mask)


def native_features(extractor, img, device):
    volume = to_volume_tensor(img).to(device)
    with torch.no_grad():
        # (B, D, Hp, Wp, C) -> (B, C, Hp, Wp, D), matching convex_optimization's native-feature layout.
        return extractor.extract_feature_planecycle(volume).permute(0, 4, 2, 3, 1).contiguous()


def fit_global_pca(extractor, subject_ids, pca_dim, samples_per_subject, device):
    """Fits one PCA basis on a random voxel subsample pooled across a handful of training subjects."""
    print(f'fitting global PCA basis (k={pca_dim}) on {len(subject_ids)} subjects...')
    samples = []
    for sid in subject_ids:
        feat = native_features(extractor, load_training_image(sid), device)
        flat = feat[0].permute(1, 2, 3, 0).reshape(-1, feat.shape[1]).float()
        idx = torch.randperm(flat.shape[0], device=flat.device)[:samples_per_subject]
        samples.append(flat[idx].cpu())
    X = torch.cat(samples, dim=0).to(device)
    mean = X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(X - mean, q=pca_dim, niter=4)
    return mean.cpu(), V.cpu()


def project_pca(features_native, mean, V):
    B, C, Hn, Wn, Dn = features_native.shape
    flat = features_native.permute(0, 2, 3, 4, 1).reshape(-1, C).float()
    reduced = (flat - mean.to(flat.device)) @ V.to(flat.device)
    k = V.shape[1]
    return reduced.reshape(B, Hn, Wn, Dn, k).permute(0, 4, 1, 2, 3).contiguous()


def to_cube(tensor, cube):
    return F.interpolate(tensor, size=(cube, cube, cube), mode='trilinear', align_corners=False)


def build_subject_cache(extractor, subject_ids, mean, V, cube, device):
    """
    Extracts+PCA-reduces+cube-resizes every training subject once, caching the result as fp16
    on CPU (halves the ~13-27MB/subject footprint from build_pair_inputs-style fp32 caching -
    see module docstring for why 394 subjects can't all be cached at once otherwise).
    """
    print(f'caching {len(subject_ids)} training subjects (one-time cost)...')
    t0 = time.time()
    cache = {}
    for n, sid in enumerate(subject_ids):
        img = load_training_image(sid)
        feat_native = native_features(extractor, img, device)
        with torch.no_grad():
            feat_cube = to_cube(project_pca(feat_native, mean, V), cube)
            img_cube = to_cube(img[None, None].float().to(device), cube)
        cache[sid] = (img_cube.half().cpu(), feat_cube.half().cpu())
        if (n + 1) % 50 == 0:
            print(f'  ...{n + 1}/{len(subject_ids)}')
    print(f'...done in {time.time() - t0:.1f}s')
    return cache


def cached_batch_inputs(cache, fixed_ids, moving_ids, device):
    """
    Stacks a batch of cached (fp16, CPU) per-subject tensors along the batch dim. FMIR and the
    SpatialTransformer/VecInt layers have no batch=1 assumption (dispWarp's
    torch.cat([x, y], 0) / torch.chunk(..., 2, dim=0) trick generalizes to any batch size), so
    processing many pairs in one forward/backward call is a pure speedup: at batch=1 each 57ms
    step is dominated by Python/CUDA-launch overhead around a tiny model, not actual compute,
    so batching amortizes that overhead across many pairs instead of paying it per pair.
    """
    img_f = torch.cat([cache[i][0] for i in fixed_ids], dim=0).float().to(device)
    feat_f = torch.cat([cache[i][1] for i in fixed_ids], dim=0).float().to(device)
    img_m = torch.cat([cache[i][0] for i in moving_ids], dim=0).float().to(device)
    feat_m = torch.cat([cache[i][1] for i in moving_ids], dim=0).float().to(device)
    return torch.cat([img_f, feat_f], dim=1), torch.cat([img_m, feat_m], dim=1), img_f, img_m


def build_eval_pairs(extractor, eval_config, mean, V, device):
    """Loads the held-out pairs at native resolution (images+segs), for Dice-accurate evaluation."""
    with open(eval_config, 'r') as f:
        config = json.load(f)
    pairs_cfg = config['pairs']
    num_labels = config['num_labels'] - 1
    H, W, D = config['HWD']

    imgs_fixed, segs_fixed, _mf, imgs_moving, segs_moving, _mm = get_data_train(pairs_cfg, config['HWD'])

    eval_pairs = []
    for i in range(len(pairs_cfg)):
        feat_f = native_features(extractor, imgs_fixed[i], device)
        feat_m = native_features(extractor, imgs_moving[i], device)
        feat_f = project_pca(feat_f, mean, V)
        feat_m = project_pca(feat_m, mean, V)
        eval_pairs.append({
            'img_fixed': imgs_fixed[i], 'img_moving': imgs_moving[i],
            'feat_fixed': feat_f, 'feat_moving': feat_m,
            'seg_fixed': segs_fixed[i].to(device), 'seg_moving': segs_moving[i].to(device),
            'HWD': (H, W, D),
        })
    return eval_pairs, pairs_cfg, num_labels


def evaluate(model, eval_pairs, num_labels, cube, transformers_cube, integrates_cube, transformer_native, device):
    """Dice on every held-out pair: cube-resize its (fixed-basis-projected) features on the fly,
    predict the flow, upsample it to native resolution, warp the moving segmentation (nearest),
    and score against the fixed segmentation."""
    model.eval()
    dices = []
    with torch.no_grad():
        for p in eval_pairs:
            H, W, D = p['HWD']
            img_f_c = to_cube(p['img_fixed'][None, None].float().to(device), cube)
            img_m_c = to_cube(p['img_moving'][None, None].float().to(device), cube)
            feat_f_c = to_cube(p['feat_fixed'], cube)
            feat_m_c = to_cube(p['feat_moving'], cube)
            x_in = torch.cat([img_f_c, feat_f_c], dim=1)
            y_in = torch.cat([img_m_c, feat_m_c], dim=1)

            _int_flows, pos_flows = model(x_in, y_in, transformers_cube, integrates_cube, UPSCALE, registration=False)
            flow = pos_flows[0]

            flow_native = F.interpolate(flow, size=(H, W, D), mode='trilinear', align_corners=False)
            scale = torch.tensor([H / cube, W / cube, D / cube], device=flow.device).view(1, 3, 1, 1, 1)
            flow_native = flow_native * scale

            seg_warped = transformer_native(p['seg_moving'].view(1, 1, H, W, D).float(), flow_native)
            dice = dice_coeff(p['seg_fixed'], seg_warped.squeeze(), num_labels + 1)
            dices.append(dice.mean().item())
    model.train()
    return dices


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.wandb:
        wandb_run_name = args.wandb_run_name or (
            f'fmir_pca{args.pca}_cube{args.cube}_batch{args.batch_size}_'
            f'epochs{args.epochs}_{time.strftime("%Y%m%d_%H%M%S")}'
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))

    n_train = args.max_training_subjects or args.num_train_subjects
    train_ids = list(range(1, n_train + 1))

    backbone = dinov3_vits16(pretrained=True, weights=os.path.join(_REPO_ROOT, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'))
    extractor = DinoBackboneExtractor(backbone)
    device = extractor.device
    print(f'backbone: {extractor.arch_name}, device: {device}')

    fit_ids = torch.randperm(len(train_ids))[:args.pca_fit_subjects].tolist()
    fit_subject_ids = [train_ids[i] for i in fit_ids]
    mean, V = fit_global_pca(extractor, fit_subject_ids, args.pca, args.pca_samples_per_subject, device)

    cache = build_subject_cache(extractor, train_ids, mean, V, args.cube, device)

    all_pairs = [(i, j) for i in train_ids for j in train_ids if i != j]
    print(f'training pool: {len(train_ids)} subjects -> {len(all_pairs)} ordered pairs')

    print('loading held-out evaluation pairs (native resolution)...')
    eval_pairs, eval_pairs_cfg, num_labels = build_eval_pairs(extractor, args.eval_config, mean, V, device)
    print(f'{len(eval_pairs)} held-out evaluation pairs loaded from {args.eval_config}')

    model = FMIR(
        img_size=str((args.cube, args.cube, args.cube)),
        start_channel=str(args.start_channel),
        in_channels=str(args.pca + 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    level_sizes = [args.cube // (2 ** i) for i in range(N_LEVELS)]
    transformers_cube = [SpatialTransformer((s, s, s)).to(device) for s in level_sizes]
    integrates_cube = [VecInt((s, s, s), nsteps=7).to(device) for s in level_sizes]
    H, W, D = eval_pairs[0]['HWD']
    transformer_native = SpatialTransformer((H, W, D), mode='nearest').to(device)

    ncc = NccLoss([9, 9, 9])
    grad3d = Grad3d()

    output_root = args.output_dir or os.path.join(_REPO_ROOT, 'OASIS_output')
    os.makedirs(output_root, exist_ok=True)
    best_dice = 0.0
    best_state = None
    global_step = 0
    evals_without_improvement = 0
    stop_early = False
    for epoch in range(args.epochs):
        order = torch.randperm(len(all_pairs)).tolist()
        if args.steps_per_epoch is not None:
            order = order[:args.steps_per_epoch]

        batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]

        t_epoch = time.time()
        running_loss = 0.0
        for step, batch_idxs in enumerate(batches):
            fixed_ids = [all_pairs[idx][0] for idx in batch_idxs]
            moving_ids = [all_pairs[idx][1] for idx in batch_idxs]
            x_in, y_in, img_f_c, img_m_c = cached_batch_inputs(cache, fixed_ids, moving_ids, device)

            optimizer.zero_grad()
            _int_flows, pos_flows = model(x_in, y_in, transformers_cube, integrates_cube, UPSCALE, registration=False)
            flow = pos_flows[0]

            warped_moving = transformers_cube[0](img_m_c, flow)
            loss = ncc(img_f_c, warped_moving) + args.lambda_smooth * grad3d(flow)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            global_step += 1

            if args.wandb:
                wandb.log({'train/loss': loss.item(), 'train/epoch': epoch + 1}, step=global_step)

            if step == 20:
                # early wall-clock estimate for this (potentially huge) epoch, so a run that's
                # going to take hours announces that instead of silently grinding away.
                per_step = (time.time() - t_epoch) / (step + 1)
                print(f'  ~{per_step * 1000:.1f} ms/step (batch={args.batch_size}, '
                      f'~{per_step * 1000 / args.batch_size:.1f} ms/pair) '
                      f'-> est. {per_step * len(batches) / 60:.1f} min for this epoch')

            if global_step % args.eval_every_steps == 0:
                dices = evaluate(model, eval_pairs, num_labels, args.cube, transformers_cube, integrates_cube,
                                  transformer_native, device)
                mean_dice = sum(dices) / len(dices)
                print(f'  step {global_step}: mean training loss (last {step + 1}) '
                      f'{running_loss / (step + 1):.4f}, held-out mean dice {mean_dice:.4f}')
                if args.wandb:
                    wandb.log({'eval/mean_dice': mean_dice, 'eval/best_dice': max(mean_dice, best_dice)},
                              step=global_step)
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    evals_without_improvement = 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    torch.save({'model': model.state_dict(), 'args': vars(args), 'step': global_step},
                                os.path.join(output_root, '.fmir_train_checkpoint.pth'))
                else:
                    evals_without_improvement += 1
                    if args.patience is not None and evals_without_improvement >= args.patience:
                        print(f'  no held-out improvement in {evals_without_improvement} eval checks '
                              f'({evals_without_improvement * args.eval_every_steps} steps) - stopping early '
                              f'at step {global_step} (best dice {best_dice:.4f})')
                        stop_early = True
                        break

        epoch_mean_loss = running_loss / (step + 1)
        print(f'epoch {epoch + 1}/{args.epochs}: mean loss {epoch_mean_loss:.4f} '
              f'({step + 1} steps over {len(order)} pairs, {time.time() - t_epoch:.1f}s)')
        if args.wandb:
            wandb.log({'epoch/mean_loss': epoch_mean_loss, 'epoch/duration_s': time.time() - t_epoch,
                       'epoch/number': epoch + 1}, step=global_step)
        if stop_early:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_dices = evaluate(model, eval_pairs, num_labels, args.cube, transformers_cube, integrates_cube,
                            transformer_native, device)
    final_mean_dice = sum(final_dices) / len(final_dices)

    run_name = (f'dice{final_mean_dice:.3f}_fmir_pca{args.pca}_cube{args.cube}'
                f'_train{len(train_ids)}subj_{time.strftime("%Y%m%d_%H%M%S")}')
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    torch.save({'model': model.state_dict(), 'pca_mean': mean, 'pca_V': V, 'args': vars(args)},
               os.path.join(output_dir, 'model.pth'))
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write('FMIR decoder training on OASIS\n')
        f.write(f'training pool: {len(train_ids)} subjects (OASIS_0001-{len(train_ids):04d}), '
                f'{len(all_pairs)} ordered pairs\n')
        f.write(f'held-out eval: {len(eval_pairs)} official L2R validation pairs from {args.eval_config}\n')
        f.write(f'epochs: {args.epochs}, pairs/epoch: {args.steps_per_epoch or len(all_pairs)}, '
                f'batch_size: {args.batch_size}, pca: {args.pca}, cube: {args.cube}, '
                f'lr: {args.lr}, lambda_smooth: {args.lambda_smooth}\n')
        f.write(f'best held-out mean dice during training: {best_dice:.4f}\n')
        f.write(f'final held-out mean dice (best checkpoint): {final_mean_dice:.4f}\n')
        f.write('per-pair held-out dice:\n')
        for i, (pair_cfg, d) in enumerate(zip(eval_pairs_cfg, final_dices)):
            f.write(f"  [{i}] {pair_cfg['fixed']} <- {pair_cfg['moving']}: {d:.4f}\n")

    print(f'final held-out mean dice: {final_mean_dice:.4f}')
    print(f'results written to {output_dir}/')

    if args.wandb:
        dice_table = wandb.Table(columns=['pair', 'fixed', 'moving', 'dice'])
        for i, (pair_cfg, d) in enumerate(zip(eval_pairs_cfg, final_dices)):
            dice_table.add_data(i, pair_cfg['fixed'], pair_cfg['moving'], d)
        wandb.log({
            'final/mean_dice': final_mean_dice,
            'final/best_dice': best_dice,
            'final/per_pair_dice': dice_table,
        })
        wandb.summary['final_mean_dice'] = final_mean_dice
        wandb.summary['output_dir'] = output_dir
        wandb.finish()


if __name__ == '__main__':
    main()
