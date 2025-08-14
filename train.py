# train.py - improved training with optional boundary loss
import os, math, time, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from model import build_model
from dataset import build_combined_dataset
from utils import dice_loss, save_checkpoint, iou_score, dice_score, count_parameters, boundary_target
from tqdm import tqdm
from matplotlib import pyplot as plt

def seed_everything(seed=42):
    random.seed(seed); import numpy as np; np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_sampler(subsets, component_lengths):
    weights = []
    cum = [0]
    for l in component_lengths:
        cum.append(cum[-1]+l)
    for gi in subsets.indices:
        for ds_id in range(len(component_lengths)):
            if cum[ds_id] <= gi < cum[ds_id+1]:
                w = 1.0 / float(component_lengths[ds_id])
                weights.append(w); break
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

def combined_loss_logits(preds_logits, targets, bce_fn, dice_weight=1.0):
    """
    preds_logits: raw logits (B,1,H,W)
    targets: {0,1}
    """
    bce = bce_fn(preds_logits, targets)
    preds_sig = torch.sigmoid(preds_logits)
    d = dice_loss(preds_sig, targets)
    return bce + dice_weight * d

def train(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    # build combined dataset
    train_ds, val_ds, components = build_combined_dataset(args.data_roots, img_size=args.img_size, augment=True, val_split=args.val_split, seed=args.seed)
    component_lengths = [len(c) for c in components]
    print('Per-component sizes:', component_lengths)
    sampler = make_sampler(train_ds, component_lengths)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    use_boundary = (args.boundary_weight is not None and float(args.boundary_weight) > 0.0)
    model = build_model(num_classes=1, pretrained=True, base_channels=args.base_channels, use_boundary=use_boundary, backbone=args.backbone)
    model = model.to(device)
    print('Model params:', count_parameters(model))

    # optionally freeze backbone
    if args.freeze_backbone_epochs > 0:
        for name, p in model.named_parameters():
            if 'enc' not in name and 'initial' not in name and 'aspp' not in name:
                continue
            p.requires_grad = False
        print(f'Backbone frozen for first {args.freeze_backbone_epochs} epochs')

    bce_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        warmup = args.warmup_epochs
        if epoch < warmup:
            return float(epoch+1) / float(max(1, warmup))
        else:
            progress = float(epoch - warmup) / float(max(1, args.epochs - warmup))
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    best_iou = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    # Lists to store metrics for plotting
    train_losses = []
    val_ious = []
    val_dices = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for imgs, masks in pbar:
            imgs = imgs.to(device); masks = masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                outputs = model(imgs)
                # model returns either logits or (logits, boundary_logits)
                if use_boundary:
                    logits, boundary_logits = outputs
                else:
                    logits = outputs
                    boundary_logits = None

                main_loss = combined_loss_logits(logits, masks, bce_fn, dice_weight=1.0)
                loss = main_loss
                if use_boundary and boundary_logits is not None:
                    # compute edge ground truth from mask
                    edge_gt = boundary_target(masks)  # returns float 0/1
                    bce_boundary = bce_fn(boundary_logits, edge_gt)
                    # You may also use dice on boundary if you want; here we use BCE for boundary
                    loss = loss + float(args.boundary_weight) * bce_boundary

            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item() * imgs.size(0)
            # update progress with averaged loss
            seen = (pbar.n + 1) * train_loader.batch_size
            avg_loss = epoch_loss / max(1, seen)
            pbar.set_postfix(loss=f'{avg_loss:.4f}')
        scheduler.step()

        # unfreeze if needed
        if epoch+1 == args.freeze_backbone_epochs:
            for p in model.parameters():
                p.requires_grad = True
            print('Unfroze backbone parameters.')

        # validation
        model.eval()
        iou_m = 0.0; dice_m = 0.0; n_val = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device); masks = masks.to(device)
                outputs = model(imgs)
                if use_boundary:
                    logits, _ = outputs
                else:
                    logits = outputs
                preds = torch.sigmoid(logits)
                preds_bin = (preds > 0.5).float()
                batch_n = imgs.size(0)
                iou_m += iou_score(preds_bin, masks).item() * batch_n
                dice_m += dice_score(preds_bin, masks).item() * batch_n
                n_val += batch_n

        iou_avg = iou_m / max(1, n_val); dice_avg = dice_m / max(1, n_val)
        print(f'Epoch {epoch}: val IoU={iou_avg:.4f}, Dice={dice_avg:.4f}')
        
        # Store metrics for plotting
        train_losses.append(epoch_loss / len(train_loader.dataset))
        val_ious.append(iou_avg)
        val_dices.append(dice_avg)
        
        # save best
        if iou_avg > best_iou:
            best_iou = iou_avg
            save_checkpoint(model, optimizer, epoch, best_iou, os.path.join(args.save_dir, 'best.pth'))
            print('Saved new best model.')
        save_checkpoint(model, optimizer, epoch, best_iou, os.path.join(args.save_dir, 'last.pth'))
    

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot metrics
    epochs_range = range(args.epochs)
    ax.plot(epochs_range, val_ious, label='IoU', marker='o', markersize=3)
    ax.plot(epochs_range, val_dices, label='Dice', marker='s', markersize=3)
    
    # Set labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation IoU and Dice Coefficients')
    ax.legend()
    ax.grid(True)
    
    # Save plot
    plot_path = os.path.join(args.save_dir, 'metrics_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Metrics plot saved to {plot_path}')

    
    # Convert best.pth to ONNX format
    try:
        import subprocess
        import sys
        onnx_convert_cmd = [
            sys.executable, 'convert_to_onnx.py',
            '--pth-path', os.path.join(args.save_dir, 'best.pth'),
            '--onnx-path', os.path.join(args.save_dir, 'best.onnx'),
            '--img-size', str(args.img_size),
            '--base-channels', str(args.base_channels),
            '--backbone', args.backbone
        ]
        if use_boundary:
            onnx_convert_cmd.append('--use-boundary')
        print('Converting best.pth to ONNX format...')
        subprocess.run(onnx_convert_cmd, check=True)
        print('ONNX conversion completed successfully.')
    except FileNotFoundError:
        print('Failed to convert model to ONNX: convert_to_onnx.py script not found.')
        print('Please ensure you have the convert_to_onnx.py script in the project directory.')
    except subprocess.CalledProcessError as e:
        print(f'Failed to convert model to ONNX: convert_to_onnx.py script failed with return code {e.returncode}.')
        print('Please check if you have all the required dependencies installed, especially onnxruntime.')
        print('You can install it with: pip install onnxruntime')
    except Exception as e:
        print(f'Failed to convert model to ONNX: {e}')
        print('Please check if you have all the required dependencies installed, especially onnxruntime.')
        print('You can install it with: pip install onnxruntime')
