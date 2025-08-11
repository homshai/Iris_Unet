# train.py - improved training: weighted sampling to balance datasets, mix precision, combined loss, cosine lr with warmup
import os, math, time, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from model import build_model
from dataset import build_combined_dataset
from utils import dice_loss, save_checkpoint, iou_score, dice_score, count_parameters
from tqdm import tqdm

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

def combined_loss(preds, targets, bce_fn, dice_weight=1.0):
    bce = bce_fn(preds, targets)
    preds_sig = torch.sigmoid(preds)
    d = dice_loss(preds_sig, targets)
    return bce + dice_weight * d

def train(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    train_ds, val_ds, components = build_combined_dataset(args.data_roots, img_size=args.img_size, augment=True, val_split=args.val_split, seed=args.seed)
    component_lengths = [len(c) for c in components]
    print('Per-component sizes:', component_lengths)
    sampler = make_sampler(train_ds, component_lengths)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = build_model(num_classes=1, pretrained=True, base_channels=args.base_channels).to(device)
    print('Model params:', count_parameters(model))
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
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for imgs, masks in pbar:
            imgs = imgs.to(device); masks = masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                preds = model(imgs)
                loss = combined_loss(preds, masks, bce_fn, dice_weight=1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f'{(epoch_loss/((pbar.n+1)*train_loader.batch_size)):.4f}')
        scheduler.step()
        if epoch+1 == args.freeze_backbone_epochs:
            for p in model.parameters():
                p.requires_grad = True
            print('Unfroze backbone parameters.')
        model.eval()
        iou_m = 0.0; dice_m = 0.0; n_val = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device); masks = masks.to(device)
                preds = model(imgs)
                preds_bin = (torch.sigmoid(preds) > 0.5).float()
                iou_m += iou_score(preds_bin, masks).item() * imgs.size(0)
                dice_m += dice_score(preds_bin, masks).item() * imgs.size(0)
                n_val += imgs.size(0)
        iou_avg = iou_m / max(1, n_val); dice_avg = dice_m / max(1, n_val)
        print(f'Epoch {epoch}: val IoU={iou_avg:.4f}, Dice={dice_avg:.4f}')
        if iou_avg > best_iou:
            best_iou = iou_avg
            save_checkpoint(model, optimizer, epoch, best_iou, os.path.join(args.save_dir, 'best.pth'))
            print('Saved new best model.')
        save_checkpoint(model, optimizer, epoch, best_iou, os.path.join(args.save_dir, 'last.pth'))
