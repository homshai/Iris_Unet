# utils.py - metrics, losses, checkpoint helpers
import torch, os, cv2, numpy as np, time

def iou_score(preds, targets, eps=1e-7):
    preds = preds.float(); targets = targets.float()
    inter = (preds * targets).sum(dim=[1,2,3])
    union = (preds + targets - preds*targets).sum(dim=[1,2,3])
    iou = (inter + eps) / (union + eps)
    return iou.mean()

def dice_score(preds, targets, eps=1e-7):
    preds = preds.float(); targets = targets.float()
    inter = (preds * targets).sum(dim=[1,2,3])
    denom = preds.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3])
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()

def dice_loss(preds, targets, eps=1e-7):
    preds = preds.float(); targets = targets.float()
    inter = (preds * targets).sum(dim=[1,2,3])
    denom = preds.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3])
    loss = 1.0 - ((2 * inter + eps) / (denom + eps))
    return loss.mean()

def save_checkpoint(model, optimizer, epoch, best_iou, path):
    ck = {'model': model.state_dict(), 'optimizer': getattr(optimizer,'state_dict', lambda: None)(), 'epoch': epoch, 'best_iou': best_iou}
    torch.save(ck, path)

def save_mask(mask_np, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    ext = os.path.splitext(path)[1][1:] or 'png'
    _, buf = cv2.imencode('.' + ext, mask_np)
    buf.tofile(path)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    params_mb = total * 4 / (1024**2)
    return f'{total} params ({params_mb:.2f} MB)'

def measure_inference_fps(model, loader, device, warmup=5, runs=100):
    model.to(device); model.eval()
    
    # Warmup
    it = iter(loader)
    with torch.no_grad():
        for _ in range(warmup):
            try:
                imgs, _ = next(it)
            except StopIteration:
                break
            imgs = imgs.to(device)
            _ = model(imgs)
    
    # Measure FPS
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            _ = model(imgs)
            n += imgs.size(0)
            if n >= runs:
                break
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.time()
    elapsed = t1 - t0
    
    return n / (elapsed + 1e-12) if elapsed > 0 else 0
