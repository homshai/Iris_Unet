# inference.py - inference adapted to logits output
import os, glob, cv2, numpy as np, torch
from model import build_model
from utils import save_mask, count_parameters, measure_inference_fps
from torch.utils.data import DataLoader
from dataset import IrisDataset
import time

def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    model = build_model(num_classes=1, pretrained=False, base_channels=32)
    ck = torch.load(args.weights, map_location=device)
    state = ck.get('model', ck)
    model.load_state_dict(state)
    model.to(device); model.eval()
    print('Model params:', count_parameters(model))
    
    inputs = []
    if os.path.isdir(args.input):
        exts = ['*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff']
        for e in exts:
            inputs += glob.glob(os.path.join(args.input, e))
        inputs = sorted(inputs)
    elif os.path.isfile(args.input):
        inputs = [args.input]
    else:
        raise FileNotFoundError('Input not found')
    
    dataset = IrisDataset(inputs, inputs, img_size=args.img_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Measure FPS multiple times to get statistics
    fps_measurements = []
    num_measurements = 5
    for i in range(num_measurements):
        fps = measure_inference_fps(model, loader, device, warmup=5, runs=100)
        fps_measurements.append(fps)
        print(f'FPS measurement {i+1}/{num_measurements}: {fps:.2f}')
    
    # Calculate statistics
    avg_fps = np.mean(fps_measurements)
    std_fps = np.std(fps_measurements)
    min_fps = np.min(fps_measurements)
    max_fps = np.max(fps_measurements)
    
    print(f'FPS Statistics: Avg={avg_fps:.2f}, Std={std_fps:.2f}, Min={min_fps:.2f}, Max={max_fps:.2f}')
    
    os.makedirs(args.output, exist_ok=True)
    idx=0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            probs = torch.sigmoid(preds).cpu().numpy()
            for b in range(probs.shape[0]):
                prob = probs[b,0]
                mask = (prob > 0.5).astype('uint8')*255
                
                # 使用原图文件名作为基础，添加_mask后缀
                original_img_path = inputs[idx]
                base_name = os.path.splitext(os.path.basename(original_img_path))[0]
                fname = f'{base_name}_mask.png'
                save_mask(mask, os.path.join(args.output, fname))
                
                # 如果启用了可视化参数，将预测的mask叠加到原图进行对比
                if hasattr(args, 'visualize') and args.visualize:
                    # 获取原始图像
                    original_img = cv2.imread(original_img_path)
                    if original_img is None:
                        original_img = cv2.imdecode(np.fromfile(original_img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    
                    # 如果是单通道图像，转换为三通道
                    if len(original_img.shape) == 2 or original_img.shape[2] == 1:
                        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
                    
                    # 调整mask尺寸以匹配原始图像
                    resized_mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # 创建叠加图像
                    overlay = original_img.copy()
                    # 将mask区域标记为绿色
                    overlay[resized_mask > 0] = [0, 255, 0]
                    
                    # 混合原图和mask
                    alpha = 0.6
                    combined = cv2.addWeighted(original_img, alpha, overlay, 1-alpha, 0)
                    
                    # 保存叠加图像，使用原图文件名作为基础，添加_overlay后缀
                    overlay_fname = f'{base_name}_overlay.png'
                    cv2.imwrite(os.path.join(args.output, overlay_fname), combined)
                
                idx+=1
