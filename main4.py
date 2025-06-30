import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import UE_datasets, RealFakeDataset
from tensorboardX import SummaryWriter
from models.MCDF import MCDF
from models.Dual_MCDF import DualBranchMCDF
from models.gilm_det import GILMNet
from models.glim_clip import GILM_Clip
from models.clip.clip_mamba import CLIP_VMamba_S  # 确保从正确的文件导入
from models.clip import clip
from models.vmamba_model import VSSM
from engine import *
import os
import sys
import csv
from tqdm import tqdm
from tqdm import tqdm
from earlystop import EarlyStopping, adjust_learning_rate

from utils import *
from utils import SupConLoss
from configs.config_setting import setting_config
from configs.config_mcdf import config_mcdf
from configs.config_clip import config_clip
from configs.config_clip_norm import config_mcdf_clip
from configs.config_dual_mcdf import config_dual_mcdf
from configs.config_dual_mcdf_2d import config_dual_mcdf_2d

import warnings
warnings.filterwarnings("ignore")

CHANNELS = {
    "CLIP_VMamba_S": 512,
    "CLIP_VMamba_B": 512,
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "CLIP_Simba_B": 512,
    "CLIP_Simba_L": 512,
}

def run_custom_test(model,
                    criterion_cls,      # ← nn.BCEWithLogitsLoss()
                    config, logger):
    """
    Custom test each generator folder under `config.test_root`.
    保存结果到 <work_dir>/test_results_by_generator.csv
    """
    print('#---------- Custom Testing ----------#')

    # ---------- 载入最优权重 ----------
    best_model_path = (config.best_ckpt_path if config.only_test_and_save_figs
                       else os.path.join(
                               config.work_dir, 'checkpoints',
                               'best_ap.pth' if config.metric_mode == 'ap'
                               else 'best_loss.pth'))
    print(f'Loading model weights from: {best_model_path}')
    best_weight = torch.load(best_model_path, map_location='cpu')
    model.load_state_dict(best_weight.get('model_state_dict', best_weight))
    model = model.cuda()
    model.eval()

    test_root = config.test_root
    results = []

    # ---------- 断点续测 ----------
    resume_txt_path = getattr(config, 'resume_txt_path', None)
    tested_generators = set()
    if resume_txt_path and os.path.exists(resume_txt_path):
        with open(resume_txt_path, 'r') as f:
            tested_generators = {line.split()[0].strip(',') for line in f if line.strip()}
        print(f'[Resume] Loaded {len(tested_generators)} tested generators from {resume_txt_path}')
    else:
        print('[Resume] No resume file provided or file does not exist. Starting fresh.')

    # ---------- 逐生成器测试 ----------
    for gen_name in sorted(os.listdir(test_root)):
        gen_path = os.path.join(test_root, gen_name)
        if not os.path.isdir(gen_path) or gen_name in tested_generators:
            continue

        print(f'Testing {gen_name}...')
        config.data_path = gen_path
        test_dataset = RealFakeDataset(config, split='custom_test')
        test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                  pin_memory=True, num_workers=config.num_workers)

        # === 单轮推理 ===
        loss_meter, ap_meter = 0.0, 0.0
        r_acc0 = f_acc0 = acc0 = 0.0
        r_acc1 = f_acc1 = acc1 = 0.0
        best_thres = 0.5  # 若 test_one_epoch 返回其它信息，可替换
        with torch.no_grad():
            for images, targets in tqdm(test_loader, leave=False):
                images   = images.cuda(non_blocking=True).float()
                targets  = targets.cuda(non_blocking=True).float()
                logits   = model(images).squeeze(1)        # 只要 logits
                loss     = criterion_cls(logits, targets)

                # ===== 根据你自己的 metrics 更新 =====
                # 这里只演示累加 loss，其他指标请用你已有的 `test_one_epoch`
                loss_meter += loss.item() * images.size(0)

        loss_meter /= len(test_loader.dataset)

        # 若有自定义 metrics 可替换下列变量
        loss = loss_meter
        ap   = ap_meter

        msg = (f'{gen_name}, Loss={loss:.4f}, AP={ap:.4f}, '
               f'Acc@0.5={acc0:.4f}, Recall={r_acc0:.4f}, '
               f'Specificity={f_acc0:.4f}, Thres={best_thres:.3f}, '
               f'Acc_best={acc1:.4f}, R_best={r_acc1:.4f}, S_best={f_acc1:.4f}')
        logger.info(f'[TEST] {msg}')
        print(f'[TEST] {msg}')
        results.append((gen_name, loss, ap, acc0, r_acc0, f_acc0,
                        acc1, r_acc1, f_acc1))

        # 断点续测记录
        if resume_txt_path:
            with open(resume_txt_path, 'a') as f:
                f.write(msg + '\n')

    # ---------- 保存 CSV ----------
    csv_path = os.path.join(config.work_dir, 'test_results_by_generator.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generator', 'Loss', 'AP', 'Acc@0.5', 'Recall',
                         'Specificity', 'Acc_best', 'Recall_best', 'Spec_best'])
        writer.writerows(results)
    print(f'Results saved to {csv_path}')


def main(config):

    print('#----------Creating logger----------#')
    os.makedirs(config.work_dir + '/log', exist_ok=True)
    os.makedirs(config.work_dir + '/checkpoints', exist_ok=True)
    os.makedirs(config.work_dir + '/outputs', exist_ok=True)

    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()
    
    print('#----------Preparing dataset----------#')
    if config.datasets == 'RealFake':
        train_dataset = RealFakeDataset(config, split="train")
        val_dataset = RealFakeDataset(config, split="val")
    else:
        train_dataset = UE_datasets(config.data_path, config, train=True)
        val_dataset = UE_datasets(config.data_path, config, train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=config.num_workers)

    print("#----------Creating CLIP backbone----------#")
    def get_clip_backbone(name):
        from models.clip.clip_mamba import CLIP_VMamba_S, CLIP_VMamba_B, CLIP_Simba_B, CLIP_Simba_L
        backbones = {
            "CLIP_VMamba_S": CLIP_VMamba_S,
            "CLIP_VMamba_B": CLIP_VMamba_B,
            "CLIP_Simba_B": CLIP_Simba_B,
            "CLIP_Simba_L": CLIP_Simba_L,
        }
        if name in backbones:
            return backbones[name]()
        elif name == "ViT-L/14":
            model, preprocess = clip.load(name, device="cpu")
            return model

    clip_backbone = get_clip_backbone(config.clip_backbone_type)

    if config.clip_backbone_type in {"CLIP_VMamba_S", "CLIP_VMamba_B", "CLIP_Simba_B", "CLIP_Simba_L"}:
        ckpt = torch.load(config.clip_backbone_path, map_location='cpu',weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        clip_backbone.load_state_dict(state_dict)
        for param in clip_backbone.parameters():
            param.requires_grad = False

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = GILMNet(backbone=None, **model_cfg)
        model.load_from()
    elif config.network == 'mamba_clip':
        model = GILM_Clip(backbone=clip_backbone, backbone_type=config.clip_backbone_type, **model_cfg)
    elif config.network == 'mcdf':
        model = MCDF(backbone=clip_backbone, num_classes=model_cfg['num_classes'], feature_dim=CHANNELS[config.clip_backbone_type])
    elif config.network == 'dual_mcdf':
        model = DualBranchMCDF(
                    backbone       = clip_backbone,
                    num_classes    = model_cfg['num_classes'],
                    spatial_dim    = CHANNELS[config.clip_backbone_type],
                    freq_dim_red   = model_cfg.get('freq_dim_red', 384),
                    freeze_clip    = model_cfg.get('freeze_clip', False)
                )
    else:
        raise ValueError('Unsupported network')

    model = model.cuda()

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion_cls = nn.BCEWithLogitsLoss()

    if getattr(config, "use_supcon", False):
        criterion_sup = SupConLoss(config.supcon_temperature)
        lambda_sup    = config.supcon_weight
        print(f"[Loss] SupCon ON  λ={lambda_sup}, τ={config.supcon_temperature}")
    else:
        criterion_sup = None
        lambda_sup    = 0.0
        print("[Loss] SupCon OFF")
    
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    # 【注意】这里的patience现在代表epoch数而不是验证次数
    earlystop = EarlyStopping(patience=config.earlystop_epoch, delta=config.earlystop_delta)

    print('#----------Set other params----------#')
    step = 0
    start_epoch = 1
    min_loss, min_ap, min_epoch = float('inf'), -1, 0

    if config.only_test_and_save_figs:
        # ... (这部分逻辑不变)
        run_custom_test(model, criterion_cls, config, logger)
        return

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if hasattr(config, 'adjust_lr') and config.adjust_lr is not None:
            # ... (这部分逻辑不变)
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.adjust_lr

        # 从 checkpoint 恢复 epoch 和 step
        start_epoch = checkpoint.get('epoch', 1)
        step = checkpoint.get('step', 0)
        min_loss = checkpoint.get('min_loss', float('inf'))
        min_ap = checkpoint.get('min_ap', -1)
        min_epoch_resume = checkpoint.get('min_epoch', 0) # 使用不同的变量名以防混淆

        log_info = (f'Resuming model from {resume_model}. '
                    f'Starting at Epoch: {start_epoch}, Step: {step}, '
                    f'Best Loss: {min_loss:.4f}, Best AP: {min_ap:.4f} at epoch {min_epoch_resume}')
        logger.info(log_info)
        print(log_info)

    print('#----------Training with Epochs----------#')
    # ======================= 主要修改部分：改为 Epoch-based 循环 =======================
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        
        # 使用 tqdm 显示每个 epoch 内的进度
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch in pbar:
            step = train_one_step(batch, model, criterion_cls=criterion_cls, criterion_sup=criterion_sup,
                                  lambda_sup=lambda_sup, optimizer=optimizer, scheduler=scheduler,
                                  step=step, logger=logger, config=config, writer=writer)

        # --- 每个 Epoch 结束后执行一次验证和保存 ---
        loss, ap, *_ = val_one_epoch(val_loader, model, criterion_cls, step, logger, config)
        writer.add_scalar('Val/Epoch_Loss', loss, epoch)
        writer.add_scalar('Val/Epoch_AP', ap, epoch)

        # 保存最佳模型
        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_loss.pth'))
            min_loss = loss
        if ap > min_ap:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_ap.pth'))
            min_ap = ap
            min_epoch = epoch # 记录最好的 epoch
        
        # 保存用于断点续训的最新模型 (每轮都保存)
        torch.save({
            'epoch': epoch + 1,  # 保存下一个要开始的 epoch
            'step': step,
            'min_loss': min_loss,
            'min_ap': min_ap,
            'min_epoch': min_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'latest.pth'))
        logger.info(f"Epoch {epoch} finished. Checkpoint saved to 'latest.pth'")
        # 保存当前 epoch 独立的权重（不会被覆盖）
        epoch_ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.pth')
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, epoch_ckpt_path)
        logger.info(f"Saved epoch checkpoint: {epoch_ckpt_path}")
        # 早停检查和学习率调整
        earlystop(ap) # 使用验证集 AP 进行判断
        if earlystop.early_stop:
            print(f"[Main Loop] Early stopping patience ({earlystop.patience} epochs) reached. Attempting to adjust learning rate.")
            
            can_continue = adjust_learning_rate(optimizer, decay_factor=0.1, min_lr=config.min_lr)
            
            if can_continue:
                current_lr = optimizer.param_groups[0]['lr']
                msg = f"[Main Loop] Learning rate adjusted to {current_lr}. Resetting early stopping counter."
                print(msg)
                logger.info(msg)
                earlystop.reset() # 重置早停计数器
            else:
                msg = "[Main Loop] Learning rate is at its minimum. Stopping training."
                print(msg)
                logger.info(msg)
                break # 学习率已到最小，跳出训练循环

    # ======================= 修改结束 =======================

    print("\n#----------Training Finished----------#")
    # 训练结束后，使用最好的模型进行最终测试
    run_custom_test(model, criterion_cls, config, logger)

if __name__ == '__main__':
    # 确保你的配置文件函数返回的对象中包含 `epochs` 属性
    config = config_mcdf()
    # 示例：手动添加 epochs 参数，实际应在 config 文件中定义
    # if not hasattr(config, 'epochs'):
    #     config.epochs = 100 
    main(config)
