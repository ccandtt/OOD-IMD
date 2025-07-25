from torchvision import transforms
from utils import *
from datetime import datetime
import torch.nn as nn

CHANNELS = {
    "CLIP_VMamba_S": 512,
    "CLIP_VMamba_B": 512
}

class config_mcdf:
    """
    Training and model configuration for MCDF
    """

    network = 'mcdf'
    model_config = {
        'num_classes': 1,
    }

    clip_backbone_type = "CLIP_VMamba_B"
    clip_backbone_path = "models/clip/VMamba_B_clip.pt"
    backbone_embed_dim = CHANNELS[clip_backbone_type]

    # 数据相关
    datasets = 'RealFake'  
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    elif datasets == 'UltraEdit':
        data_path = "../datasets/UltraEdit"
    elif datasets == 'RealFake':
        data_path = "./datasets/RealFake"
    else:
        raise Exception('datasets is not right!')

    data_label = "train"
    data_mode = "wang2020"
    isTrain = True
    cropSize = 224
    loadSize = [256, 256]
    no_flip = True
    no_crop = False
    no_resize = False
    arch = "clip"
    wang2020_data_path = "/mnt/d/dataset/RFNT"
    test_root = "/mnt/d/dataset/fdmas"
    # wang2020_data_path = "/mnt/d/dataset/FD/fdmas_sample"
    # test_root = "/mnt/d/dataset/FD/fdmas_sample"
    resume_txt_path = None

    # 数据增强参数
    blur_prob = 0
    blur_sig = [0.0, 3.0]
    jpg_prob = 0
    jpg_method = ["cv2", "pil"]
    jpg_qual = [30, 100]
    rz_interp =  ["bilinear"]   # ["bilinear", "bicubic", "lanczos", "nearest"]

    # criterion = nn.BCEWithLogitsLoss()
    # === loss 超参数 ===
    supcon_temperature = 0.07      # SupCon 的温度
    supcon_weight      = 0.1       # λ_sup，建议 0.05~0.2
    use_supcon         = True      # 若想快速关闭，对这里改 False 即可

    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3

    distributed = False
    local_rank = -1
    num_workers = 16
    seed = 42
    world_size = None
    rank = None
    amp = False

    gpu_id = '0'
    batch_size = 64
    epochs = 30

    max_steps = 15000         
    eval_freq = 600
    save_freq = 600         
    min_lr = 1e-7            
    earlystop_epoch = 5     
    earlystop_delta = 0.01     

    print_interval = 20       
    val_interval = 30          
    save_interval = 10         

    threshold = 0.5            
    only_test_and_save_figs = True
    metric_mode = 'ap'        
    latest_mode = 'ap'

    best_ckpt_path = 'results/MCDF_batch64_norm/checkpoints/best_ap.pth' 
    img_save_path = 'test_images/'

    # work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '/'
    work_dir = 'results/MCDF_test_norm/'

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])

    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    opt = 'AdamW'  # 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01 # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9 # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6 # default: 1e-6 – term added to the denominator to improve numerical stability 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'Adagrad':
        lr = 0.01 # default: 0.01 – learning rate
        lr_decay = 0 # default: 0 – learning rate decay
        eps = 1e-10 # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.00005 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability 
        weight_decay = 1e-5 # default: 0 – weight decay (L2 penalty) 
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 3e-4 # default: 1e-3 – learning rate
        adjust_lr = 2e-4
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-4 # default: 1e-2 – weight decay coefficient
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
    elif opt == 'Adamax':
        lr = 2e-3 # default: 2e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'ASGD':
        lr = 0.01 # default: 1e-2 – learning rate 
        lambd = 1e-4 # default: 1e-4 – decay term
        alpha = 0.75 # default: 0.75 – power for eta update
        t0 = 1e6 # default: 1e6 – point at which to start averaging
        weight_decay = 0 # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        momentum = 0 # default: 0 – momentum factor
        alpha = 0.99 # default: 0.99 – smoothing constant
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes 
    elif opt == 'SGD':
        lr = 0.01 # – learning rate
        momentum = 0.9 # default: 0 – momentum factor 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
        dampening = 0 # default: 0 – dampening for momentum
        nesterov = False # default: False – enables Nesterov momentum 
    
    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5 # – Period of learning rate decay.
        gamma = 0.5 # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150] # – List of epoch indices. Must be increasing.
        gamma = 0.1 # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99 #  – Multiplicative factor of learning rate decay.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        # T_max = 50 # – Maximum number of iterations. Cosine function period.
        eta_min = lr*0.05  # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1.  
    elif sch == 'ReduceLROnPlateau':
        mode = 'min' # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1 # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10 # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001 # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel' # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0 # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0 # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08 # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50 # – Number of iterations for the first restart.
        T_mult = 2 # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1. 
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20
