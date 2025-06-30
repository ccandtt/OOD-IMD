import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
import os
from sklearn.metrics import average_precision_score, accuracy_score
from copy import deepcopy


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out,patch_mining_list = model(images)
        loss = criterion(out.squeeze(1), targets)
        # loss = loss_function(out, targets, patch_mining_list, criterion)

        """check training label"""

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step

def train_one_step(batch,
                   model,
                   criterion_cls,        # nn.BCEWithLogitsLoss()
                   criterion_sup,        # SupConLoss 或 None
                   lambda_sup,           # float，=0 则忽略 SupCon
                   optimizer, scheduler,
                   step, logger, config, writer):
    """
    兼容原有调用，仅多传 3 个参数：criterion_sup, lambda_sup, criterion_cls 改名
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    images, targets = batch
    images   = images.cuda(non_blocking=True).float()
    targets  = targets.cuda(non_blocking=True).float()

    # ----- 前向：拿 logits 与 feat -----
    logits, feats = model(images, return_feat=True)   # MCDF 已按前文修改
    logits = logits.squeeze(1)

    # ----- 主分类损失（BCE） -----
    loss_cls = criterion_cls(logits, targets)

    # ----- SupCon 损失（可选） -----
    if (criterion_sup is not None) and (lambda_sup > 0):
        loss_sup = criterion_sup(feats, targets.long())
    else:
        loss_sup = torch.tensor(0.0, device=loss_cls.device)

    # ----- 总损失 -----
    loss = loss_cls + lambda_sup * loss_sup

    # ----- 反向传播 -----
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 可选
    optimizer.step()
    # scheduler.step()

    # ----- 日志 -----
    step += 1
    now_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('loss/total', loss.item(),   global_step=step)
    writer.add_scalar('loss/bce',   loss_cls.item(), global_step=step)
    if lambda_sup > 0:
        writer.add_scalar('loss/sup', loss_sup.item(), global_step=step)

    if step % config.print_interval == 0:
        log_info = (f'train: step {step}, '
                    f'loss={loss.item():.4f} '
                    f'(bce={loss_cls.item():.4f}, sup={loss_sup.item():.4f}), '
                    f'lr={now_lr:.6f}')
        print(log_info)
        logger.info(log_info)

    return step


# def loss_function(out,targets,patch_mining_list, criterion):
#     loss_criterion = criterion(out, targets)
#     """
#     """
#     PRM1_out, PRM2_out, PRM3_out, PRM4_out = patch_mining_list[0], patch_mining_list[1], patch_mining_list[2], patch_mining_list[3]
#     loss_PRM1_out_val = loss_fn(PRM1_out, targets)
#     loss_PRM2_out_val = loss_fn(PRM2_out, targets)
#     loss_PRM3_out_val = loss_fn(PRM3_out, targets)
#     loss_PRM4_out_val = loss_fn(PRM4_out, targets)

#     return loss_criterion + loss_PRM1_out_val/2 + loss_PRM2_out_val/4 + loss_PRM3_out_val/8 + loss_PRM4_out_val/16

# def loss_fn(pred, mask): # Renamed from 'loss' to avoid conflict with total_loss variable
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
#     wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
#     pred_sigmoid = torch.sigmoid(pred) # Use a different variable name for sigmoid output
#     inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
#     union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1) / (union - inter + 1)
#     return (wbce + wiou).mean()


def calculate_acc(y_true, y_pred, thres):
    """
    在给定threshold时，分别计算真实集和假集的准确率，以及总体准确率
    """
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc

def find_best_threshold(y_true, y_pred):
    """
    We assume first half is real(0), second half is fake(1).
    Return the threshold that yields the best accuracy.
    """
    N = y_true.shape[0]
    if y_pred[0:N//2].max() <= y_pred[N//2:N].min():
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0
        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc
    return best_thres


# def val_one_epoch(test_loader,
#                     model,
#                     criterion, 
#                     epoch, 
#                     logger,
#                     config):
#     # switch to evaluate mode
#     model.eval()
#     preds = []
#    #  gts = []
#     loss_list = []
#     with torch.no_grad():
#         for data in tqdm(test_loader):
#             img, msk = data
#             img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

#             out,patch_mining_list = model(img)
#             # loss = criterion(out, targets)
#             loss = criterion(out.squeeze(1), msk)

#             loss_list.append(loss.item())
#             # gts.append(msk.squeeze(1).cpu().detach().numpy())
#             if type(out) is tuple:
#                 out = out[0]
#             out = out.squeeze(1).cpu().detach().numpy()
#             preds.append(out) 

#     if epoch % config.val_interval == 0:
#         preds = np.array(preds).reshape(-1)
#         gts = np.array(gts).reshape(-1)

#         y_pre = np.where(preds>=config.threshold, 1, 0)
#         y_true = np.where(gts>=0.5, 1, 0)

#         confusion = confusion_matrix(y_true, y_pre)
#         TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

#         accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
#         sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
#         specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
#         f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
#         miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

#         log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
#                 specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
#         print(log_info)
#         logger.info(log_info)

#     else:
#         log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
#         print(log_info)
#         logger.info(log_info)
    
#     return np.mean(loss_list)



def val_one_epoch(val_loader, model, criterion, step, logger, config):
    """
    Validate the model for one evaluation phase.
    Returns loss, AP, accuracy@0.5, recall, specificity, best-threshold-based metrics.
    """
    model.eval()
    y_true, y_pred, loss_list = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True).float()

            outputs, _ = model(images)
            loss = criterion(outputs.squeeze(1), labels)

            loss_list.append(loss.item())
            scores = outputs.sigmoid().flatten().tolist()

            y_pred.extend(scores)
            y_true.extend(labels.flatten().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_loss = np.mean(loss_list)

    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, thres=0.5)
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, thres=best_thres)

    log_msg = (
        f'[VAL] Step {step}, Loss: {avg_loss:.4f}, AP: {ap:.4f}, '
        f'Acc@0.5: {acc0:.4f}, Recall: {r_acc0:.4f}, Specificity: {f_acc0:.4f}, '
        f'Best_Thres: {best_thres:.3f}, Acc_best: {acc1:.4f}, '
        f'Recall_best: {r_acc1:.4f}, Specificity_best: {f_acc1:.4f}'
    )
    logger.info(log_msg)
    print(log_msg)

    return avg_loss, ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres



# def test_one_epoch(test_loader,
#                     model,
#                     criterion,
#                     logger,
#                     config,
#                     test_data_name=None):
#     # switch to evaluate mode
#     model.eval()
#     preds = []
#     gts = []
#     loss_list = []
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(test_loader)):
#             img, msk = data
#             img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

#             out, patch_mining_list = model(img)
#             loss = criterion(out.squeeze(1), msk)

#             loss_list.append(loss.item())
#             msk = msk.squeeze(1).cpu().detach().numpy()
#             gts.append(msk)
#             if type(out) is tuple:
#                 out = out[0]
#             out = out.squeeze(1).cpu().detach().numpy()
#             preds.append(out) 
#             if i % config.save_interval == 0:
#                 save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

#         preds = np.array(preds).reshape(-1)
#         gts = np.array(gts).reshape(-1)

#         y_pre = np.where(preds>=config.threshold, 1, 0)
#         y_true = np.where(gts>=0.5, 1, 0)

#         confusion = confusion_matrix(y_true, y_pre)
#         TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

#         accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
#         sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
#         specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
#         f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
#         miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

#         if test_data_name is not None:
#             log_info = f'test_datasets_name: {test_data_name}'
#             print(log_info)
#             logger.info(log_info)
#         log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
#                 specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
#         print(log_info)
#         logger.info(log_info)

#     return np.mean(loss_list)

def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
    model.eval()
    y_pred, y_true = [], []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc='Testing')):
            img, label = data
            img, label = img.cuda(), label.cuda().float()

            out, _ = model(img)
            loss = criterion(out.squeeze(1), label)
            loss_list.append(loss.item())

            scores = out.sigmoid().flatten().tolist()
            y_pred.extend(scores)
            y_true.extend(label.flatten().tolist())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    loss = np.mean(loss_list)

    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, thres=0.5)
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, thres=best_thres)

    if test_data_name:
        logger.info(f'[TEST SET: {test_data_name}]')

    logger.info(f'[TEST RESULT] Loss: {loss:.4f}, AP: {ap:.4f}, '
                f'Acc@0.5: {acc0:.4f}, Sensitivity: {r_acc0:.4f}, Specificity: {f_acc0:.4f}, '
                f'Best_Threshold: {best_thres:.3f}, Acc_best: {acc1:.4f}, '
                f'Sensitivity_best: {r_acc1:.4f}, Specificity_best: {f_acc1:.4f}')

    return loss, ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres
