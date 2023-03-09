import torch
import torch.nn as nn
import pandas as pd
from eval import eval_acc, eval_edit, eval_f1_score


def train_epoch(train_loader, model, loss_fn, optimizer, device='cuda'):
    model.train()
    model.training = True
    total_loss = 0
    for iter, sample in enumerate(train_loader):
        x = sample['feature'].to(device)
        y = sample['label'].to(device)
        outs = model(x)
        loss = 0.0
        if isinstance(outs, list):
          n = len(outs)
          for out in outs:
            loss += loss_fn(out, y) / n
        else:
          loss = loss_fn(outs, y)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total_loss /= (iter + 1)
    return total_loss


def val_epoch(val_loader, model, loss_fn, device='cuda'):
    # only batch size = 1
    model.eval()
    model.training = True
    total_loss = 0
    gts = []
    preds = []
    with torch.no_grad():
        for iter, sample in enumerate(val_loader):
            x = sample['feature'].to(device)
            y = sample['label'].to(device)

            outs = model(x)
            loss = 0.0
            if isinstance(outs, list):
                n = len(outs)
                pred = outs[-1][0].argmax(dim=0)
                pred = pred.detach().cpu().numpy()
                target = y[0].detach().cpu().numpy()
                gts.append(target)
                preds.append(pred)
                for out in outs:
                    loss += loss_fn(out, y) / n
            else:
                loss = loss_fn(outs, y)
                pred = outs[0].argmax(dim=0)
                pred = pred.detach().cpu().numpy()
                target = y[0].detach().cpu().numpy()
                gts.append(target)
                preds.append(pred)
            total_loss += loss
    total_loss /= (iter + 1)
    acc = eval_acc(preds, gts)
    f1s = eval_f1_score(preds, gts)
    edit = eval_edit(preds, gts)
    return total_loss, acc, f1s, edit


import copy


def get_lr(opt):
  for param_group in opt.param_groups:
    return param_group['lr']

def train_val(model, params):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]

    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    save_config = params["save_config"]
    save_best_score = params["best_score"] # Acc, F1, Edit


    history = {
        "lr": [],
        "train loss": [],
        "val loss": [],
        "acc": [],
        "F1": [],
        "edit": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0

    for epoch in range(num_epochs):

        train_loss = train_epoch(train_dl, model, loss_func, opt)
        history['train loss'].append(train_loss)
        val_loss, acc, f1s, edit = val_epoch(val_dl, model, loss_func)
        history['lr'].append(get_lr(opt))
        history['val loss'].append(val_loss)
        history['acc'].append(acc)
        history['F1'].append(f1s[-1])
        history['edit'].append(edit)
        if save_best_score == 'acc':
            if acc > best_score:
                best_score = f1s[-1]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights + 'model_best.pth')
                print("Saved best model weights!")
        elif save_best_score == 'f1':
            if f1s[-1] > best_score:
                best_score = f1s[-1]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights + 'model_best.pth')
                print("Copied best model weights!")
        elif save_best_score == 'edit':
            if edit > best_score:
                best_score = f1s[-1]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights + 'model_best.pth')
                print("Saved best model weights!")
        lr_scheduler.step()

        print(
            'Epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}\tval loss:{:.4f}\t Acc:{:.4f}\t F1:{:.4f}\t Edit:{:.4f}'.format(
                epoch, get_lr(opt),
                train_loss, val_loss,
                acc, f1s[-1], edit))
        if epoch % save_config == 0:
            torch.save(model.state_dict(), path2weights + f'model_{epoch}.pth')
    pd.to_pickle(history, path2weights + 'log.pkl')
    torch.save(model.state_dict(), path2weights + 'model_final.pth')
    model.load_state_dict(best_model_wts)
    return model, history


def Evaluation(model, test_dl, device='cuda'):
  model.eval()
  model.training = False
  # n_total = 0
  # n_correct = 0
  # edit_score = 0
  # n_video = 0
  gts = []
  preds = []
  with torch.no_grad():
    for sample in test_dl:
      x = sample['feature'].to(device)
      y = sample['label']
      batch_size = x.shape[0]
      out = model(x)
      pred = out[0].argmax(dim=0)
      pred = pred.detach().cpu().numpy()
      target = y[0].numpy()
      # p_label, p_start, p_end = get_segments(pred, {0:'background',1:'back swing', 2:'down swing', 3:'follow through'})
      # g_label, g_start, g_end = get_segments(target, {0:'background',1:'back swing', 2:'down swing', 3:'follow through'})
      gts.append(target)
      preds.append(pred)
      # edit_score += levenshtein(p_label, g_label, norm=True)
      # n_correct += (pred==target).sum().item()
      # n_total += len(pred)
      # n_video += 1
  acc = eval_acc(preds, gts)
  f1s = eval_f1_score(preds, gts)
  edit = eval_edit(preds, gts)
  return  acc, f1s, edit
