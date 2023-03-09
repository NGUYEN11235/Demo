from Trainer import train_val, Evaluation
from model import ST_GCN, Multi_Stage_ST_CGN
from Loss import Loss_fcn
from Dataset import ActionData
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch import optim
import yaml

if __name__  == '__main__':
    with open(r'cfg.yaml') as file:
        config = yaml.full_load(file)

        # Model Config
        model_config = config['Model']
        if model_config['Name'] == 'ST_GCN':
            in_channels = model_config['inchannels']
            n_features = model_config['n_features']
            n_classes = model_config['n_classes']
            n_layers = model_config['n_layers']
            model = ST_GCN(in_channels, n_features, n_classes, n_layers)
            model.to('cuda')

        # Dataset Config
        Dataset_config = config['Dataset']
        train_ds = ActionData(Dataset_config['train_path'], augmentation=True)
        val_ds = ActionData(Dataset_config['val_path'])

        train_dl = DataLoader(train_ds, batch_size=Dataset_config['batch size'])
        val_dl = DataLoader(val_ds, batch_size=1)

        criterion = Loss_fcn()
        training_config = config['Training config']
        opt_config = training_config['Optimizers']
        if opt_config['Name'] == 'Adam':
            opt = optim.Adam(model.parameters(), lr=opt_config['lr'], weight_decay=1e-5)
        elif opt_config['Name'] == 'SGD':
            opt = optim.SGD(model.parameters(), lr=opt_config['lr'] ,weight_decay=1e-5, momentum=0.9, nesterov=True)
        lr_scheduler_config = training_config['lr_scheduler']
        if lr_scheduler_config['Name'] == 'Step':
            lr_scheduler = StepLR(opt, step_size=lr_scheduler_config['step size'], gamma=lr_scheduler_config['factor'])
        elif lr_scheduler_config['Name'] == 'Multi-Step':
            lr_scheduler = MultiStepLR(opt, milestones=lr_scheduler_config['step'], gamma=lr_scheduler_config['factor'])

        train_para = {
            "num_epochs": training_config['Epoch'],
            "optimizer": opt,
            "loss_func": criterion,
            "train_dl": train_dl,
            "val_dl": val_dl,
            "lr_scheduler": lr_scheduler,
            "save_config": training_config['save checkpoint'],
            "best_score":training_config['save best score'],
            "path2weights": training_config['path2save']}
    print('-'*20)
    print('---Training---')
    model, history = train_val(model, train_para)
    print('---Evalution---')
    acc, f1, edit = Evaluation(model, val_dl)
    print('Accuracy: ', acc)
    print('F1@(25,50,75) = ', f1)
    print('Edit:', edit)
