import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
os.environ['TORCH_HOME'] = '/data2/hulh/pretrain'
warnings.filterwarnings("ignore")
sys.path.append('..')
import time
import torch

# torch.manual_seed(2023)  # 2023
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CASIADataset, NISTDataset, COVERAGEDataset, ColumbiaDataset, IMDDataset, DEFACTODataset
from data.transforms import get_transforms
from model.upernet import UperNet
from utils.losses import BCEDicedLoss
from utils.utils import Logger, AverageMeter, cal_f1, cal_iou, cal_auc, calculate_metric_score


def train_epoch(epoch, data_set, model, criterion, optimizer, logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    losses = AverageMeter()
    aucs = AverageMeter()
    f1s = AverageMeter()

    train_loader = DataLoader(dataset=data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    training_process = tqdm(train_loader)
    for i, data in enumerate(training_process):
        if use_au:
            inputs, targets, _ = data
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
        else:
            inputs, targets = data
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
        if i > 0:
            training_process.set_description("Epoch %d -- Loss: %.4f, F1: %.4f" %
                                             (epoch, losses.avg.item(), f1s.avg.item()))

        outputs = model(inputs)

        if use_clf:
            labels = []
            for i in range(len(targets)):
                labels.append(torch.max(targets))
            labels = torch.Tensor(labels)
            labels = labels.cuda()
            scores = []
            for i in range(len(outputs)):
                scores.append(torch.max(outputs[i][0]))
            scores = torch.Tensor(scores)
            scores = scores.cuda()
            au_loss = torch.nn.BCELoss()
            au_loss = au_loss.cuda()
            loss = criterion(outputs, targets) + au_loss(scores, labels)
        else:
            loss = criterion(outputs, targets)
        f1 = cal_f1(outputs.data.cpu().detach().numpy(), targets.cpu().detach().numpy(), threshold=0.5)
        auc = cal_auc(outputs.data.cpu().detach().numpy(), targets.cpu().detach().numpy())

        losses.update(loss.cpu().detach(), inputs.size(0))
        f1s.update(f1, inputs.size(0))
        aucs.update(auc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Train:\t Loss:{0:.4f}\t F1:{1:.4f}\t".format(losses.avg, f1s.avg))

    logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'f1': format(f1s.avg.item(), '.4f'),
        # 'acc': format(accs.avg.item(), '.4f'),
        # 'auc': format(aucs.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    return losses.avg


def val_epoch(epoch, data_set, model, criterion, optimizer, logger=None):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    losses = AverageMeter()
    aucs = AverageMeter()
    f1s = AverageMeter()
    ious = AverageMeter()

    valildation_loader = DataLoader(dataset=data_set,
                                    batch_size=val_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
    val_process = tqdm(valildation_loader)
    start_time = time.time()
    for i, data in enumerate(val_process):
        if use_au:
            inputs, targets, _ = data
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
        else:
            inputs, targets = data
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
        if i > 0:
            val_process.set_description("Epoch %d -- Loss: %.4f, F1: %.4f, AUC: %.4f" %
                                        (epoch, losses.avg.item(), f1s.avg.item(), aucs.avg))
        with torch.no_grad():
            outputs = model(inputs)
            if use_clf:
                labels = []
                for i in range(len(targets)):
                    labels.append(torch.max(targets))
                labels = torch.Tensor(labels)
                labels = labels.cuda()
                scores = []
                for i in range(len(outputs)):
                    scores.append(torch.max(outputs[i][0]))
                scores = torch.Tensor(scores)
                scores = scores.cuda()
                au_loss = torch.nn.BCELoss()
                au_loss = au_loss.cuda()
                loss = criterion(outputs, targets) + au_loss(scores, labels)
            else:
                loss = criterion(outputs, targets)

        auc, f1, iou = calculate_metric_score(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy(),
                                              threshold=0.5,
                                              metric_name=None)

        losses.update(loss.cpu(), inputs.size(0))
        aucs.update(auc, inputs.size(0))
        f1s.update(f1, inputs.size(0))
        ious.update(iou, inputs.size(0))
    epoch_time = time.time() - start_time
    print("Test:\t Loss:{0:.4f}\t AUC:{1:.4f} \t F1:{2:.4f} \t IOU:{3:.4f} \t using:{4:.3f} minutes".
          format(losses.avg, aucs.avg, f1s.avg, ious.avg, epoch_time / 60))

    if logger is not None:
        logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg.item(), '.4f'),
            'f1': format(f1s.avg.item(), '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })

    return losses.avg, f1s.avg


if __name__ == '__main__':
    is_train = True
    checkpoint = False
    is_resize = False
    is_one_hot = False
    test_post = False
    self_aug = True
    self_random = False
    use_au = False
    use_clf = False
    use_crop_data = False
    feature_name = None
    crop_shape = [256, 384]
    start_epoch = 1
    epochs = start_epoch + 100
    lr = 1e-4  # default:1e-4
    batch_size = 32
    val_batch_size = 1
    num_workers = 4
    root_path = '/data2/hulh/dataset'
    data_name = 'CASIA'  # CASIA NIST COVERAGE Columbia IMD2020 DEFACTO
    model_name = f'convnext_b_upernet_{crop_shape[0]}_{crop_shape[1]}'

    if is_resize:
        model_name += '_rs'
    if use_crop_data:
        model_name += '_crop'
    if self_aug:
        model_name += '_aug6'
        if self_random:
            model_name += '_random'

    write_file = f'../output/{data_name}/logs/{model_name}'
    save_dir = f'../output/{data_name}/weights/{model_name}'
    model = UperNet(backbone='convnext_base_22k', num_classes=1, in_channels=3, use_edge=False,
                    pretrained=True, input_size=crop_shape[0])

    model_path = None
    # model_path = '/data2/hulh/workplace/ForgeryDemo/output/CASIA/weights/mae_b_upernet_224_224_finetune_rs_aug6/epoch_84_auc_0.9429.pth'

    if model_path is not None:
        # model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'], strict=False)
        # model.backbone.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'], strict=False)
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    loss_function = BCEDicedLoss(bce_weight=0.1)
    loss_function = loss_function.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    if is_train:
        print(f'Training in {data_name}!!!')
        if model_path is not None and checkpoint:
            optimizer.load_state_dict(torch.load(model_path, map_location='cpu')['optimizer'])
            start_epoch = torch.load(model_path, map_location='cpu')['epoch']
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if data_name == 'CASIA':
            training_data = CASIADataset(root_path=root_path, transform=get_transforms(data_type='train'),
                                         data_type='train', is_resize=is_resize, crop_shape=crop_shape,
                                         self_aug=self_aug, is_one_hot=is_one_hot, self_random=self_random,
                                         use_au=use_au, feature_name=feature_name)
            validation_data = CASIADataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                           data_type='test',
                                           is_resize=is_resize, crop_shape=crop_shape, use_au=use_au,
                                           is_one_hot=is_one_hot, feature_name=feature_name)
        elif data_name == 'NIST':
            training_data = NISTDataset(root_path=root_path, transform=get_transforms(data_type='train'),
                                        data_type='train',
                                        is_resize=is_resize, crop_shape=crop_shape, self_aug=self_aug,
                                        is_one_hot=is_one_hot, self_random=self_random, feature_name=feature_name,
                                        test_post=test_post)
            validation_data = NISTDataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                          data_type='test',
                                          is_one_hot=is_one_hot,
                                          is_resize=is_resize, crop_shape=crop_shape, feature_name=feature_name)
        elif data_name == 'COVERAGE':
            training_data = COVERAGEDataset(root_path=root_path, transform=get_transforms(data_type='train'),
                                            data_type='train',
                                            is_resize=is_resize, crop_shape=crop_shape, self_aug=self_aug,
                                            is_one_hot=is_one_hot, self_random=self_random, feature_name=feature_name)
            validation_data = COVERAGEDataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                              data_type='test',
                                              is_one_hot=is_one_hot, is_resize=is_resize, crop_shape=crop_shape,
                                              feature_name=feature_name)
        elif data_name == 'Columbia':
            training_data = ColumbiaDataset(root_path=root_path, transform=get_transforms(data_type='train'),
                                            data_type='train',
                                            is_resize=is_resize, crop_shape=crop_shape, self_aug=self_aug,
                                            is_one_hot=is_one_hot, self_random=self_random, feature_name=feature_name)
            validation_data = ColumbiaDataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                              data_type='test',
                                              is_one_hot=is_one_hot, is_resize=is_resize, crop_shape=crop_shape,
                                              feature_name=feature_name)
        elif data_name == 'IMD2020':
            training_data = IMDDataset(root_path=root_path, transform=get_transforms(data_type='train'),
                                       data_type='train',
                                       is_resize=is_resize, crop_shape=crop_shape, self_aug=self_aug,
                                       is_one_hot=is_one_hot, self_random=self_random, feature_name=feature_name)
            validation_data = IMDDataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                         data_type='test',
                                         is_one_hot=is_one_hot, is_resize=is_resize, crop_shape=crop_shape,
                                         feature_name=feature_name)
        elif data_name == 'DEFACTO':
            training_data = DEFACTODataset(root_path=root_path, transform=get_transforms(data_type='train'),
                                           data_type='train',
                                           is_resize=is_resize, crop_shape=crop_shape, self_aug=self_aug,
                                           is_one_hot=is_one_hot, self_random=self_random, feature_name=feature_name)
            validation_data = DEFACTODataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                             data_type='test',
                                             is_one_hot=is_one_hot, is_resize=is_resize, crop_shape=crop_shape,
                                             feature_name=feature_name)

        train_logger = Logger(model_name=write_file, header=['epoch', 'loss', 'f1', 'lr'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=2)
        max_val_acc = 0.
        for i in range(start_epoch, epochs):
            train_loss = train_epoch(epoch=i,
                                     data_set=training_data,
                                     model=model,
                                     criterion=loss_function,
                                     optimizer=optimizer,
                                     logger=train_logger)
            val_loss, val_acc = val_epoch(epoch=i,
                                          data_set=validation_data,
                                          model=model,
                                          criterion=loss_function,
                                          optimizer=optimizer,
                                          logger=train_logger)
            scheduler.step(val_loss)
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                save_states_path = os.path.join(save_dir, 'epoch_{0}_f1_{1:.4f}.pth'.format(i, val_acc))
                if torch.cuda.device_count() > 1:
                    states = {
                        'epoch': i + 1,
                        'state_dict': model.module.state_dict(),  # TODO, 加上module
                        'optimizer': optimizer.state_dict(),
                    }
                else:
                    states = {
                        'epoch': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                torch.save(states, save_states_path)

            print('Current auc:', val_acc, 'Best auc:', max_val_acc)
        save_model_path = os.path.join(save_dir, "last_model_file" + str(epochs - 1) + ".pth")
        if os.path.exists(save_model_path):
            os.system("rm " + save_model_path)
        torch.save(states, save_model_path)
    else:
        print(f'Validation in {data_name}!!!')
        if data_name == 'CASIA':
            validation_data = CASIADataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                           data_type='test', is_resize=is_resize, crop_shape=crop_shape, use_au=use_au,
                                           is_one_hot=is_one_hot, feature_name=feature_name)
        elif data_name == 'NIST':
            validation_data = NISTDataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                          data_type='test', is_one_hot=is_one_hot, is_resize=is_resize,
                                          crop_shape=crop_shape, feature_name=feature_name)
        elif data_name == 'COVERAGE':
            validation_data = COVERAGEDataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                              data_type='test', is_one_hot=is_one_hot, is_resize=is_resize,
                                              crop_shape=crop_shape, feature_name=feature_name)
        elif data_name == 'Columbia':
            validation_data = ColumbiaDataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                              data_type='test', is_one_hot=is_one_hot, is_resize=is_resize,
                                              crop_shape=crop_shape, eature_name=feature_name)
        elif data_name == 'IMD2020':
            validation_data = IMDDataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                         data_type='test', is_one_hot=is_one_hot, is_resize=is_resize,
                                         crop_shape=crop_shape,
                                         feature_name=feature_name)
        elif data_name == 'DEFACTO':
            validation_data = DEFACTODataset(root_path=root_path, transform=get_transforms(data_type='test'),
                                             data_type='test', is_one_hot=is_one_hot, is_resize=is_resize,
                                             crop_shape=crop_shape, feature_name=feature_name)
        val_epoch(epoch=0,
                  data_set=validation_data,
                  model=model,
                  criterion=loss_function,
                  optimizer=optimizer,
                  logger=None)
