from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import sys
import os
import torch
from matplotlib import pyplot as plt
import imgaug.augmenters as iaa

from .metrics import *
from .dataloaders import Nuclei
from .dataloaders import MRIDataset
from .SIDN_model import SIDN


import pytorch_version.SIDN_model
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data_ddp(args, rank, world_size):
    """
    load data and create data loaders with DistributedSampler
    """
    # augmentations
    transforms = iaa.Sequential([
        iaa.Rotate((-5., 5.)),
        iaa.TranslateX(percent=(-0.05, 0.05)),
        iaa.TranslateY(percent=(-0.05, 0.05)),
        iaa.Affine(shear=(-10, 10)),
        iaa.Affine(scale=(0.8, 1.2)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])

    # load data
    train_set = Nuclei(args.train_data, 'monuseg', batchsize=args.batch_size, transforms=transforms)
    test_set = Nuclei(args.valid_data, args.valid_dataset)

    # create samplers
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)

    # create data loaders
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, sampler=train_sampler, drop_last=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, sampler=test_sampler,
                             num_workers=8,
                             pin_memory=True, persistent_workers=True)

    return train_set, test_set, train_loader, test_loader


def load_data(args):
    """
    load data and create data loaders
    """
    # augmentations
    transforms = iaa.Sequential([
        iaa.Rotate((-5., 5.)),
        iaa.TranslateX(percent=(-0.05, 0.05)),
        iaa.TranslateY(percent=(-0.05, 0.05)),
        iaa.Affine(shear=(-10, 10)),
        iaa.Affine(scale=(0.8, 1.2)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])

    # load data and create data loaders
    train_set = Nuclei(args.train_data, args.data_name, batchsize=args.batch_size, transforms=transforms)
    test_set = Nuclei(args.valid_data, args.valid_dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)

    return train_set, test_set, train_loader, test_loader


# 加载 MRI数据集，由命令行参数指定具体器官，然后加载对应器官的数据
def load_MRI(args):
    """
    load data and create data loaders
    """
    # augmentations
    transforms = iaa.Sequential([
        # iaa.Rotate((-5., 5.)),
        # iaa.TranslateX(percent=(-0.05, 0.05)),
        # iaa.TranslateY(percent=(-0.05, 0.05)),
        # iaa.Affine(shear=(-10, 10)),
        # iaa.Affine(scale=(0.8, 1.2)),
        # iaa.Fliplr(0.5),
        # iaa.Flipud(0.5)
    ])
    # 这里应该要划分5折交叉验证，一个MRIDataset包含全部数据
    # load data and create data loaders
    # train_set = MRIDataset(args.train_data, args.data_name, batch_size=args.batch_size, transform=transforms)
    # test_set = MRIDataset(args.valid_data, args.valid_dataset)
    # train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
    #                           drop_last=True, num_workers=8, pin_memory=True)
    # test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
    #                          shuffle=False, num_workers=8, pin_memory=True)
    k_folders = []
    dataset = MRIDataset(args.train_data, args.data_name,
                         batch_size=args.batch_size, transform=transforms)
    kf = KFold(n_splits=5, shuffle=True, random_state=24)

    for train_idx, test_idx in kf.split(np.arange(len(dataset))):
        train_subset = Subset(dataset, indices=train_idx)
        test_subset = Subset(dataset, indices=test_idx)
        train_loader = DataLoader(dataset=train_subset, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(dataset=test_subset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=8, pin_memory=True)
        k_folders.append((train_subset, test_subset, train_loader, test_loader))

    return k_folders


def create_model(args):
    # device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    if args.gene is None:  
        model = SIDN(in_channel=args.in_channel,
                       iterations=args.iter,
                       num_classes=args.num_class,
                       multiplier=args.multiplier,
                       num_layers=args.num_layers,
                       acti=args.activate,
                       is_skipATT=args.is_skipATT)
    print('generated model:', type(model))

    return model

    # return Unet(1, 1)



def compute_metrics(x, y, tot_iou, tot_dice, args):
    """
    compute accuracy metrics: IoU, DICE
    """
    if args.num_class == 1:
        iou_score = iou(y, x)
        dice_score = dice_coef(y, x)
    else:
        iou_score = miou(y, x, args.num_class)
        dice_score = mdice(y, x, args.num_class)
    tot_iou += iou_score
    tot_dice += dice_score
    return tot_iou, tot_dice


def get_flops(args, model, new_network=None):
    """
    compute computations: MACs, #Params
    """
    if new_network is not None:
        model.reset_gene(new_network, BFS(args.iter, new_network))

    from ptflops import get_model_complexity_info
    save_stdout = sys.stdout
    sys.stdout = open('./trash', 'w')
    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=False,
                                             print_per_layer_stat=False, verbose=False)
    sys.stdout = save_stdout
    print("macs: %.4f x 10^9, num params: %.4f x 10^6" % (float(macs) * 1e-9, float(params) * 1e-6))
    return macs, params


def save_model(args, model):
    weights = '_weights'
    cpt_name = 'iter_' + str(args.iter) + '_mul_' + str(args.multiplier) + '_best' + weights + '.pt'
    save_path = args.save_gene + '/checkpoint'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'state_dict': model.state_dict()}, save_path + "/" + cpt_name)


def save_model_MRI(args, model, fold):
    weights = '_weights'
    cpt_name = 'iter_' + str(args.iter) + '_mul_' + str(args.multiplier) + '_best' + weights + '.pt'
    save_path = args.save_gene + f'_fold{fold}/checkpoint'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'state_dict': model.state_dict()}, save_path + "/" + cpt_name)


def save_metrics(val_loss, val_iou, val_dice, args):
    save_path = args.save_gene + '/checkpoint/outputs'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + '/result.txt', 'w+') as f:
        f.write('Validation loss:\t' + str(val_loss) + '\n')
        f.write('Validation  iou:\t' + str(val_iou) + '\n')
        f.write('Validation dice:\t' + str(val_dice) + '\n')
    print('Metrics have been saved to:', save_path + '/result.txt')


def save_metrics_MRI(val_loss, val_iou, val_dice, args, fold):
    save_path = args.save_gene + f'_fold{fold}/checkpoint/outputs'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + '/result.txt', 'w+') as f:
        f.write('Validation loss:\t' + str(val_loss) + '\n')
        f.write('Validation  iou:\t' + str(val_iou) + '\n')
        f.write('Validation dice:\t' + str(val_dice) + '\n')
    print('Metrics have been saved to:', save_path + '/result.txt')


def save_masks(segmentations, inputs, gts, args):
    """
    save segmentation masks
    :segmentations: predictions
    :inputs: input images
    "gts: ground truth
    """
    results = np.transpose(np.concatenate(segmentations, axis=0), (0, 2, 3, 1))
    inputs = np.transpose(np.concatenate(inputs, axis=0), (0, 2, 3, 1))
    gts = np.concatenate(gts, axis=0)
    if len(gts.shape) == 4:
        gts = np.transpose(gts, (0, 2, 3, 1))
    if args.num_class == 1:
        results = (results > 0.5).astype(np.float32)  # Binarization. Comment out this line if you don't want to
    else:
        r_map = {0: 0., 1: 1., 2: 0., 3: 1., 4: 0.}
        g_map = {0: 0., 1: 0., 2: 1., 3: 1., 4: 0.}
        b_map = {0: 0., 1: 0., 2: 1., 3: 0., 4: 1.}
        ph = np.zeros(results.shape[:3] + (3,))
        gt_ph = np.zeros(results.shape[:3] + (3,))
        results = np.argmax(results, axis=-1)
        for b in range(results.shape[0]):
            for i in range(results.shape[1]):
                for j in range(results.shape[2]):
                    if results[b, i, j] > 0:
                        ph[b, i, j, 0] = r_map[results[b, i, j]]
                        ph[b, i, j, 1] = g_map[results[b, i, j]]
                        ph[b, i, j, 2] = b_map[results[b, i, j]]
                    if gts[b, i, j] > 0:
                        gt_ph[b, i, j, 0] = r_map[gts[b, i, j]]
                        gt_ph[b, i, j, 1] = g_map[gts[b, i, j]]
                        gt_ph[b, i, j, 2] = b_map[gts[b, i, j]]
        results = ph

    print('Saving segmentations...')
    save_path = args.save_gene + '/checkpoint/outputs/segmentations/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(results.shape[0]):
        if args.num_class == 1:
            plt.imsave(save_path + str(i) + ".png", results[i, :, :, 0], cmap='gray')  # binary segmenation
            plt.imsave(save_path + str(i) + "_gt.png", gts[i, :, :, 0], cmap='gray')
            plt.imsave(save_path + str(i) + "_input.png", inputs[i, :, :])
        else:
            plt.imsave(save_path + str(i) + ".png", results[i, :, :, :])
            plt.imsave(save_path + str(i) + "_gt.png", gt_ph[i, :, :, :])
            plt.imsave(save_path + str(i) + "_input.png", inputs[i, :, :, 0], cmap='gray')
    print('A total of ' + str(results.shape[0]) + ' segmentation results have been saved to:', save_path)


def save_masks_MRI(segmentations, inputs, gts, args, fold):
    """
    save segmentation masks
    :segmentations: predictions
    :inputs: input images
    "gts: ground truth
    """
    results = np.transpose(np.concatenate(segmentations, axis=0), (0, 2, 3, 1))
    inputs = np.transpose(np.concatenate(inputs, axis=0), (0, 2, 3, 1))
    gts = np.concatenate(gts, axis=0)
    if len(gts.shape) == 4:
        gts = np.transpose(gts, (0, 2, 3, 1))
    if args.num_class == 1:
        results = (results > 0.5).astype(np.float32)  # Binarization. Comment out this line if you don't want to
    else:
        r_map = {0: 0., 1: 1., 2: 0., 3: 1., 4: 0.}
        g_map = {0: 0., 1: 0., 2: 1., 3: 1., 4: 0.}
        b_map = {0: 0., 1: 0., 2: 1., 3: 0., 4: 1.}
        ph = np.zeros(results.shape[:3] + (3,))
        gt_ph = np.zeros(results.shape[:3] + (3,))
        results = np.argmax(results, axis=-1)
        for b in range(results.shape[0]):
            for i in range(results.shape[1]):
                for j in range(results.shape[2]):
                    if results[b, i, j] > 0:
                        ph[b, i, j, 0] = r_map[results[b, i, j]]
                        ph[b, i, j, 1] = g_map[results[b, i, j]]
                        ph[b, i, j, 2] = b_map[results[b, i, j]]
                    if gts[b, i, j] > 0:
                        gt_ph[b, i, j, 0] = r_map[gts[b, i, j]]
                        gt_ph[b, i, j, 1] = g_map[gts[b, i, j]]
                        gt_ph[b, i, j, 2] = b_map[gts[b, i, j]]
        results = ph

    print('Saving segmentations...')
    save_path = args.save_gene + f'_fold{fold}/checkpoint/outputs/segmentations/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('inputs.shape is ', inputs.shape)
    for i in range(results.shape[0]):
        if args.num_class == 1:
            plt.imsave(save_path + str(i) + ".png", results[i, :, :, 0], cmap='gray')  # binary segmenation
            plt.imsave(save_path + str(i) + "_gt.png", gts[i, :, :, 0], cmap='gray')
            plt.imsave(save_path + str(i) + "_input.png", inputs[i, :, :, 0], cmap='gray')
        else:
            plt.imsave(save_path + str(i) + ".png", results[i, :, :, :])
            plt.imsave(save_path + str(i) + "_gt.png", gt_ph[i, :, :, :])
            plt.imsave(save_path + str(i) + "_input.png", inputs[i, :, :, 0], cmap='gray')
    print('A total of ' + str(results.shape[0]) + ' segmentation results have been saved to:', save_path)


def get_skip(model):
    """
    """
    encoder = []
    decoder = []
    for name, module in model.named_modules():
        if isinstance(module, pytorch_version.SIDN.Block):
            topk = torch.topk(module.alpha, 1, dim=0)[1].cpu().numpy()
            topk = sorted(topk.tolist()[0])
            if module.alpha.shape[-1] == 2:  # last decoder
                topk = [0] + [v + 1 for v in topk]

            if name[:7] == 'encoder':
                encoder.append(topk)
            else:
                decoder.append(topk)
    return encoder, decoder


def encode(args):

    if args.model_path is None:
        weights = '_weights'
        cpt_name = 'iter_' + str(args.iter) + '_mul_' + str(args.multiplier) + '_best' + weights + '.pt'
        model_path = args.save_gene + '/checkpoint/' + cpt_name
    else:
        model_path = args.save_gene
    print('Restoring model from path: ' + model_path)

    checkpoint = torch.load(model_path)
    model = create_model(args)

    new_dict = OrderedDict()
    print(type(checkpoint['state_dict']), len(checkpoint['state_dict']))
    for k, v in checkpoint['state_dict'].items():
        name = k.replace('module.', '')
        new_dict[name] = v
    checkpoint['state_dict'] = new_dict

    model.load_state_dict(checkpoint['state_dict'])
    level = args.num_layers

    if not args.save_gene:
        # if do not specify the path to save  genes
        if not os.path.exists("./_genes/"):
            os.mkdir("./_genes/")
        if args.gene is None:
            encoder, decoder = get_skip(model)
            with open('./_genes/' + '_gene_' + args.exp + '.txt', 'w+') as f:
                f.write('%.3f_%f\n' % (0.0, 0.0))
                for it in range(len(encoder) // level):
                    for l in range(level):
                        print('enc', encoder[it * level + l])
                        f.write('enc_' + str(it) + '_' + str(l))
                        for i in encoder[it * level + l]:
                            f.write('_' + str(i))
                        f.write('\n')
                    for l in range(level):
                        print('dec', decoder[it * level + l])
                        f.write('dec_' + str(it) + '_' + str(l))
                        for i in decoder[it * level + l]:
                            f.write('_' + str(i))
                        f.write('\n')
    else:
        encoder, decoder = get_skip(model)
        dir_name = args.save_gene
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(args.save_gene + '/' + '_gene_' + args.exp + '.txt', 'w+') as f:
            f.write('%.3f_%f\n' % (0.0, 0.0))
            for it in range(len(encoder) // level):
                for l in range(level):
                    print('enc', encoder[it * level + l])
                    f.write('enc_' + str(it) + '_' + str(l))
                    for i in encoder[it * level + l]:
                        f.write('_' + str(i))
                    f.write('\n')
                for l in range(level):
                    print('dec', decoder[it * level + l])
                    f.write('dec_' + str(it) + '_' + str(l))
                    for i in decoder[it * level + l]:
                        f.write('_' + str(i))
                    f.write('\n')


def encode_MRI(args, fold):
    if args.model_path is None:
        weights = '_weights'
        cpt_name = 'iter_' + str(args.iter) + '_mul_' + str(args.multiplier) + '_best' + weights + '.pt'
        model_path = args.save_gene + f'_fold{fold}/checkpoint/' + cpt_name
    else:
        model_path = args.save_gene
    print('Restoring model from path: ' + model_path)

    checkpoint = torch.load(model_path)
    model = create_model(args)

    new_dict = OrderedDict()
    print(type(checkpoint['state_dict']), len(checkpoint['state_dict']))
    for k, v in checkpoint['state_dict'].items():
        name = k.replace('module.', '')
        new_dict[name] = v
    checkpoint['state_dict'] = new_dict

    model.load_state_dict(checkpoint['state_dict'])
    level = args.num_layers

    if not args.save_gene:
        if not os.path.exists("./_genes/"):
            os.mkdir("./_genes/")
        if args.gene is None:
            encoder, decoder = get_skip(model)
            with open('./_genes/' + '_gene_' + args.exp + '.txt', 'w+') as f:
                f.write('%.3f_%f\n' % (0.0, 0.0))
                for it in range(len(encoder) // level):
                    for l in range(level):
                        print('enc', encoder[it * level + l])
                        f.write('enc_' + str(it) + '_' + str(l))
                        for i in encoder[it * level + l]:
                            f.write('_' + str(i))
                        f.write('\n')
                    for l in range(level):
                        print('dec', decoder[it * level + l])
                        f.write('dec_' + str(it) + '_' + str(l))
                        for i in decoder[it * level + l]:
                            f.write('_' + str(i))
                        f.write('\n')
    else:
        encoder, decoder = get_skip(model)
        dir_name = args.save_gene
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(args.save_gene + '/' + '_gene_' + args.exp + '.txt', 'w+') as f:
            f.write('%.3f_%f\n' % (0.0, 0.0))
            for it in range(len(encoder) // level):
                for l in range(level):
                    print('enc', encoder[it * level + l])
                    f.write('enc_' + str(it) + '_' + str(l))
                    for i in encoder[it * level + l]:
                        f.write('_' + str(i))
                    f.write('\n')
                for l in range(level):
                    print('dec', decoder[it * level + l])
                    f.write('dec_' + str(it) + '_' + str(l))
                    for i in decoder[it * level + l]:
                        f.write('_' + str(i))
                    f.write('\n')


def BFS(iters, codes):
    """
    check if a block is skipped or not
    BFS from the last extraction stage to the first one
    """
    new_codes = []

    pre_iter = [0, 1, 2, 3] # last block at last extraction stage
    compensate = [[] for _ in range(iters * 2)]

    for i in range(iters * 2 - 1, 0, -1): # for each extraction stage pair
        temp = {}
        temp_code = []
        for level in pre_iter: # for each non-skipped block 
            for out in codes[i][level]: # for each outgoing skip

                if i % 2== 0: # if BFS from encoder -> decoder
                    if out == 0: # if the skip is sequential 
                        if level == 0: 
                            continue
                        else:
                            pre_iter.append(level-1) 
                    else: # if not sequential
                        temp[out - 1] = 1
                        
                else: # if BFS from decoder -> encoder
                    if out == 0: # if the skip is sequential 
                        if level == 0: # if has bridge
                            temp[3] = 1 
                        else:
                            pre_iter.append(level-1)
                    else: # if not sequential
                        temp[out - 1] = 1

        for k in temp.keys():
            temp_code.append(k)
        temp_code = list(set(temp_code))
        pre_iter = temp_code[:] # to be searched in next round
        new_codes.append(pre_iter) # append non-skipped blocks at preceding stage to new_codes

    new_codes = [[0, 1, 2, 3]]+ new_codes #[:4] + [[0, 1, 2, 3]]
    new_codes = new_codes[::-1]

    skip_codes = [[True for _ in range(4)] for _ in range(iters*2)]
    for i in range(iters*2):
        new_codes[i] = list(set(new_codes[i]))
        for code in new_codes[i]: 
            skip_codes[i][code] = False
    print('These blocks as skipped: ', skip_codes)
    return skip_codes