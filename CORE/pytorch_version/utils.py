from torch.utils.data import DataLoader
import sys
import os
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch
from tqdm import tqdm
from .metrics import *
from .dataloaders import Nuclei
from .tools import *
from time import perf_counter
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

torch.autograd.set_detect_anomaly(True)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.manual_seed(1)
# np.random.seed(1)
# random.seed(1)
# torch.cuda.manual_seed_all(1)
# torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def setup(rank, world_size):
    # print('set up 1')
    os.environ['MASTER_ADDR'] = 'localhost'
    # print('set up 2')
    os.environ['MASTER_PORT'] = '2077'
    # print('set up 3')
    print(f'rank{rank}, word_size{world_size}')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # print('set up 4')


def cleanup():
    dist.destroy_process_group()


def train(args):
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    print('Using device: ', device)
    # record metrics
    epoch_metrics = {
        'epoch': list(range(1, args.epochs + 1)),  
        'tot_loss': [],
        'tot_iou': [],
        'tot_dice': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }

    # load data
    train_set, test_set, train_loader, test_loader = load_data(args)

    # create model
    model = create_model(args)
    model = model.to(device).float()
    # define criterion
    criterion = BCELoss() if args.num_class == 1 else CrossEntropyLoss(weight=train_set.CLASS_WEIGHTS)
    # create optimizer
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.)
    # keras lr decay equivalent
    lr_decay = 3e-3
    fcn = lambda step: 1. / (1. + lr_decay * step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)

    print('model successfully built and compiled.')

    # make dictionary for saving results
    if not os.path.exists("./checkpoints/" + args.exp + "/outputs"):
        # mkdir in linux way
        # os.mkdir("./checkpoints/"+args.exp)
        # os.mkdir("./checkpoints/"+args.exp+"/outputs")
        # mkdir in win way
        os.makedirs("./checkpoints/" + args.exp)
        os.makedirs("./checkpoints/" + args.exp + "/outputs")

    best_iou = 0.
    best_dice = 0.
    print('\nStart training...')
    start = perf_counter()
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch} / {args.epochs}')
        tot_loss = 0.
        tot_iou = 0.
        tot_dice = 0.
        val_loss = 0.
        val_iou = 0.
        val_dice = 0.

        # training
        model.train()
        for step, ret in tqdm(enumerate(train_loader),
                              desc='[TRAIN] Epoch ' + str(epoch + 1) + '/' + str(args.epochs),
                              total=len(train_loader)):
            # if step >= 1:
            #     break

            x = ret['x'].to(device)
            y = ret['y'].to(device)

            optimizer.zero_grad()
            output = model(x)

            # loss
            l = criterion(output, y)
            tot_loss += l.item()
            l.backward()
            optimizer.step()

            # compute metrics
            x, y = output.detach().cpu().numpy(), y.detach().cpu().numpy()
            tot_iou, tot_dice = compute_metrics(x, y, tot_iou, tot_dice, args)

        scheduler.step()

        print('[TRAIN] Epoch: ' + str(epoch + 1) + '/' + str(args.epochs),
              'loss:', tot_loss / len(test_loader),
              'iou:', tot_iou / len(test_loader),
              'dice:', tot_dice / len(test_loader))

        epoch_metrics['tot_loss'].append(tot_loss)
        epoch_metrics['tot_iou'].append(tot_iou)
        epoch_metrics['tot_dice'].append(tot_dice)

        # validation
        model.eval()
        with torch.no_grad():
            for step, ret in enumerate(
                    tqdm(test_loader, desc='[VAL] Epoch ' + str(epoch + 1) + '/' + str(args.epochs), disable=True)):
                x = ret['x'].to(device)
                y = ret['y'].to(device)

                output = model(x)

                # loss
                l = criterion(output, y)
                val_loss += l.item()

                # compute metrics
                x, y = output.detach().cpu().numpy(), y.cpu().numpy()
                val_iou, val_dice = compute_metrics(x, y, val_iou, val_dice, args)
                print('val_iou compute:', val_iou)

        val_iou = val_iou / len(test_loader)
        val_dice = val_dice / len(test_loader)

        # save model
        if val_iou >= best_iou:
            best_iou = val_iou
            best_dice = val_dice
            save_model(args, model)
            
            with open(args.save_gene + '/best_iou_and_dice.txt', 'w+') as f:
                f.write('Best  iou:\t' + str(best_iou) + '\n')
                f.write('Best dice:\t' + str(best_dice) + '\n')
        
        print('[VAL] Epoch: ' + str(epoch + 1) + '/' + str(args.epochs),
              'val_loss:', val_loss / len(test_loader),
              'val_iou:', val_iou,
              'val_dice:', val_dice,
              'best val_iou:', best_iou)

        epoch_metrics['val_loss'].append(val_loss)
        epoch_metrics['val_iou'].append(val_iou)
        epoch_metrics['val_dice'].append(val_dice)

    end = perf_counter()
    print('\nTraining fininshed!')
    print('\nBest model saved to', args.save_gene + '/checkpoint')
    print(f'\nBest iou:{best_iou}')
    print(f'\nTraining cost {end - start} seconds')
    save_metric(epoch_metrics, args.save_gene)


# 训练MRI数据集的函数，使用5-折交叉验证，
def train_MRI(args):
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    print('Using device: ', device)
    # record metrics
    epoch_metrics = {
        'epoch': list(range(1, args.epochs + 1)),  # 从1到300的列表
        'tot_loss': [],
        'tot_iou': [],
        'tot_dice': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }

    # load data, k 折交叉验证，返回k个下面这个的组合
    k_folder = load_MRI(args)


    # create model
    model = create_model(args)
    model = model.to(device).float()
    print('model successfully built and compiled.')

    # 开始k次training
    for k in range(5):
        # if k == 0 or k == 1 == 2:
        #     continue
        fold = k_folder[k]
        # automatically unpack
        train_set, test_set, train_loader, test_loader = fold
        print('train set len:', len(train_set))
        print('test set len:', len(test_set))
        save_path = args.save_gene + f'_fold{k}'
        # define criterion
        criterion = BCELoss() if args.num_class == 1 else CrossEntropyLoss(weight=train_set.CLASS_WEIGHTS)
        # create optimizer
        optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.)
        # keras lr decay equivalent
        lr_decay = 3e-3
        fcn = lambda step: 1. / (1. + lr_decay * step)
        scheduler = LambdaLR(optimizer, lr_lambda=fcn)

        # make dictionary for saving results
        if not os.path.exists("./checkpoints/" + args.exp + "/outputs"):
            # mkdir in linux way
            # os.mkdir("./checkpoints/"+args.exp)
            # os.mkdir("./checkpoints/"+args.exp+"/outputs")
            # mkdir in win way
            os.makedirs("./checkpoints/" + args.exp)
            os.makedirs("./checkpoints/" + args.exp + "/outputs")

        best_iou = 0.
        best_dice = 0.
        print(f'\nFold {k} Start training...')
        start = perf_counter()
        # training
        model.train()
        for epoch in range(args.epochs):
            print(f'\nFold {k} Epoch {epoch} / {args.epochs}')
            tot_loss = 0.
            tot_iou = 0.
            tot_dice = 0.
            val_loss = 0.
            val_iou = 0.
            val_dice = 0.


            for step, ret in tqdm(enumerate(train_loader),
                                  desc=f'Fold {k}: [TRAIN] Epoch{epoch + 1} / {args.epochs}',
                                  total=len(train_loader)):

                x = ret['x'].to(device)
                y = ret['y'].to(device)

                optimizer.zero_grad()
                output = model(x)

                # loss
                l = criterion(output, y)
                tot_loss += l.item()
                l.backward()
                optimizer.step()

                # compute metrics
                x, y = output.detach().cpu().numpy(), y.detach().cpu().numpy()
                tot_iou, tot_dice = compute_metrics(x, y, tot_iou, tot_dice, args)

            scheduler.step()

            print(f'Fold {k}: [TRAIN] Epoch{epoch + 1} / {args.epochs}',
                  'loss:', tot_loss / len(test_loader),
                  'iou:', tot_iou / len(test_loader),
                  'dice:', tot_dice / len(test_loader))

            epoch_metrics['tot_loss'].append(tot_loss)
            epoch_metrics['tot_iou'].append(tot_iou)
            epoch_metrics['tot_dice'].append(tot_dice)

            # validation
            model.eval()
            with torch.no_grad():
                for step, ret in enumerate(
                        tqdm(test_loader, desc='[VAL] Epoch ' + str(epoch + 1) + '/' + str(args.epochs), disable=True)):
                    x = ret['x'].to(device)
                    y = ret['y'].to(device)

                    output = model(x)

                    # loss
                    l = criterion(output, y)
                    val_loss += l.item()

                    # compute metrics
                    x, y = output.detach().cpu().numpy(), y.cpu().numpy()
                    val_iou, val_dice = compute_metrics(x, y, val_iou, val_dice, args)
                    print('val_iou compute:', val_iou)

            val_iou = val_iou / len(test_loader)
            val_dice = val_dice / len(test_loader)

            # save model
            if val_iou >= best_iou:
                best_iou = val_iou
                best_dice = val_dice
                save_model_MRI(args, model, k)

            print('[VAL] Epoch: ' + str(epoch + 1) + '/' + str(args.epochs),
                  'val_loss:', val_loss / len(test_loader),
                  'val_iou:', val_iou,
                  'val_dice:', val_dice,
                  'best val_iou:', best_iou)

            epoch_metrics['val_loss'].append(val_loss)
            epoch_metrics['val_iou'].append(val_iou)
            epoch_metrics['val_dice'].append(val_dice)

        end = perf_counter()
        print(f'\nFold {k} Training finished!')
        print('\nBest model saved to', save_path + '/checkpoint')
        print(f'\nBest iou:{best_iou}')
        print(f'\nBest dice:{best_dice}')
        print(f'\nTraining cost {end - start} seconds')

        with open(save_path + '/best_iou_and_dice.txt', 'w+') as f:
            f.write('Best  iou:\t' + str(best_iou) + '\n')
            f.write('Best dice:\t' + str(best_dice) + '\n')
        
        for key in epoch_metrics:
            if key != 'epoch':
                epoch_metrics[key].clear()

        # 每个fold后验证一次
        # validation
        weights = '_weights'
        cpt_name = 'iter_' + str(args.iter) + '_mul_' + str(args.multiplier) + '_best' + weights + '.pt'
        model_path = save_path + '/checkpoint/' + cpt_name
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # cpu

        new_dict = OrderedDict()
        print(type(checkpoint['state_dict']), len(checkpoint['state_dict']))
        for key, v in checkpoint['state_dict'].items():
            name = key.replace('module.', '')
            new_dict[name] = v
        checkpoint['state_dict'] = new_dict

        model.load_state_dict(checkpoint['state_dict'], strict=True)
        segmentations = []
        inputs = []
        gts = []

        print(f'\nFold {k} Start evaluation...')
        model.eval()
        with torch.no_grad():
            for step, ret in enumerate(tqdm(test_loader)):

                x = ret['x'].to(device)
                y = ret['y'].to(device)
                input_x = x.cpu().numpy()
                output = model(x)

                # loss
                l = criterion(output, y)
                val_loss += l.item()

                # compute metrics
                x, y = output.detach().cpu().numpy(), y.cpu().numpy()
                val_iou, val_dice = compute_metrics(x, y, val_iou, val_dice, args)

                # save predictions
                if args.save_result:
                    segmentations.append(x)
                    inputs.append(input_x)
                    gts.append(y)

        val_iou = val_iou / len(test_loader)
        val_dice = val_dice / len(test_loader)
        val_loss = val_loss / len(test_loader)

        print('Validation loss:\t', val_loss)
        print('Validation  iou:\t', val_iou)
        print('Validation dice:\t', val_dice)

        # compute computations
        # get_flops(args, model)
        print('\nEvaluation finished!')

        # save results
        if args.save_result:
            save_metrics_MRI(val_loss, val_iou, val_dice, args, k)
            save_masks_MRI(segmentations, inputs, gts, args, k)

        if args.Phase1:
            encode_MRI(args, k)


'''
train in parallel mode
'''
def train_parallel(args):
    # device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    device_ids = [2, 3, 4, 5]
    print('Using devices: ', device_ids)
    # record metrics
    epoch_metrics = {
        'epoch': list(range(1, args.epochs + 1)),  # 从1到300的列表
        'tot_loss': [],
        'tot_iou': [],
        'tot_dice': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }

    # load data
    train_set, test_set, train_loader, test_loader = load_data(args)

    # create model
    model = create_model(args)
    model = model.to(f'cuda:{device_ids[0]}').float()
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    # define criterion
    criterion = BCELoss() if args.num_class == 1 else CrossEntropyLoss(weight=train_set.CLASS_WEIGHTS)
    # create optimizer
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.)
    # keras lr decay equivalent
    lr_decay = 3e-3
    fcn = lambda step: 1. / (1. + lr_decay * step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)

    print('model successfully built and compiled.')

    # make dictionary for saving results
    if not os.path.exists("./checkpoints/" + args.exp + "/outputs"):
        os.makedirs("./checkpoints/" + args.exp)
        os.makedirs("./checkpoints/" + args.exp + "/outputs")

    best_iou = 0.
    print('\nStart training...')
    start = perf_counter()
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch} / {args.epochs}')
        tot_loss = 0.
        tot_iou = 0.
        tot_dice = 0.
        val_loss = 0.
        val_iou = 0.
        val_dice = 0.

        # training
        model.train()
        for step, ret in tqdm(enumerate(train_loader),
                              desc='[TRAIN] Epoch ' + str(epoch + 1) + '/' + str(args.epochs),
                              total=len(train_loader)):
            x = ret['x'].to(device)
            y = ret['y'].to(device)

            optimizer.zero_grad()
            output = model(x)

            # loss
            l = criterion(output, y)
            tot_loss += l.item()
            l.backward()
            optimizer.step()

            # compute metrics
            x, y = output.detach().cpu().numpy(), y.detach().cpu().numpy()
            tot_iou, tot_dice = compute_metrics(x, y, tot_iou, tot_dice, args)

        scheduler.step()

        print('[TRAIN] Epoch: ' + str(epoch + 1) + '/' + str(args.epochs),
              'loss:', tot_loss / len(test_loader),
              'iou:', tot_iou / len(test_loader),
              'dice:', tot_dice / len(test_loader))

        epoch_metrics['tot_loss'].append(tot_loss)
        epoch_metrics['tot_iou'].append(tot_iou)
        epoch_metrics['tot_dice'].append(tot_dice)

        # validation
        model.eval()
        with torch.no_grad():
            for step, ret in enumerate(
                    tqdm(test_loader, desc='[VAL] Epoch ' + str(epoch + 1) + '/' + str(args.epochs), disable=True)):
                x = ret['x'].to(device)
                y = ret['y'].to(device)

                output = model(x)

                # loss
                l = criterion(output, y)
                val_loss += l.item()

                # compute metrics
                x, y = output.detach().cpu().numpy(), y.cpu().numpy()
                val_iou, val_dice = compute_metrics(x, y, val_iou, val_dice, args)
                print('val_iou compute:', val_iou)

        val_iou = val_iou / len(test_loader)
        val_dice = val_dice / len(test_loader)

        # save model
        if val_iou >= best_iou:
            best_iou = val_iou
            save_model(args, model)

        print('[VAL] Epoch: ' + str(epoch + 1) + '/' + str(args.epochs),
              'val_loss:', val_loss / len(test_loader),
              'val_iou:', val_iou,
              'val_dice:', val_dice,
              'best val_iou:', best_iou)

        epoch_metrics['val_loss'].append(val_loss)
        epoch_metrics['val_iou'].append(val_iou)
        epoch_metrics['val_dice'].append(val_dice)

    end = perf_counter()
    print('\nTraining fininshed!')
    print('\nBest model saved to', args.save_gene + '/checkpoint')
    print(f'\nBest iou:{best_iou}')
    print(f'\nTraining cost {end - start} seconds')
    save_metric(epoch_metrics, args.save_gene)



def train_DDP(rank, args, world_size):
    # print('setting up')
    setup(rank, world_size)
    # print('setup done')
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    epoch_metrics = {
        'epoch': list(range(1, args.epochs + 1)),  # 从1到300的列表
        'tot_loss': [],
        'tot_iou': [],
        'tot_dice': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }

    # load data
    train_set, test_set, train_loader, test_loader = load_data_ddp(args, rank, world_size)

    print('In tarin ddp...')

    model = create_model(args).to(rank).float()
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    print('model successfully built and compiled.')
    # make dictionary for saving results
    path = "./checkpoints/" + args.exp + "/outputs"
    if not os.path.exists(path):
        # os.makedirs("./checkpoints/" + args.exp)
        os.makedirs(path)

    # define criterion
    criterion = BCELoss() if args.num_class == 1 else CrossEntropyLoss(weight=train_set.CLASS_WEIGHTS)
    # create optimizer
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.)
    # keras lr decay equivalent
    lr_decay = 3e-3
    fcn = lambda step: 1. / (1. + lr_decay * step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)

    print('model successfully built and compiled.')

    best_iou = 0.
    print(f'\nStart training on rank {rank}...')
    start = perf_counter()
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch} / {args.epochs}')
        tot_loss = 0.
        tot_iou = 0.
        tot_dice = 0.
        val_loss = 0.
        val_iou = 0.
        val_dice = 0.

        # training
        model.train()
        for step, ret in tqdm(enumerate(train_loader),
                              desc='[TRAIN] Epoch ' + str(epoch + 1) + '/' + str(args.epochs),
                              total=len(train_loader)):
            x = ret['x'].to(device)
            y = ret['y'].to(device)

            optimizer.zero_grad()
            output = model(x)

            # loss
            l = criterion(output, y)
            tot_loss += l.item()
            l.backward()
            optimizer.step()

            # compute metrics
            x, y = output.detach().cpu().numpy(), y.detach().cpu().numpy()
            tot_iou, tot_dice = compute_metrics(x, y, tot_iou, tot_dice, args)

        scheduler.step()

        print('[TRAIN] Epoch: ' + str(epoch + 1) + '/' + str(args.epochs),
              'loss:', tot_loss / len(test_loader),
              'iou:', tot_iou / len(test_loader),
              'dice:', tot_dice / len(test_loader))

        epoch_metrics['tot_loss'].append(tot_loss)
        epoch_metrics['tot_iou'].append(tot_iou)
        epoch_metrics['tot_dice'].append(tot_dice)

        # validation
        if rank == 0:
            model.eval()
            with torch.no_grad():
                for step, ret in enumerate(
                        tqdm(test_loader, desc='[VAL] Epoch ' + str(epoch + 1) + '/' + str(args.epochs), disable=True)):
                    x = ret['x'].to(device)
                    y = ret['y'].to(device)

                    output = model(x)

                    # loss
                    l = criterion(output, y)
                    val_loss += l.item()

                    # compute metrics
                    x, y = output.detach().cpu().numpy(), y.cpu().numpy()
                    val_iou, val_dice = compute_metrics(x, y, val_iou, val_dice, args)
                    print('val_iou compute:', val_iou)

            val_iou = val_iou / len(test_loader)
            val_dice = val_dice / len(test_loader)

            # save model
            if val_iou >= best_iou:
                best_iou = val_iou
                save_model(args, model)

            print('[VAL] Epoch: ' + str(epoch + 1) + '/' + str(args.epochs),
                  'val_loss:', val_loss / len(test_loader),
                  'val_iou:', val_iou,
                  'val_dice:', val_dice,
                  'best val_iou:', best_iou)

            epoch_metrics['val_loss'].append(val_loss)
            epoch_metrics['val_iou'].append(val_iou)
            epoch_metrics['val_dice'].append(val_dice)

    end = perf_counter()
    if rank == 0:
        print('\nTraining finished!')
        print('\nTraining fininshed!')
        print('\nBest model saved to', args.save_gene + '/checkpoint')
        print(f'\nBest iou:{best_iou}')
        print(f'\nTraining cost {end - start} seconds')
        save_metric(epoch_metrics, args.save_gene)

    cleanup()


def evaluate(args):
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print('Using device: ', device)
    # load data and create data loader
    if args.dataset == 'nuclei':
        test_set = Nuclei(args.valid_data, args.valid_dataset)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 pin_memory=True)

    else:
        print('Invalid dataset!')

    # load best model
    if args.model_path is None:
        weights = '_weights'
        cpt_name = 'iter_' + str(args.iter) + '_mul_' + str(args.multiplier) + '_best' + weights + '.pt'
        model_path = args.save_gene + '/checkpoint/' + cpt_name
    else:
        model_path = args.model_path
    print('Restoring model from path: ' + model_path)

    # create model
    model = create_model(args)
    model.to(device)
    # checkpoint = torch.load(model_path) # cuda
    print(os.getcwd())
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # cpu

    new_dict = OrderedDict()
    print(type(checkpoint['state_dict']), len(checkpoint['state_dict']))
    for k, v in checkpoint['state_dict'].items():
        name = k.replace('module.', '')
        new_dict[name] = v
    checkpoint['state_dict'] = new_dict

    model.load_state_dict(checkpoint['state_dict'], strict=True)

    criterion = BCELoss() if args.num_class == 1 else CrossEntropyLoss()

    val_loss = 0.
    val_iou = 0.
    val_dice = 0.

    segmentations = []
    inputs = []
    gts = []

    print('\nStart evaluation...')
    model.eval()
    with torch.no_grad():
        for step, ret in enumerate(tqdm(test_loader)):

            x = ret['x'].to(device)
            y = ret['y'].to(device)
            input_x = x.cpu().numpy()
            output = model(x)

            # loss
            l = criterion(output, y)
            val_loss += l.item()

            # compute metrics
            x, y = output.detach().cpu().numpy(), y.cpu().numpy()
            val_iou, val_dice = compute_metrics(x, y, val_iou, val_dice, args)

            # save predictions
            if args.save_result:
                segmentations.append(x)
                inputs.append(input_x)
                gts.append(y)

    val_iou = val_iou / len(test_loader)
    val_dice = val_dice / len(test_loader)
    val_loss = val_loss / len(test_loader)

    print('Validation loss:\t', val_loss)
    print('Validation  iou:\t', val_iou)
    print('Validation dice:\t', val_dice)

    # compute computations
    get_flops(args, model)
    print('\nEvaluation finished!')

    # save results
    if args.save_result:
        save_metrics(val_loss, val_iou, val_dice, args)
        save_masks(segmentations, inputs, gts, args)

    if args.Phase1:
        encode(args)



def save_metric(metrics, path):
    if not os.path.exists(path):
        os.makedirs(path)
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epoch'], metrics['tot_iou'], label='tot_iou', color='green')
    plt.plot(metrics['epoch'], metrics['tot_dice'], label='tot_dice', color='blue')
    plt.xlabel('epoch')
    plt.ylabel('train metric')
    plt.legend(loc='best')
    plt.title('Training Metrics over Epochs')
    plt.savefig(path + '/training_metrics.png')

    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epoch'], metrics['val_iou'], label='val_iou', color='green')
    plt.plot(metrics['epoch'], metrics['val_dice'], label='val_dice', color='blue')
    plt.xlabel('epoch')
    plt.ylabel('metric')
    plt.legend(loc='best')
    plt.title('Validation Metrics over Epochs')
    plt.savefig(path + '/validation_metrics.png')
    print(f'Save metrics to {path}/xxx_metrics.png')

    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epoch'], metrics['val_loss'], label='val_loss', color='red')
    plt.plot(metrics['epoch'], metrics['tot_loss'], label='train_loss', color='blue')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.title('Validation Metrics over Epochs')
    plt.savefig(path + '/validation_loss.png')