from __future__ import unicode_literals, print_function, division
import numpy as np
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
import torch.nn as nn
import os
from torch import optim
import torch.nn.functional as F
import time
import math
import json
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
from SAAM.model import *
from utils import *
from datasets import *
from evaluate import *
from optimization import *
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

logger = logging.getLogger('ConvTrans.Train')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Transformer on Time series forecasting')
parser.add_argument('--path', default='', type=str,
                    help='Time series data path')
parser.add_argument('--outdir', default='', type=str,
                    help='The name of saved model')
parser.add_argument('--input-size', default=5, type=int,
                    help='input_size (default: 5 = (4 covariates + 1 dim point))')
parser.add_argument('--train-ins-num', default=500000, type=int,
                    help='num of train instances (default: 500K)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--eval_batch_size', default=-1, type=int,
                    help='eval_batch_size default is equal to training batch_size')
parser.add_argument('--n_head', default=8, type=int,
                    help='n_head (default: 8)')
parser.add_argument('--num-layers', default=3, type=int,
                    help='num-layers (default: 3)')
parser.add_argument('--epoch', default=200, type=int,
                    help='epoch (default: 20)')
parser.add_argument('--embedded_dim', default=20, type=int,
                    help=' The dimention of Position embedding and time series ID embedding')
parser.add_argument('--pred-samples', default=200, type=int,
                    help='series samples during prediction (default: 200)')
parser.add_argument('--print-freq', default=100, type=int,
                    help='print-freq (default: 100)')
parser.add_argument('--lr',default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight_decay',default=0, type=float,
                    help='weight_decay')
parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--overlap',default=False, action='store_true',
                    help='If we overlap prediction range during sampling')
parser.add_argument('--scale_att',default=False, action='store_true',
                    help='Scaling Attention')
parser.add_argument('--sparse',default=False, action='store_true',
                    help='Perform the simulation of sparse attention ')
parser.add_argument('--pred_days', default=7, type=int,
                    help='How many days to predict after the train_end date')
parser.add_argument('--seed', default=0, type=int,
                    help='set random seed')
parser.add_argument('--enc_len', default=168, type=int,
                    help='The lengh of training range (encoder part)')
parser.add_argument('--dec_len', default=24, type=int,
                    help='The lengh of test range (decoder part)')
parser.add_argument('--acc_steps', default=1, type=int,
                    help='The steps to accumulate gradient)')
parser.add_argument("--dataset", default='',type=str,
                    help="Dataset you want to train")
parser.add_argument('--v_partition',default=0.1, type=float,
                    help='validation_partition')
parser.add_argument('--q_len',default=1, type=int,
                    help='kernel size for generating key-query')
parser.add_argument('--early_stop_ep',default=200, type=int,
                    help='early_stop_ep')
parser.add_argument('--pred_samples',default=200, type=int,
                    help='samples of prediction')
parser.add_argument('--sub_len',default=1, type=int,
                    help='sub_len of sparse attention')
parser.add_argument('--warmup_proportion',default=-1, type=float,
                    help='warmup_proportion for BERT Adam')
parser.add_argument('--optimizer',default="Adam", type=str,
                    help='Choice BERTAdam or Adam')

def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def plot_epoch(args, h, h_filtered, epoch, text):
    random.seed()
    idx = random.randint(0, h.shape[0]-1)
    n_features = h_filtered.shape[-1]
    emb_features_plotting = list(range(n_features)) if n_features < 10 else sorted(random.sample(range(n_features), 10))
    plt.close("all")
    rows = len(emb_features_plotting)
    cols = 1
    rows_names = range(0, rows)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for ax, row in zip(axes[:], rows_names):
        ax.set_ylabel("Dim " + str(rows_names[row]))
    for row in range(rows):
        axes[row].plot(h[idx, :, row].cpu().data.numpy(), 'b', linestyle="dotted")
        axes[row].plot(h_filtered[idx, :, row].cpu().data.numpy(), 'r')
    fig.suptitle("Emb. hidden dims. filtering", verticalalignment="bottom")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "epoch_" + str(epoch) + "_" + str(text) +".pdf"))
    plt.close()

def set_logger(log_path):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    _logger = logging.getLogger('ConvTrans')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))

def save_checkpoint(state, epoch , loss_is_best, filename):
    torch.save(state, filename+'/'+str(epoch)+'_v_loss.pth.tar')
    if(loss_is_best):
        torch.save(state, filename+'/best_v_loss.pth.tar')

def valid(train_seq,gt,series_id,scaling_factor, encoder):

    with torch.no_grad():
        mu, sigma, h, h_filtered = encoder(series_id,train_seq)
        scaling_factor = torch.unsqueeze(torch.unsqueeze(scaling_factor,dim=1),dim=1)
        criterion = GaussianLoss(scaling_factor*mu,scaling_factor*sigma)
        gt = torch.unsqueeze(gt,dim=2)
        loss = criterion(gt)

    return loss.item(), h, h_filtered


def validIters(valid_loader,encoder):
    encoder.eval()
    total_loss = 0
    for id, data in enumerate(valid_loader):
        train_seq,gt,series_id, scaling_factor = data  # (batch_size,seq, value) --> (seq,batch_size,value)
        train_seq = train_seq.to(torch.float32).to(device)
        gt = gt.to(torch.float32).to(device)
        series_id = series_id.to(device)
        scaling_factor = scaling_factor.to(torch.float32).to(device)
        loss, h, h_filtered = valid(train_seq,gt,series_id,scaling_factor,encoder)
        total_loss += loss
    loss = total_loss/(id+1)
    return loss, h, h_filtered #printeo el ultimo h_filtered que es el que se quedará guardado

def trainIters(args,train_loader,valid_loader,test_loader,encoder,weight_decay, epochs, print_every, learning_rate, outdir,pred_samples):


    writer = SummaryWriter(outdir)
    start = time.time()
    if(args.optimizer=="Adam"):
        logger.info("Vanilla Adam")
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate,weight_decay=weight_decay)
    elif(args.optimizer=="BERTAdam"):
        logger.info("BERTAdam")
        num_train_steps = int(len(train_loader)/args.acc_steps * epochs) # This is equal to default Adam.
        encoder_optimizer = BERTAdam(encoder.parameters(),
                         lr=learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps,
                         weight_decay_rate=args.weight_decay
                         )
    else:
        raise NameError('Currently, we only support Adam and BERTAdam')

    epoch_len = len(train_loader)
    tr_loss = 0
    global_step = 0
    iter_th = 0
    batch_num = len(train_loader)
    n_iters = epochs*batch_num
    logger.info('Start training')
    loss_best = float('inf')
    best_epoch = -1

    best_test_ND = float('inf')
    train_len = len(train_loader)
    ND_summary = np.zeros(epochs)

    for epoch in range(epochs):
        encoder.train()
        for step, data in enumerate(train_loader):

            train_seq,gt,series_id, scaling_factor = data
            train_seq = train_seq.to(torch.float32).to(device)
            gt = gt.to(torch.float32).to(device)
            series_id = series_id.to(device)
            scaling_factor = scaling_factor.to(torch.float32).to(device)
            mu, sigma, h, h_filtered = encoder(series_id,train_seq)
            #Plot here
            scaling_factor = torch.unsqueeze(torch.unsqueeze(scaling_factor,dim=1),dim=1)
            criterion = GaussianLoss(scaling_factor*mu,scaling_factor*sigma)
            gt = torch.unsqueeze(gt,dim=2)
            loss = criterion(gt)
            loss = loss/args.acc_steps
            loss.backward()
            tr_loss += loss.item()
            iter_th += 1
            if  (step+1)%args.acc_steps ==0:
                if(args.optimizer=="Adam"): # Since BERTAdam already performs gradient clipping, we only perform gradient clipping for Adam.
                    nn.utils.clip_grad_norm_(encoder.parameters(),1)
                encoder_optimizer.step()
                encoder.zero_grad()
                global_step += 1
                writer.add_scalar('training/train_loss',tr_loss,global_step)
                tr_loss = 0

            if (iter_th+1) % (print_every+1) == 0:
                logger.info('%s (%d %d%%)' % (timeSince(start, iter_th / n_iters),
                                     iter_th, iter_th / n_iters * 100))

            if (step == 0):
                plot_epoch(args, h, h_filtered, epoch, "valid")
            
        logger.info ("EPOCH "+str(epoch+1))
        v_loss, h, h_filtered = validIters(valid_loader,encoder)
        plot_epoch(args, h, h_filtered, epoch, "train")
        writer.add_scalar('training/validation_loss',v_loss,epoch+1)
        if(v_loss < loss_best):
            loss_best = v_loss
            loss_is_best = True
            best_epoch = epoch
        else:
            loss_is_best = False

        predictions = evaluateIters(test_loader, args.enc_len, args.dec_len, encoder, pred_samples)
        ND_metrics = ND_Metrics(predictions, test_loader)
        writer.add_scalar('training/ND_error', ND_metrics, epoch + 1)
        RMSE_metrics = RMSE_Metrics(predictions, test_loader)
        writer.add_scalar('training/RMSE_error', RMSE_metrics, epoch + 1)
        rou_metrics_09 = Rou_Risk(0.9, predictions, test_loader, pred_samples)
        writer.add_scalar('training/Rou_09', rou_metrics_09, epoch + 1)
        rou_metrics_05 = Rou_Risk(0.5, predictions, test_loader, pred_samples)
        writer.add_scalar('training/Rou_05', rou_metrics_05, epoch + 1)
        logger.info('ND: {}'.format(ND_metrics))
        logger.info('RMSE: {}'.format(RMSE_metrics))
        logger.info('rou-90: {}'.format(rou_metrics_09))
        logger.info('rou-50: {}'.format(rou_metrics_05))

        ND_summary[epoch] = ND_metrics
        is_best = ND_summary[epoch] <= best_test_ND
        test_metrics = {'ND': ND_metrics, 'RMSE': RMSE_metrics, 'rou-90': rou_metrics_09, 'rou-50': rou_metrics_05}

        if is_best:
            logger.info('****** Found new best ND')
            best_test_ND = ND_summary[epoch]
            best_json_path = os.path.join(outdir, 'metrics_test_best_weights.json')
            save_dict_to_json(test_metrics, best_json_path)
        logger.info('Current Best ND is: %.5f' % best_test_ND)
        last_json_path = os.path.join(outdir, 'metrics_test_last_weights.json')
        save_dict_to_json(test_metrics, last_json_path)

        logger.info('Current v loss: {}, Best v loss: {}'.format(v_loss, loss_best))
        if(epoch-best_epoch>=args.early_stop_ep):
            logger.info("Achieve early_stop_ep and current epoch is: {}".format(epoch))
            break
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': encoder.state_dict(),
            'optimizer' : encoder_optimizer.state_dict(),
        },epoch+1,loss_is_best,outdir)

    encoder.load_state_dict(torch.load(outdir+'/best_v_loss.pth.tar')['state_dict'])
    predictions = evaluateIters(test_loader, args.enc_len, args.dec_len, encoder, pred_samples)
    ND_metrics = ND_Metrics(predictions, test_loader)
    writer.add_scalar('training/ND_error', ND_metrics, epoch + 1)
    RMSE_metrics = RMSE_Metrics(predictions, test_loader)
    writer.add_scalar('training/RMSE_error', RMSE_metrics, epoch + 1)
    rou_metrics_09 = Rou_Risk(0.9, predictions, test_loader, pred_samples)
    writer.add_scalar('training/Rou_09', rou_metrics_09, epoch + 1)
    rou_metrics_05 = Rou_Risk(0.5, predictions, test_loader, pred_samples)
    writer.add_scalar('training/Rou_05', rou_metrics_05, epoch + 1)
    logger.info('ND: {}'.format(ND_metrics))
    logger.info('RMSE: {}'.format(RMSE_metrics))
    logger.info('rou-90: {}'.format(rou_metrics_09))
    logger.info('rou-50: {}'.format(rou_metrics_05))

    """
    encoder.load_state_dict(torch.load(outdir+'/best_v_loss.pth.tar')['state_dict'])
    predictions = evaluateIters(test_loader,args.enc_len,args.dec_len,encoder,pred_samples)
    ND_metrics = ND_Metrics(predictions,test_loader)
    RMSE_metrics = RMSE_Metrics(predictions,test_loader)
    rou_metrics = Rou_Risk(0.9,predictions,test_loader,pred_samples)
    logger.info('ND: ',ND_metrics)
    logger.info('RMSE: ',RMSE_metrics)
    logger.info('rou-90: ',rou_metrics)
    """

    return 0



def main():

    global args
    args = parser.parse_args()
    np.random.seed(0) # Fix training and validation set
    num_train = args.train_ins_num
    indices = list(range(num_train))
    split = int(args.v_partition* num_train)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    try:
        os.mkdir(args.outdir)
    except FileExistsError:
        pass

    set_logger(os.path.join(args.outdir, 'train.log'))
    logger.info(torch.__version__)
    logger.info("TIME RIGHT NOW!: " + str(datetime.now()))
    logger.info('Loading the datasets...')


    if(args.dataset=='traffic'):
        train_set = Traffic(args.path,args.train_ins_num,overlap = args.overlap ,pred_days=args.pred_days,win_len= args.enc_len+args.dec_len)
        test_set = TrafficTest(train_set.points,train_set.covariates,train_set.withhold_len,args.enc_len,args.dec_len)
    elif(args.dataset=='traffic_fine'):
        train_set = Traffic_fine(args.path,args.train_ins_num,overlap = args.overlap ,pred_days=args.pred_days,win_len= args.enc_len+args.dec_len)
        test_set = Traffic_fineTest(train_set.points,train_set.covariates,train_set.withhold_len,args.enc_len,args.dec_len)
    elif(args.dataset=='ele'):
        train_set = Ele(args.path,args.train_ins_num,overlap = args.overlap,pred_days=args.pred_days,win_len= args.enc_len+args.dec_len)
        test_set = EleTest(train_set.points,train_set.covariates,train_set.withhold_len,args.enc_len,args.dec_len)
    elif(args.dataset=='ele_fine'):
        train_set = Ele_fine(args.path,args.train_ins_num,overlap = args.overlap,pred_days=args.pred_days,win_len= args.enc_len+args.dec_len)
        test_set = Ele_fineTest(train_set.points,train_set.covariates,train_set.withhold_len,args.enc_len,args.dec_len)
    elif(args.dataset=='m4'):
        train_set = M4(args.path,args.train_ins_num,overlap = args.overlap,pred_days=args.pred_days,win_len= args.enc_len+args.dec_len)
        test_set = M4Test(train_set.points,train_set.covariates,train_set.withhold_len,args.enc_len,args.dec_len)
    elif(args.dataset=='wind'):
        train_set = Wind(args.path,args.train_ins_num,overlap = args.overlap,pred_days=args.pred_days,win_len= args.enc_len+args.dec_len)
        test_set = WindTest(train_set.points,train_set.covariates,train_set.withhold_len,args.enc_len,args.dec_len)
    elif(args.dataset=='solar'):
        train_set = Solar(args.path,args.train_ins_num,overlap = args.overlap,pred_days=args.pred_days,win_len= args.enc_len+args.dec_len)
        test_set = SolarTest(train_set.points,train_set.covariates,train_set.withhold_len,args.enc_len,args.dec_len)
    else:
        raise NameError('Currently, we only support traffic, ele, m4, solar and wind')
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, sampler =train_sampler)
    if(args.eval_batch_size ==-1):
        eval_batch_size = args.batch_size
    else:
        eval_batch_size = args.eval_batch_size

    valid_loader = DataLoader(train_set, batch_size=eval_batch_size, num_workers=4, drop_last=True,sampler =valid_sampler)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size,num_workers=4)

    encoder = DecoderTransformer(args,input_dim = args.input_size, n_head= args.n_head, layer= args.num_layers, seq_num = train_set.seq_num , n_embd = args.embedded_dim,win_len= args.enc_len+args.dec_len)
    encoder = nn.DataParallel(encoder).to(device)
    logger.info(args)
    logger.info(encoder)
    trainIters(args,train_loader,valid_loader,test_loader,encoder,args.weight_decay,args.epoch, print_every=args.print_freq, learning_rate=args.lr, outdir = args.outdir, pred_samples = args.pred_samples)

if __name__ == '__main__':
    main()
