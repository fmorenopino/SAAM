"""
fmorenopino

Code based on DeepAR implementation.
"""

import argparse
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import SAAM.net as net
from evaluate_SAAM import evaluate
from dataloader import *
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import utils as utils
from tensorboardX import SummaryWriter
logger = logging.getLogger('SAAM.Train')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ele', help='Name of the dataset')
parser.add_argument('--path', default='', type=str, help='Time series data path')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='ele', help='Directory containing params.json')
parser.add_argument('--cuda-device', default='0', help='GPU device to use')
parser.add_argument('--save-weights', default=True, help='Whether to save best ND to param_search.txt')
parser.add_argument('--force-cpu', default=False, help='Whether to force cpu to run the code')
parser.add_argument('--iterations-b-evaluations', default=2000, help='Whether to sample during evaluation')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,help='Optional, name of the file in --model_dir containing weights to reload before training')  # 'best' or 'epoch_#'
parser.add_argument('--overlap',default=False, action='store_true',help='If we overlap prediction range during sampling')


def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader: DataLoader,
          test_loader: DataLoader,
          params: utils.Params,
          epoch: int, writer) -> float:
    '''Train the SAAM on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of SAAM
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        train_loader: load train data_ and labels
        test_loader: load test data_ and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()

    loss_epoch = np.zeros(len(train_loader))
    # Train_loader:
    # train_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
    # idx ([batch_size]): one integer denoting the time series id;
    # labels_batch ([batch_size, train_window]): z_{1:T}.
    for i, (train_batch, idx, labels_batch) in enumerate(tqdm(train_loader)):

        optimizer.zero_grad()
        batch_size = train_batch.shape[0]

        train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(params.device)  # not scaled
        labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(params.device)  # not scaled
        idx = idx.unsqueeze(0).to(params.device)

        #Global Frequency Characterization
        emb = model.getEmbedding(train_batch, idx).to(params.device)
        Rx = model.get_Rx(emb.permute(1, 0, 2)).to(params.device)
        PSD = model.get_FFT_(Rx).to(params.device)

        loss = torch.zeros(1, device=params.device)
        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)

        H = torch.Tensor().to(params.device)

        for t in range(params.train_window):
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (train_batch[t, :, 0] == 0)
            if t > 0 and torch.sum(zero_index) > 0:
                train_batch[t, zero_index, 0] = mu[zero_index]
            mu, sigma, hidden, cell, H, H_filtered, _, _, _ = model(train_batch[t].unsqueeze_(0).clone(), idx, hidden, cell, PSD, H)
            loss += loss_fn(mu, sigma, labels_batch[t])

        loss.backward()
        #plot_grad_flow(model.named_parameters(),os.path.join(params.plot_dir_gradients, "epoch_" + str(epoch) + ".pdf"))
        optimizer.step()
        loss = loss.item() / params.train_window  # loss per timestep
        loss_epoch[i] = loss
        writer.add_scalar('training/train_loss', loss, epoch + i)
        if (epoch % params.evaluate_every_x_epochs  == 0) and (i % params.iterations_b_evaluations == 0):
            test_metrics = evaluate(model, loss_fn, test_loader, params, epoch, sample=args.sampling)
            model.train()
            logger.info(f'train_loss: {loss}')

        if i == 0:
            logger.info(f'train_loss: {loss}')

    return loss_epoch


def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       optimizer: optim, loss_fn,
                       params: utils.Params,
                       restore_file: str = None) -> None:
    '''Train the SAAM and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the SAAM
        train_loader: load train data_ and labels
        test_loader: load test data_ and labels
        optimizer: (torch.optim) optimizer for parameters of SAAM
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    '''
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    logger.info('begin training and evaluation')
    best_test_ND = float('inf')
    train_len = len(train_loader)
    ND_summary = np.zeros(params.num_epochs)
    writer = SummaryWriter(params.model_dir)


    if (params.sampling):
        q50_summary = np.zeros(params.num_epochs)
        q90_summary = np.zeros(params.num_epochs)
        best_test_q50 = float('inf')
        best_test_q90 = float('inf')

    loss_summary = np.zeros((train_len * params.num_epochs))
    for epoch in range(params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader,
                                                                        test_loader, params, epoch, writer)
        test_metrics = evaluate(model, loss_fn, test_loader, params, epoch, sample=args.sampling)

        writer.add_scalar('training/test_loss', test_metrics['test_loss'], epoch)
        writer.add_scalar('training/ND', test_metrics['ND'], epoch)
        writer.add_scalar('training/RMSE', test_metrics['RMSE'], epoch)
        if (params.sampling):
            writer.add_scalar('training/Rp_05', test_metrics['rou50'], epoch)
            writer.add_scalar('training/Rp_09', test_metrics['rou90'], epoch)

        ND_summary[epoch] = test_metrics['ND']
        is_best = ND_summary[epoch] <= best_test_ND

        if (params.sampling):
             q50_summary[epoch] = test_metrics['rou50']
             q90_summary[epoch] = test_metrics['rou90']
             is_best_q50 = q50_summary[epoch] <= best_test_q50
             is_best_q90 = q50_summary[epoch] <= best_test_q90


        if (args.save_weights):
            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                  epoch=epoch,
                                  is_best=is_best,
                                  checkpoint=params.model_dir)

        if is_best:
            logger.info('****** Found new best ND')
            best_test_ND = ND_summary[epoch]
            best_json_path = os.path.join(params.model_dir, 'metrics_test_best_weights.json')
            utils.save_dict_to_json(test_metrics, best_json_path)

        if (params.sampling):
             if is_best_q50:
                 logger.info('- Found new best q50')
                 best_test_q50 = q50_summary[epoch]
                 best_json_path_q50 = os.path.join(params.model_dir, 'metrics_test_best_weights_q50.json')
                 utils.save_dict_to_json(test_metrics, best_json_path_q50)

             if is_best_q90:
                 logger.info('- Found new best q90')
                 best_test_q90 = q90_summary[epoch]
                 best_json_path_q90 = os.path.join(params.model_dir, 'metrics_test_best_weights_q90.json')
                 utils.save_dict_to_json(test_metrics, best_json_path_q90)



        logger.info('Current Best ND is: %.5f' % best_test_ND)
        if (params.sampling):
            logger.info('Current Best q50 is: %.5f' % best_test_q50)
            logger.info('Current Best q90 is: %.5f' % best_test_q90)




        utils.plot_all_epoch(ND_summary[:epoch + 1], args.dataset + '_ND', params.plot_dir)

        if (params.sampling):
            utils.plot_all_epoch(q50_summary[:epoch + 1], args.dataset + '_q50', params.plot_dir)
            utils.plot_all_epoch(q90_summary[:epoch + 1], args.dataset + '_q90', params.plot_dir)

        utils.plot_all_epoch(loss_summary[:(epoch + 1) * train_len], args.dataset + '_loss', params.plot_dir)

        last_json_path = os.path.join(params.model_dir, 'metrics_test_last_weights.json')
        utils.save_dict_to_json(test_metrics, last_json_path)

    if args.save_best:
        f = open('./param_search.txt', 'w')
        f.write('-----------\n')
        list_of_params = args.search_params.split(',')
        print_params = ''
        for param in list_of_params:
            param_value = getattr(params, param)
            print_params += f'{param}: {param_value:.2f}'
        print_params = print_params[:-1]
        f.write(print_params + '\n')
        f.write('Best ND: ' + str(best_test_ND) + '\n')
        logger.info(print_params)
        logger.info(f'Best ND: {best_test_ND}')
        f.close()
        utils.plot_all_epoch(ND_summary, print_params + '_ND', location=params.plot_dir)
        utils.plot_all_epoch(loss_summary, print_params + '_loss', location=params.plot_dir)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)

    params.relative_metrics = args.relative_metrics
    params.sampling =  args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.force_cpu = args.force_cpu
    params.iterations_b_evaluations = int(args.iterations_b_evaluations)
    params.plot_dir_gradients = os.path.join(params.plot_dir, 'gradients')
    params.plot_dir_filtering = os.path.join(params.plot_dir, 'filtering')

    # create missing directories
    try:
        os.mkdir(params.plot_dir)
        os.mkdir(params.plot_dir_gradients)
        os.mkdir(params.plot_dir_filtering)
    except FileExistsError:
        pass


    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logger.info(torch.__version__)
    logger.info("TIME RIGHT NOW!: " + str(datetime.now()))
    logger.info('Loading the datasets...')

    num_train = 500000

    if(args.dataset=='traffic'):
        train_set = Traffic(args.path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = TrafficTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)
    elif(args.dataset=='ele'):
        train_set = Ele(args.path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = EleTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)
    elif(args.dataset=='m4'):
        train_set = M4(args.path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = M4Test(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)
    elif(args.dataset=='wind'):
        train_set = Wind(args.path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = WindTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)
    elif(args.dataset=='solar'):
        train_set = Solar(args.path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = SolarTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)


    params.num_class = train_set.seq_num

    if (params.force_cpu):
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)
    else:
        # use GPU if available
        cuda_exist = torch.cuda.is_available()
        # Set random seeds for reproducible experiments if necessary
        if cuda_exist:
            #params.device = torch.device('cuda')
            # torch.cuda.manual_seed(240)
            dev = str("cuda:") + str(args.cuda_device)  # send it as parser
            params.device = torch.device(dev)
            logger.info('Using Cuda...')
            model = net.Net(params).cuda(dev)
        else:
            params.device = torch.device('cpu')
            # torch.manual_seed(230)
            logger.info('Not using cuda...')
            model = net.Net(params)

    train_loader = DataLoader(train_set, batch_size=params.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, num_workers=4)

    logger.info('Loading complete.')

    logger.info(f'Model: \n{str(model)}')
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(total_params)

    # fetch loss function
    loss_fn = net.loss_fn

    # Train the SAAM
    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    train_and_evaluate(model,
                       train_loader,
                       test_loader,
                       optimizer,
                       loss_fn,
                       params,
                       args.restore_file)
