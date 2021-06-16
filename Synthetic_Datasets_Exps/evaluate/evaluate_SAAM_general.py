"""
fmorenopino

This code is the same to "SAAM_LSTM_Embedding" but here we do not use any covariate.

Code based on DeepAR implementation.
"""
import argparse
import SAAM_general.net as net
from dataloaders.dataloader_sin_cos import Data_Seq_sin_cos
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils_SAAM_general as utils
import random
import os
import logging
import torch
import numpy as np
from tqdm import tqdm
import sys

logger = logging.getLogger('SAAM.Eval')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='synthetic', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='synthetic', help='Directory containing params.json')
parser.add_argument('--compute-rp', default=True, help='Whether to sample during evaluation')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def Dp(y_pred, y_true, q):
    return max([q * (y_pred - y_true), (q - 1) * (y_pred - y_true)])


def Rp_num_den(y_preds, y_trues, q):
    numerator = np.sum([Dp(y_pred, y_true, q) for y_pred, y_true in zip(y_preds, y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator, denominator

def evaluate(model, loss_fn, test_loader, params, plot_num, sample=True, force_plot_spectral=False):
    '''Evaluate the SAAM on the test set.
    Args:
        model: (torch.nn.Module) the SAAM
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate_SAAM.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    with torch.no_grad():

      plot_batch = np.random.randint(len(test_loader)-1)

      Rp_05 = 0
      Rp_09 = 0


      summary_metric = {}
      raw_metrics = utils.init_metrics(sample=sample)

      for i, (test_batch, labels) in enumerate(tqdm(test_loader)):
          test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
          labels = labels.to(torch.float32).to(params.device)
          batch_size = test_batch.shape[1]
          input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          hidden = model.init_hidden(batch_size)
          cell = model.init_cell(batch_size)

          test_batch_PSD, labels_batch_PSD = next(iter(test_loader))
          test_batch_PSD, labels_batch_PSD = test_batch_PSD.to(params.device), labels_batch_PSD.to(params.device)
          test_batch_PSD = test_batch_PSD.permute(1, 0, 2).to(torch.float32).to(params.device)

          emb = model.getEmbedding(test_batch_PSD).to(params.device)
          Rx = model.get_Rx(emb.permute(1, 0, 2)).to(params.device)
          PSD = model.get_FFT_(Rx).to(params.device)

          H = torch.Tensor().to(params.device)

          plotting = True if (i == plot_batch) else False

          alpha_list = torch.empty(size=(params.test_predict_start, test_batch.shape[1], params.lstm_layers * params.lstm_hidden_dim, params.filtering_window))
          FFT_l_list = torch.empty(size=(params.test_predict_start, params.filtering_window, test_batch.shape[1], params.lstm_layers * params.lstm_hidden_dim, 2))
          attentive_FFT_l_list = torch.empty_like(alpha_list)


          for t in range(params.test_predict_start):
              # if z_t is missing, replace it by output mu from the last time step
              zero_index = (test_batch[t,:,0] == 0)
              if t > 0 and torch.sum(zero_index) > 0:
                  test_batch[t,zero_index,0] = mu[zero_index]

              mu, sigma, hidden, cell, H, H_filtered, alpha, FFT_l, attentive_FFT_l = model(test_batch[t].unsqueeze(0), hidden, cell, PSD, H)
              #mu, sigma, hidden, cell, H, H_filtered, alpha, FFT_l, attentive_FFT_l = model(test_batch[t].unsqueeze(0), hidden, cell, PSD, H)
              #input_mu[:,t] = v_batch[:, 0] * mu + v_batch[:, 1]
              #input_sigma[:,t] = v_batch[:, 0] * sigma
              input_mu[:,t] = mu
              input_sigma[:,t] = sigma

              if (plotting):
                  alpha_list[t] = alpha
                  FFT_l_list[t] = FFT_l
                  attentive_FFT_l_list[t] = attentive_FFT_l

          if sample:
              #samples, sample_mu, sample_sigma = model.test(test_batch, hidden, cell, PSD, sampling=True)
              samples, sample_mu, sample_sigma = model.test(test_batch, hidden, cell, PSD, sampling=True)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics)
          else:
              #sample_mu, sample_sigma = model.test(test_batch,  hidden, cell, PSD)
              sample_mu, sample_sigma = model.test(test_batch, hidden, cell, PSD)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, relative = params.relative_metrics)


          ###Voy a calcular las metricas, para todos los elementos de evaluation
          if (params.compute_rp):
              n_samples = sample_mu.shape[0]
              predictions = sample_mu.tolist()
              observations = labels[:, params.test_predict_start:].tolist()
              num_05 = 0
              den_05 = 0
              num_09 = 0
              den_09 = 0
              for y_preds, y_trues in zip(predictions, observations):
                  num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
                  num_05 += num_i
                  den_05 += den_i

                  num_i, den_i = Rp_num_den(y_preds, y_trues, .9)
                  num_09 += num_i
                  den_09 += den_i
              rp_05 = (2 * num_05) / den_05
              rp_09 = (2 * num_09) / den_09

              Rp_05 += (rp_05 * n_samples)  # ponderado a todos los batches de evaluation
              Rp_09 += (rp_09 * n_samples)


          if i == plot_batch:
              n_FFTs_per_sequence = 10
              plot_t_idx = sorted(random.sample(range(params.test_predict_start), n_FFTs_per_sequence))
              if sample:
                  sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics)
              else:
                  sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start, relative = params.relative_metrics)
              # select 10 from samples with highest error and 10 from the rest
              size = 5 #Small number of plots for datasets with less sequences, DEFAULT=10
              top_10_nd_sample = (-sample_metrics['ND']).argsort()[:batch_size // size]  # hard coded to be 10
              chosen = set(top_10_nd_sample.tolist())
              all_samples = set(range(batch_size))
              not_chosen = np.asarray(list(all_samples - chosen))
              if batch_size < 100: # make sure there are enough unique samples to choose top 10 from
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=size, replace=True)
              else:
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=size, replace=False)
              if batch_size < 12: # make sure there are enough unique samples to choose bottom 90 from
                  random_sample_90 = np.random.choice(not_chosen, size=size, replace=True)
              else:
                  random_sample_90 = np.random.choice(not_chosen, size=size, replace=False)
              combined_sample = np.concatenate((random_sample_10, random_sample_90))

              z = test_batch[:,combined_sample,0].permute(1,0).data.cpu().numpy()
              label_plot = labels[combined_sample].data.cpu().numpy()
              predict_mu = sample_mu[combined_sample].data.cpu().numpy()
              predict_sigma = sample_sigma[combined_sample].data.cpu().numpy()
              plot_mu = np.concatenate((input_mu[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
              plot_sigma = np.concatenate((input_sigma[combined_sample].data.cpu().numpy(), predict_sigma), axis=1)
              plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics.items()}

              combined_sample = np.expand_dims(combined_sample, -1)
              alpha_list = alpha_list.data.cpu().numpy()[plot_t_idx, combined_sample,:,:]
              FFT_l_list = FFT_l_list.data.cpu().numpy()[plot_t_idx, :, combined_sample,:]
              attentive_FFT_l_list = attentive_FFT_l_list.data.cpu().numpy()[plot_t_idx, combined_sample,:,:]
              Rx_orig = model.get_Rx(test_batch.permute(1, 0, 2))
              FFT_orig = model.get_FFT_(Rx_orig).to(params.device)
              emb = model.getEmbedding(test_batch).to(params.device)
              Rx = model.get_Rx(emb.permute(1, 0, 2)).to(params.device)
              PSD = model.get_FFT_(Rx).to(params.device)
              plot_eight_windows(params.plot_dir, plot_mu, plot_sigma, label_plot, z, params.test_window, params.test_predict_start,
                                 plot_num, plot_metrics, size, plot_t_idx, alpha_list, FFT_l_list, attentive_FFT_l_list, FFT_orig, PSD, params, sample, force_plot_spectral)



      summary_metric = utils.final_metrics(raw_metrics, sampling=sample)

      if (params.compute_rp):
           Rp_05 = Rp_05/test_loader.sampler.num_samples
           Rp_09 = Rp_09/test_loader.sampler.num_samples

           summary_metric['Rp_05'] = Rp_05
           summary_metric['Rp_09'] = Rp_09

    return summary_metric



def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       z,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       size,
                       plot_t_idx, alpha_list,
                       FFT_l_list, attentive_FFT_l_list,
                       FFT_orig, FFT, params,
                       sampling=False, force_plot_spectral=False):

    x = np.arange(window_size)
    f = plt.figure(figsize=(8, (size*4+2)), constrained_layout=True)
    nrows = size*2
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        if k == (size-1):
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates worst and best predictions ', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x, predict_values[m], color='r')
        ax[k].fill_between(x[predict_start:], predict_values[m, predict_start:] - 2 * predict_sigma[m, predict_start:],
                         predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='red',
                         alpha=0.2)
        #for t in plot_t_idx:
        #    ax[k].axvline(t, linewidth=2, color='b', linestyle='dotted')
        ax[k].plot(x, labels[m, :], color='b')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')

        #metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})


        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
            f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f}'

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    model_dir = os.path.join('../experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.compute_rp = args.compute_rp
    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass
    
    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)

    # Create the input data pipeline
    logger.info('Loading the datasets...')

    if (args.dataset == 'synthetic'):
        data = Data_Seq_sin_cos()
        batch_size_train = params.batch_size
        batch_size_test = params.predict_batch
        train_loader, test_loader,test_set, complete_dataset_test = data.getData(batch_size_train=batch_size_train,batch_size_test=batch_size_test, sigma_noise=0.5)
        params.lstm_input_size = train_loader.sampler.data_source.tensors[0].shape[-1]
    else:
        logger.info('This script works just with the synthetic dataset. For real-world datasets use SAAM_LSTM_Embedding')
        sys.exit()
    logger.info('Loading complete.')

    logger.info('- done.')

    print('SAAM: ', model)
    loss_fn = net.loss_fn

    logger.info('Starting evaluation')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    t0 = [175, 150, 120, 110]
    tau = [25, 50, 80, 90]
    N = len(t0)
    times_to_evaluate = 10 #number of evaluations to obtain mean and variance of the errors

    for n in range(N):

        print ("********************************")
        print ("Steps: "+str(tau[n]))
        print ("********************************")

        params.predict_start = t0[n]
        params.test_predict_start = t0[n]
        params.predict_steps = tau[n]

        ND=[]
        RMSE=[]
        Rp_05 = []
        Rp_09 = []

        for i in range(times_to_evaluate):
            test_metrics = evaluate(model, loss_fn, test_loader, params, tau[n], params.sampling)
            ND.append(test_metrics['ND'])
            RMSE.append(test_metrics['RMSE'])
            Rp_05.append(test_metrics['Rp_05'])
            Rp_09.append(test_metrics['Rp_09'])

        print ("\nMean / Std")
        print ("-----ND:")
        print (str("{:.5f}".format(np.mean(ND))) + " +/- "+ str("{:.5f}".format(np.std(ND))))


        print ("-----RMSE:")
        print (str("{:.5f}".format(np.mean(RMSE))) + " +/- "+ str("{:.5f}".format(np.std(RMSE))))

        print ("-----Rp_05:")
        print (str("{:.5f}".format(np.mean(Rp_05))) + " +/- "+ str("{:.5f}".format(np.std(Rp_05))))

        print ("-----Rp_09:")
        print(str("{:.5f}".format(np.mean(Rp_09))) + " +/- " + str("{:.5f}".format(np.std(Rp_09))))

        save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
        utils.save_dict_to_json(test_metrics, save_path)