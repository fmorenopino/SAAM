import argparse
from tqdm import tqdm
import SAAM.net as net
import utils as utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataloader import *
logger = logging.getLogger('SAAM.Eval')
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data_-folder', default='data_', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--restore-file', default='best', help='Optional, name of the file in --model_dir containing weights to reload before training')  # 'best' or 'epoch_#'
parser.add_argument('--overlap',default=False, action='store_true',help='If we overlap prediction range during sampling')
parser.add_argument('--iterations-b-evaluations', default=5000, help='Whether to sample during evaluation')


def evaluate(model, loss_fn, test_loader, params, plot_num, sample=True):
    '''Evaluate the SAAM on the test set.
    Args:
        model: (torch.nn.Module) the SAAM
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data_ and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate_SAAM.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    with torch.no_grad():

      plot_batch = np.random.randint(len(test_loader)-1)

      summary_metric = {}
      raw_metrics = utils.init_metrics(sample=sample)

      # Test_loader: 
      # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
      # id_batch ([batch_size]): one integer denoting the time series id;
      # v ([batch_size, 2]): scaling factor for each window;
      # labels ([batch_size, train_window]): z_{1:T}.

      predictions = []

      for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
          test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
          id_batch = id_batch.unsqueeze(0).to(params.device)
          v_batch = v.to(torch.float32).to(params.device)
          labels = labels.to(torch.float32).to(params.device)
          batch_size = test_batch.shape[1]
          input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          hidden = model.init_hidden(batch_size)
          cell = model.init_cell(batch_size)

          emb = model.getEmbedding(test_batch, id_batch).to(params.device)
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

              mu, sigma, hidden, cell, H, _, alpha, FFT_l, attentive_FFT_l = model(test_batch[t].unsqueeze(0), id_batch, hidden, cell, PSD, H)
              input_mu[:,t] = v_batch * mu
              input_sigma[:,t] = v_batch * sigma

              if (plotting):
                  alpha_list[t] = alpha
                  FFT_l_list[t] = FFT_l
                  attentive_FFT_l_list[t] = attentive_FFT_l

          if sample:
              samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, PSD, sampling=True)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics)
          else:
              sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, PSD)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, relative = params.relative_metrics)



          if i == plot_batch:
              n_FFTs_per_sequence = 10
              plot_t_idx = sorted(random.sample(range(params.test_predict_start), n_FFTs_per_sequence))
              if sample:
                  sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics)
              else:
                  sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start, relative = params.relative_metrics)
              # select 10 from samples with highest error and 10 from the rest
              size = 10
              top_10_nd_sample = (-sample_metrics['ND']).argsort()[:batch_size // size]  # hard coded to be 10
              chosen = set(top_10_nd_sample.tolist())
              all_samples = set(range(batch_size))
              not_chosen = np.asarray(list(all_samples - chosen))
              if batch_size < 100: # make sure there are enough unique samples to choose top 10 from
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=True)
              else:
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
              if batch_size < 12: # make sure there are enough unique samples to choose bottom 90 from
                  random_sample_90 = np.random.choice(not_chosen, size=10, replace=True)
              else:
                  random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
              combined_sample = np.concatenate((random_sample_10, random_sample_90))

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
              plot_eight_windows(params.plot_dir, plot_mu, plot_sigma, label_plot, params.test_window, params.test_predict_start,
                                 plot_num, plot_metrics, size, plot_t_idx, alpha_list, FFT_l_list, attentive_FFT_l_list, FFT_orig,
                                 test_batch, id_batch, model, params,sample)





      summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
      metrics_string = '; '.join('{}: {:05.6f}'.format(k, v) for k, v in summary_metric.items())
      logger.info('- Full test metrics: ' + metrics_string)
    return summary_metric


def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       size,
                       plot_t_idx, alpha_list,
                       FFT_l_list, attentive_FFT_l_list,
                       FFT_orig, batch, id_batch, model, params,
                       sampling=False):

    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
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

    if ((plot_num % params.plots_spectral_every_x_epochs == 0 or plot_num == params.num_epochs - 1)):

        """
        Plotting Spectral Information
        """

        emb = model.getEmbedding(batch, id_batch).to(params.device)
        Rx = model.get_Rx(emb.permute(1, 0, 2)).to(params.device)
        FFT = model.get_FFT_(Rx).to(params.device)

        fVals = np.arange(0, stop=(params.NFFT / 2) + 1) * params.fs / params.NFFT
        saving_directory = os.path.join(plot_dir, str(plot_num) + '_fft_orig.png')
        plt.close("all")
        plt.plot(fVals, FFT_orig.data.cpu().numpy())
        plt.ylabel("FFT")
        plt.xlabel("Hz")
        plt.savefig(saving_directory)

        #FFT(embedding)
        saving_directory = os.path.join(plot_dir, str(plot_num) + '_fft_emb.png')
        plt.close("all")
        plt.plot(fVals, FFT.data.cpu().numpy())
        plt.ylabel("FFT")
        plt.xlabel("Hz")
        plt.savefig(saving_directory)

        #Alpha
        n_features = alpha_list.shape[2]
        emb_features_plotting = list(range(n_features)) if n_features < 4 else sorted(random.sample(range(n_features), 4))
        plt.close("all")
        directory = os.path.join(plot_dir, str(plot_num) + '_alpha')
        os.makedirs(directory, exist_ok=True)

        rows = alpha_list.shape[1]
        #cols = alpha_list.shape[-1]
        cols=len(emb_features_plotting)
        x_Hz = np.arange(alpha_list.shape[-1])
        cols_names = range(0, cols)
        rows_names = range(0, rows)

        for s in range(size * 2):
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
            saving_directory = directory + "/sequence_" + str(s) + ".png"
            for ax, col in zip(axes[0], cols_names):
                ax.set_title("Emb. Hid. Dim " + str(cols_names[col]))

            for ax, row in zip(axes[:, 0], rows_names):
                ax.set_ylabel("t = " + str(plot_t_idx[row]))

            for row in range(rows):
                for column in cols_names:
                    axes[row, column].plot(x_Hz, alpha_list[s, row, emb_features_plotting[column], :], color='b')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle("Alpha", verticalalignment="bottom")
            plt.savefig(saving_directory)
            plt.close()

        #FFT_t
        plt.close("all")
        directory = os.path.join(plot_dir, str(plot_num) +'_fft_l')
        os.makedirs(directory, exist_ok=True)

        for s in range(size * 2):
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
            saving_directory = directory + "/sequence_" + str(s) + ".png"
            for ax, col in zip(axes[0], cols_names):
                ax.set_title("Emb. Hid. Dim " + str(cols_names[col]))

            for ax, row in zip(axes[:, 0], rows_names):
                ax.set_ylabel("t = " + str(plot_t_idx[row]))

            for row in range(rows):
                for column in cols_names:
                    axes[row, column].plot(x_Hz, FFT_l_list[s, row, :, emb_features_plotting[column], 0], color='b')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle("FFT_l", verticalalignment="bottom")
            plt.savefig(saving_directory)
            plt.close()


        #Attentive FFT

        plt.close("all")
        directory = os.path.join(plot_dir, str(plot_num) +'_attentive_fft_l')
        os.makedirs(directory, exist_ok=True)

        for s in range(size * 2):
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
            saving_directory = directory + "/sequence_" + str(s) + ".png"
            for ax, col in zip(axes[0], cols_names):
                ax.set_title("Emb. Hid. Dim " + str(cols_names[col]))

            for ax, row in zip(axes[:, 0], rows_names):
                ax.set_ylabel("t = " + str(plot_t_idx[row]))

            for row in range(rows):
                for column in cols_names:
                    axes[row, column].plot(x_Hz, FFT_l_list[s, row, :, emb_features_plotting[column], 0], color='b', linestyle='dotted')
                    axes[row, column].plot(x_Hz, attentive_FFT_l_list[s, row, emb_features_plotting[column], :], color='b')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle("FFT_l (dotted), Attentive_FFT_l", verticalalignment="bottom")
            plt.savefig(saving_directory)
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
    
    params.iterations_b_evaluations = int(args.iterations_b_evaluations)


    # Create the input data_ pipeline
    logger.info('Loading the datasets...')

    num_train = 500000

    if(args.dataset=='traffic'):
        path = 'data/traffic.csv'
        train_set = Traffic(path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = TrafficTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)
    elif(args.dataset=='traffic_fine'):
        path = 'data/traffic_fine.csv'
        train_set = Traffic_fine(path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = Traffic_fineTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)
    elif(args.dataset=='ele'):
        path = 'data/electricity.txt'
        train_set = Ele(path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = EleTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)
    elif(args.dataset=='ele_fine'):
        path = 'data/electricity.txt'
        train_set = Ele_fine(path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = Ele_fineTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)
    elif(args.dataset=='m4'):
        path = 'data/M4.csv'
        train_set = M4(path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = M4Test(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)
    elif(args.dataset=='wind'):
        path = 'data/wind.csv'
        train_set = Wind(path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = WindTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)
    elif(args.dataset=='solar'):
        path = 'data/solar.csv'
        train_set = Solar(path, num_train,overlap = args.overlap ,pred_days=params.pred_days,win_len= params.predict_start+params.predict_steps)
        test_set = SolarTest(train_set.points,train_set.covariates,train_set.withhold_len,params.predict_start,params.predict_steps)

    params.num_class = train_set.seq_num
    logger.info('- done.')

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

    train_loader = DataLoader(train_set, batch_size=params.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, num_workers=4)

    print('Model: ', model)
    loss_fn = net.loss_fn

    logger.info('Starting evaluation')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    test_metrics = evaluate(model, loss_fn, test_loader, params, -1, params.sampling)
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
