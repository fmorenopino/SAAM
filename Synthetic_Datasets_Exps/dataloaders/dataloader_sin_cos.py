'''
@author: fmorenopino
'''

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils import data

class Data_Seq_sin_cos(object):

    def getData(self,batch_size_train = 1, batch_size_test = 1, sigma_noise=0.5, n_sequences = 100):


        n_sequences = n_sequences

        for j in range(n_sequences):

            n_slots = 2
            n, p = 1, .5  # number of trials, probability of each trial
            s = np.random.binomial(n, p, n_slots)
            #print(s)

            """
            #In this example we fix the frequencies for visualization purposes
            f_1 = np.random.randint(1, 5)
            f_2 = np.random.randint(20, 25)
            f_3 = np.random.randint(10, 15)
            f_4 = np.random.randint(20, 25)
            """

            mu_phase, sigma_phase = 0, 0 #Not used in the experiments reported in the paper
            mu_amp, sigma_amp = 0, sigma_noise
            mu_fq, sigma_fq = 0, 0 #Not used in the experiments reported in the paper

            for i in range(0, len(s)):

                if (s[i] == 0):

                    fs = 100  # sample rate
                    f_1 = 1  # the frequency of the signal
                    f_2 = 20 # the frequency of the signal
                    x = np.arange(fs)

                    noise_phase = np.random.normal(mu_phase, sigma_phase, 1)
                    noise_amplitude = np.random.normal(mu_amp, sigma_amp, x.shape)
                    noise_fq = np.random.normal(mu_fq, sigma_fq, 1)

                    y_1_tmp = 2 * np.sin(2 * np.pi * (f_1 + noise_fq) * (x / fs) + noise_phase)
                    y_1 = noise_amplitude + y_1_tmp
                    noise_fq = np.random.normal(mu_fq, sigma_fq, 1)
                    y_2 = (noise_amplitude + np.sin(2 * np.pi * (f_2 + noise_fq) * (x / fs) + noise_phase))
                    noise_fq = np.random.normal(mu_fq, sigma_fq, 1)
                    # y_shared = (noise_amplitude + 2 * np.sin(2 * np.pi * (f_shared + noise_fq) * (x / fs) + noise_phase))

                    signal = y_1 + y_2


                elif (s[i] == 1):
                    fs = 100  # sample rate
                    f_1 = 5  # the frequency of the signal
                    f_2 = 20  # the frequency of the signal

                    x = np.arange(fs)

                    noise_phase = np.random.normal(mu_phase, sigma_phase, 1)
                    noise_amplitude = np.random.normal(mu_amp, sigma_amp, x.shape)
                    noise_fq = np.random.normal(mu_fq, sigma_fq, 1)

                    y_1_tmp = 2 * np.sin(2 * np.pi * (f_1 + noise_fq) * (x / fs) + noise_phase)
                    y_1 = noise_amplitude + y_1_tmp
                    noise_fq = np.random.normal(mu_fq, sigma_fq, 1)
                    y_2 = (noise_amplitude + np.sin(2 * np.pi * (f_2 + noise_fq) * (x / fs) + noise_phase))
                    noise_fq = np.random.normal(mu_fq, sigma_fq, 1)
                    #y_shared = (noise_amplitude + 2 * np.sin(2 * np.pi * (f_shared + noise_fq) * (x / fs) + noise_phase))

                    signal = y_1 + y_2

                signal = np.expand_dims(signal, axis=0)
                signal = np.expand_dims(signal, axis=-1)

                if (i == 0):
                    X = signal
                else:
                    X = np.concatenate((X, signal), axis=1)


            if (j == 0):
                X_f = X
            else:
                X_f = np.concatenate((X_f, X), axis=0)


        x = np.copy(X_f)
        x[:, 0, 0] = 0
        y = np.squeeze(np.roll(x[:, :, 0], -1))

        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.25)

        #We create the tensors to use the dataset with PyTorch
        X_train = torch.from_numpy(x_train)
        Y_train = torch.from_numpy(y_train).float()

        X_test = torch.from_numpy(x_test)
        Y_test = torch.from_numpy(y_test).float()

        train_loader = data.TensorDataset(X_train, Y_train)  # create your datset
        train_loader = data.DataLoader(train_loader, batch_size=batch_size_train, shuffle=True)  # create your dataloader

        test_loader = data.TensorDataset(X_test, Y_test)  # create your datset
        test_loader = data.DataLoader(test_loader, batch_size=batch_size_test, shuffle=True)  # create your dataloader

        complete_dataset_train = data.TensorDataset(X_train, Y_train)  # create your datset
        complete_dataset_train = data.DataLoader(complete_dataset_train, batch_size=n_sequences)  # create your dataloader

        complete_dataset_test = data.TensorDataset(X_test, Y_test)  # create your datset
        complete_dataset_test = data.DataLoader(complete_dataset_test, batch_size=n_sequences)  # create your dataloader

        return train_loader, test_loader, complete_dataset_train, complete_dataset_test
