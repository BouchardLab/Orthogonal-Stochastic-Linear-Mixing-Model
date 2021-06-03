import numpy as np
import os
import sys
from hdf5storage import loadmat
from hdf5storage import savemat

sys.path.append('data')


from data.DataProcess import process


def computing_predictive_measures(res_path="results", rank=2, model='oslmm', domains=["equity"], verbose=False):
    for domain in domains:
        print("domain {}".format(domain))
        data = process(domain)

        N_maes = list()
        maes = list()

        for trail in range(5):
            if model == 'oslmm':
                signature = domain + '_Q' + str(rank) + '_t' + str(trail) + '_oslmm'
            else:
                signature = domain + '_Q' + str(rank) + '_t' + str(trail)
            res_save_path = os.path.join(res_path, signature)
            res = loadmat(res_save_path, format=7.3)
            result = res["result"]
            hist_Y_pred = result['hist_Y_pred']
            # hist_W_hyps = result["hist_W_hyps"]
            # hist_f_hyps = result["hist_f_hyps"]
            # hist_sigma2_y = result["hist_sigma2_y"]

            Y_pred = np.mean(hist_Y_pred, axis=0)
            Y_std = data["Y_std"]
            Y_test = data["Y_test"]
            N_mae = np.mean(np.abs(Y_pred - Y_test))
            N_maes.append(N_mae)
            mae = np.mean(np.abs(Y_pred - Y_test) * Y_std)
            maes.append(mae)
            relative_mae = mae / (np.std(Y_test * Y_std))
            # import pdb; pdb.set_trace()
        N_maes = np.array(N_maes)
        maes = np.array(maes)
        if verbose:
            print("Original MAE = {}({})".format(np.mean(N_maes), np.std(N_maes)))
            print("Standardized MAE = {}({})".format(np.mean(maes), np.std(maes)))
        # import pdb; pdb.set_trace()
        return N_maes, maes


if __name__ == "__main__":
    # res_path = 'results'
    # rank = 2

    # model = "slmm"
    # model = 'oslmm'

    # domains = ["jura", "concrete", "equity", "pm25", "neuron"]
    # domains = ["jura"]
    # domains = ["concrete"]
    # domains = ["equity"]
    # domains = ["pm25"]
    computing_predictive_measures(res_path="results", rank=2, model='oslmm', domains=["equity"], verbose=True)

