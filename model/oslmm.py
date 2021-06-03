import numpy as np
from model.utils import sampling, kernel
import os
import sys
from hdf5storage import loadmat
from hdf5storage import savemat
from tqdm import tqdm
import scipy
import copy
import time

sys.path.append('data')
from data.DataProcess import process


class OSLMM:
    def __init__(self, config, label=None, init_type=2, fix_hyps=[]):

        print('Initialize OSLMM ...')
        np.random.seed(seed=None)

        self.config = config
        self.fix_hyps = fix_hyps

        if label is None:
            self.signature = self.config['data']['dname']
        else:
            self.signature = label

        self.record_time = self.config['record_time']
        self.Q = self.config['Q']
        self.in_d = self.config['data']['X_train'].shape[1]
        self.out_d = self.config['data']['Y_train'].shape[1]
        self.jitter = config['jitter']

        self.X_train = self.config['data']['X_train'].reshape((-1, self.in_d))
        self.X_test = self.config['data']['X_test'].reshape((-1, self.in_d))
        self.Y_train = self.config['data']['Y_train'].reshape((-1, self.out_d))
        self.Y_test = self.config['data']['Y_test'].reshape((-1, self.out_d))
        self.Y_test_ground = self.config['data']['Y_test_ground'].reshape((-1, self.out_d))
        self.Y_mean = self.config['data']['Y_mean']
        self.Y_std = self.config['data']['Y_std']

        self.nsave = config['nsave']
        self.nburnin = config['nburnin']
        self.nchain = self.nsave + self.nburnin
        self.interval = config['interval']
        self.n_accept_ls_h = 0
        self.n_accept_ls_f = 0
        self.scale_HM_h = 0.1
        self.scale_HM_f = 0.1
        self.opt_acc_rate = 0.44

        self.M = self.X_test.shape[0]
        self.N = self.X_train.shape[0]
        self.D = self.out_d
        self.K = self.Q
        self.P = self.out_d
        self.N_W = self.P*self.Q*self.N
        self.N_f = self.Q*self.N

        DList = self.config['data']['DList']
        self.D1 = int(DList[0])
        self.D2 = int(DList[1])
        self.D3 = int(DList[2])

        if self.config['kernel'] == 'SEiso':
            self.kernel = kernel.covSEiso
            self.kernel_npars = 2
        elif self.config['kernel'] == 'SEard':
            self.kernel = kernel.covSEard
            self.kernel_npars = self.in_d + 1
        elif self.config['kernel'] == 'SEisoU':
            self.kernel = kernel.covSEisoU
            self.kernel_npars = 1
        elif self.config['kernel'] == 'RQiso':
            self.kernel = kernel.covRQiso
            self.kernel_npars = 3
        elif self.config['kernel'] == 'RQard':
            self.kernel = kernel.covRQard
            self.kernel_npars = self.in_d + 2


        # noise variance
        self.sigma2_y = 1 # UDU' diagonal element of D
        # model parameters
        if init_type == 1:
            self.f = np.random.randn(self.N, self.Q)
            self.h = np.random.randn(self.N, self.Q)
            self.V = np.random.randn(self.P, self.Q)
            self.f_hyps = np.zeros(self.kernel_npars)
            self.h_hyps = np.zeros(self.kernel_npars)
        elif init_type == 2:
            self.f = 0.001 * np.random.randn(self.N, self.Q)
            self.h = 0.001 * np.random.randn(self.N, self.Q)
            self.V = 0.001 * np.random.randn(self.P, self.Q)
            self.f_hyps = np.zeros(self.kernel_npars)
            self.h_hyps = np.zeros(self.kernel_npars)
        else:
            self.f = init_type['f']
            self.h = init_type['h']
            self.V = init_type['V']
            self.f_hyps = init_type['f_hyps']
            self.h_hyps = init_type['h_hyps']

        # sample history
        self.hist_f_hyps = list()
        self.hist_h_hyps = list()
        self.hist_sigma2_y = list()
        self.hist_f = list()
        self.hist_h = list()
        self.hist_U = list()
        self.hist_Y_pred = list()
        self.hist_Y_pred_noise = list()

        # prior information
        self.a = 0.01
        self.b = 0.01
        self.c = 0.01
        self.d = 0.01

        # self.Y_pred = self.predict(self.X_test)

    def predict(self, X):
        U = self.V2U(self.V)
        Kh11 = self.kernel(self.h_hyps, self.X_train) + self.jitter*np.eye(self.N)  # N by N
        Kh12 = self.kernel(self.h_hyps, self.X_train, X)  # N by M
        Kh11InvKh12 = np.linalg.solve(Kh11, Kh12)  # N by M
        Kf11 = self.kernel(self.f_hyps, self.X_train) + self.jitter*np.eye(self.N) # N by N
        Kf12 = self.kernel(self.f_hyps, self.X_train, X)  # N by M
        Kf11InvKf12 = np.linalg.solve(Kf11, Kf12)  # N by M

        h_mu = np.tensordot(Kh11InvKh12.T, self.h, axes=1)  # M by Q
        f_mu = np.tensordot(Kf11InvKf12.T, self.f, axes=1)  # M by Q
        g_mu = np.multiply(np.exp(h_mu), f_mu)

        g_mu_expand = np.expand_dims(g_mu, axis=-1)
        U_expand = np.expand_dims(U, axis=0)
        Y_pred = np.matmul(U_expand, g_mu_expand)[..., -1]
        # import pdb; pdb.set_trace()

        Y_pred_noise = Y_pred + np.random.randn(self.M, self.P)*np.sqrt(self.sigma2_y)

        return Y_pred, Y_pred_noise, h_mu, f_mu, self.sigma2_y

    def log_pos_V(self, V_vec):
        V = V_vec.reshape([self.P, self.Q])
        U = self.V2U(V)
        U_expand = np.expand_dims(U, axis=0)
        S_half = np.exp(self.h)
        temp = np.expand_dims(np.multiply(S_half, self.f), axis=-1)
        y_mu = np.matmul(U_expand, temp)[..., -1]
        log_lik = -0.5 * np.sum((self.Y_train - y_mu) ** 2 / self.sigma2_y)
        log_prior = scipy.stats.norm.logpdf(V_vec).sum()
        log_pos = log_lik + log_prior
        return log_pos

    def log_lik_V(self, V_vec):
        V = V_vec.reshape([self.P, self.Q])
        U = self.V2U(V)
        U_expand = np.expand_dims(U, axis=0)
        S_half = np.exp(self.h)
        temp = np.expand_dims(np.multiply(S_half, self.f), axis=-1)
        y_mu = np.matmul(U_expand, temp)[..., -1]
        log_lik = -0.5 * np.sum((self.Y_train - y_mu)**2/self.sigma2_y)
        return log_lik

    def log_lik_h(self, h_vec, debug=False):
        h = h_vec.reshape([self.N, self.Q])
        U = self.V2U(self.V)
        U_expand = np.expand_dims(U, axis=0)
        S_half = np.exp(h)
        temp = np.expand_dims(np.multiply(S_half, self.f), axis=-1)
        y_mu = np.matmul(U_expand, temp)[..., -1]
        log_lik = -0.5 * np.sum((self.Y_train - y_mu)**2/self.sigma2_y)
        if debug:
            import pdb; pdb.set_trace()
        return log_lik

    def compute_mae(self):
        U = self.V2U(self.V)
        S_half = np.exp(self.h)
        U_expand = np.expand_dims(U, axis=0)
        temp = np.expand_dims(np.multiply(S_half, self.f), axis=-1)
        y_mu = np.matmul(U_expand, temp)[..., -1]
        print("MAE={}".format(np.mean(np.abs(self.Y_train - y_mu))))

    def log_pos_hyp_ls_h(self, h_hyps_ls):
        h_hyps = np.concatenate([h_hyps_ls, [self.h_hyps[-1]]])
        K_h = self.kernel(h_hyps, self.X_train) + self.jitter * np.eye(self.N)
        K_h_chol = np.linalg.cholesky(K_h)
        h_per = np.expand_dims(np.transpose(self.h, (1, 0)), axis=-1)
        h_scaled = np.linalg.solve(K_h_chol, h_per)[..., -1]
        sign_K_h, logdet_K_h = np.linalg.slogdet(K_h)
        log_lik_h = -0.5 * np.sum(h_scaled ** 2) - 0.5 * logdet_K_h * self.Q
        return log_lik_h

    def compute_hs(self):
        h_hyps = np.concatenate([self.h_hyps[:-1], [0]])
        K_h = self.kernel(h_hyps, self.X_train) + self.jitter * np.eye(self.N)
        K_h_chol = np.linalg.cholesky(K_h)
        h_per = np.expand_dims(np.transpose(self.h, (1, 0)), axis=-1)
        h_scaled = np.linalg.solve(K_h_chol, h_per)[..., -1]
        res = 0.5 * np.sum(h_scaled ** 2)
        # import pdb; pdb.set_trace()
        return res

    def log_pos_hyp_ls_f(self, f_hyps_ls):
        f_hyps = np.concatenate([f_hyps_ls, [self.f_hyps[-1]]])
        K_f = self.kernel(f_hyps, self.X_train) + self.jitter * np.eye(self.N)
        K_f_chol = np.linalg.cholesky(K_f)
        f_per = np.expand_dims(np.transpose(self.f, (1, 0)), axis=-1)
        f_scaled = np.linalg.solve(K_f_chol, f_per)[..., -1]
        sign_K_f, logdet_K_f = np.linalg.slogdet(K_f)
        log_lik_f = -0.5 * np.sum(f_scaled ** 2) - 0.5 * logdet_K_f * self.Q
        return log_lik_f

    def generate_Sigma(self):
        K_W = self.kernel(self.W_hyps, self.X_train) + self.jitter*np.eye(self.N)
        K_f = self.kernel(self.f_hyps, self.X_train) + self.jitter*np.eye(self.N)

        bmatrices = [K_W for _ in range(self.P*self.Q)] + [K_f for _ in range(self.Q)]
        Sigma = scipy.linalg.block_diag(*bmatrices)
        # reorder Sigma
        indexes_W = np.transpose(np.arange(self.N_W).reshape([self.P, self.Q, self.N]), axes=(2,0,1)).reshape(-1)
        indexes_f = np.transpose(np.arange(self.N_f).reshape([self.Q, self.N]), axes=(1,0)).reshape(-1)
        indexes = np.concatenate([indexes_W, indexes_f+self.N_W])
        Sigma_per = Sigma[indexes,:][:,indexes]
        return Sigma_per

    def generate_u(self):
        K_W = self.kernel(self.W_hyps, self.X_train) + self.jitter * np.eye(self.N)
        K_f = self.kernel(self.f_hyps, self.X_train) + self.jitter * np.eye(self.N)

        K_W_chol = np.linalg.cholesky(K_W)
        K_f_chol = np.linalg.cholesky(K_f)
        u = [np.dot(K_W_chol, np.random.randn(self.N)) for _ in range(self.P*self.Q)] + [np.dot(K_f_chol, np.random.randn(self.N)) for _ in range(self.Q)]
        u = np.concatenate(u)
        # reorder u
        indexes_W = np.transpose(np.arange(self.N_W).reshape([self.P, self.Q, self.N]), axes=(2, 0, 1)).reshape(-1)
        indexes_f = np.transpose(np.arange(self.N_f).reshape([self.Q, self.N]), axes=(1, 0)).reshape(-1)
        indexes = np.concatenate([indexes_W, indexes_f + self.N_W])
        u_per = u[indexes]
        return u_per

    def _sample_V(self, currV_vec, verbose=False):
        if verbose:
            ts = time.time()
        sample_u = np.random.randn(self.P*self.Q)
        V_vec = sampling.ESS_sampling_s(log_L=self.log_lik_V, nu=sample_u, f=currV_vec)
        if verbose:
            print("sampling V takes {}s".format(time.time() - ts))
        return V_vec

    def _sample_f(self):
        U = self.V2U(self.V)
        S_half = np.exp(self.h)
        S_half_inv = 1. / S_half
        T = np.transpose(np.expand_dims(U, axis=0) * np.expand_dims(S_half_inv, axis=1), axes=(0, 2, 1))  # N by Q by P
        PY = np.matmul(T, np.expand_dims(self.Y_train, axis=-1))[..., -1]
        sigma2_T = S_half_inv ** 2 * self.sigma2_y
        K_f = self.kernel(self.f_hyps, self.X_train) + self.jitter * np.eye(self.N)
        K_f_inv = np.linalg.inv(K_f)
        for q in range(self.Q):
            temp = 1./sigma2_T[:, q]
            Sigma_q = np.linalg.inv(K_f_inv + np.diag(temp))
            mu_q = Sigma_q.dot(PY[:, q]*temp)
            self.f[:, q] = np.random.multivariate_normal(mean=mu_q, cov=Sigma_q)
            # import pdb; pdb.set_trace()

    def _sample_h(self, currh_vec, verbose=False):
        if verbose:
            ts = time.time()
        K_h = self.kernel(self.h_hyps, self.X_train) + self.jitter * np.eye(self.N)
        K_h_chol = np.linalg.cholesky(K_h)
        sample_h = np.concatenate([K_h_chol.dot(np.random.randn(self.N)) for _ in range(self.Q)])
        h_vec = sampling.ESS_sampling_s(log_L=self.log_lik_h, nu=sample_h, f=currh_vec)
        if verbose:
            print("sampling h takes {}s".format(time.time() - ts))
        return h_vec

    def V2U(self, V):
        T = V.T.dot(V)
        T = np.linalg.inv(scipy.linalg.sqrtm(T))
        return V.dot(T)

    def fit(self):
        if self.record_time:
            ts = time.time()
        for iter in tqdm(range(self.nchain)):
            # update V
            currV_vec = self.V.reshape(-1)
            V_vec = self._sample_V(currV_vec)
            self.V = V_vec.reshape([self.P, self.Q])
            # print("log lik V: {}".format(self.log_lik_V(currV_vec)))
            # update f
            self._sample_f()
            # update h
            currh_vec = self.h.reshape(-1)
            h_vec = self._sample_h(currh_vec)
            self.h = h_vec.reshape([self.N, self.Q])
            # print("log lik h: {}".format(self.log_lik_h(h_vec, debug=False)))
            # self.compute_mae()

            # update hyper-parameters
            self.U = self.V2U(self.V)
            self.S_half = np.exp(self.h)
            U_expand = np.expand_dims(self.U, axis=0)
            temp = np.expand_dims(np.multiply(self.S_half, self.f), axis=-1)
            self.g = np.matmul(U_expand, temp)[..., -1]
            if not ('sigma2_y' in self.fix_hyps):
                # update sigma2_y
                self.sigma2_y = 1/np.random.gamma(shape=self.a+0.5*self.N*self.P, scale=1./(self.b+0.5*np.sum((self.Y_train - self.g)**2)))
            # update h_hyps
            if not ('ls_h' in self.fix_hyps):
                ## update length-scale parameter
                currhyps_ls_h = self.h_hyps[:-1]
                hyps_res = sampling.MH(log_pos=self.log_pos_hyp_ls_h, x=currhyps_ls_h, scale=self.scale_HM_h)
                self.h_hyps[:-1] = hyps_res[0]
                self.n_accept_ls_h += hyps_res[1]
            if not ('s2_h' in self.fix_hyps):
                ## update scale parameter
                temp = self.compute_hs()
                hyps_s2_h = 1./np.random.gamma(shape=self.c+0.5*self.N*self.Q, scale=1./(self.d+temp))
                self.h_hyps[-1] = 0.5*np.log(hyps_s2_h)
                # print(self.h_hyps)
            # update f_hyps
            if not ('ls_f' in self.fix_hyps):
                ## update length-scale parameter
                currhyps_ls_f = self.f_hyps[:-1]
                hyps_res = sampling.MH(log_pos=self.log_pos_hyp_ls_f, x=currhyps_ls_f, scale=self.scale_HM_f)
                self.f_hyps[:-1] = hyps_res[0]
                self.n_accept_ls_f += hyps_res[1]

            # adapt the proposal scale
            if iter >= (self.nburnin/2):
                delta_i = np.min([0.01, (iter+1)**(-0.5)])
                if (self.n_accept_ls_h/(iter+1) > 0.44):
                    self.scale_HM_h *= np.exp(delta_i)
                if (self.n_accept_ls_h/(iter+1) < 0.44):
                    self.scale_HM_h *= np.exp(-delta_i)
                if (self.n_accept_ls_f/(iter+1) > 0.44):
                    self.scale_HM_f *= np.exp(delta_i)
                if (self.n_accept_ls_f/(iter+1) < 0.44):
                    self.scale_HM_f *= np.exp(-delta_i)
            # self.Y_pred, _, _, _ = self.predict(self.X_test)
            # print("mean absolute error:{}".format(np.mean(np.abs(self.Y_pred - self.Y_test))))

            # posterior predictive sampling
            # Y_pred, Y_pred_noise, _, _, _ = self.predict(self.X_test)
            # print("mean absolute error:{}".format(np.mean(np.abs(Y_pred - self.Y_test))))

            # save sample
            if (iter >= self.nburnin) and (iter%self.interval==0):
                print("{}th iteration, MH accept rate for length-scale h:{}".format(iter, self.n_accept_ls_h/(iter+1)))
                print("{}th iteration, MH accept rate for length-scale f:{}".format(iter, self.n_accept_ls_f/(iter+1)))
                self.hist_h_hyps.append(copy.copy(self.h_hyps))
                self.hist_f_hyps.append(copy.copy(self.f_hyps))
                self.hist_sigma2_y.append(copy.copy(self.sigma2_y))
                self.hist_h.append(copy.copy(self.h))
                self.hist_f.append(copy.copy(self.f))
                self.hist_U.append(copy.copy(self.U))
                # posterior predictive sampling
                Y_pred, Y_pred_noise, _, _, _ = self.predict(self.X_test)
                print("mean absolute error:{}".format(np.mean(np.abs(Y_pred - self.Y_test))))
                self.hist_Y_pred.append(copy.copy(Y_pred))
                self.hist_Y_pred_noise.append(copy.copy(Y_pred_noise))
                # import pdb; pdb.set_trace()

        res = dict()
        if self.record_time:
            running_time = time.time() - ts
            res["running_time"] = running_time
        res["hist_h_hyps"] = self.hist_h_hyps
        res["hist_f_hyps"] = self.hist_f_hyps
        res["hist_sigma2_y"] = self.hist_sigma2_y
        res["hist_h"] = self.hist_h
        res["hist_f"] = self.hist_f
        res["hist_U"] = self.hist_U
        res["hist_Y_pred"] = self.hist_Y_pred
        res["hist_Y_pred_noise"] = self.hist_Y_pred_noise
        return res


def run(args):
    domain = args["domain"]
    kernel = args["kernel"]
    nsave = args["nsave"]
    nburnin = args["nburnin"]
    interval = args["interval"]
    Q = args["Q"]
    record_time = args["record_time"]

    print('Experiment summary: ')
    print(' - Domain name:', domain)
    print(' - Cov Func:', kernel)
    print(' - Q:', Q)

    res_path = 'results'

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    trail = args["trail"]
    data = process(domain)

    signature = domain + '_Q' + str(Q) + '_t' + str(trail) + '_oslmm'
    cfg = {
        'data': data,
        'signature': signature,
        'jitter': 1e-3,
        'Q': Q,
        'kernel': kernel,
        'nsave': nsave,
        'nburnin': nburnin,
        'interval': interval,
        'record_time': record_time
    }

    model = OSLMM(cfg, label=signature)
    res = model.fit()
    cfg['result'] = res
    res_save_path = os.path.join(res_path, signature)
    savemat(res_save_path, cfg, format='7.3')
    print('results saved to', res_save_path + '.mat')
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    pass