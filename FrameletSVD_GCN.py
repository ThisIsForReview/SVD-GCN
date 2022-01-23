from model.utils import load_dataset, train_test_split, load_npz  #, score_link_prediction, score_node_classification
import numpy as np
from scipy import linalg, sparse
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

# function for pre-processing
def get_SVD_operator(S, DFilters, scale, J, Lev, device):
    r = len(DFilters)
    # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = np.sqrt(S)
    dd = dict()
    for l in range(Lev):
        for j in range(r):
            xx = (scale ** (-J - l)) * S
            xx[xx>np.pi] = np.pi
            dd[j, l] = (DFilters[j](xx)) * FD1
        FD1 = dd[0,l]
    d_list = list()
    for i in range(r):
        for l in range(Lev):
            d_list.append(torch.tensor(dd[i, l]).to(device))
    A = torch.zeros(r * Lev, len(S)).to(device)
    for j in range(len(d_list)):
        A[j, :] = d_list[j]
    A = A[Lev-1:,:]
    return A

def waveletShrinkage(x, thr, mode='soft'):
    """
    Perform soft or hard thresholding. The shrinkage is only applied to high frequency blocks.

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param thr: thresholds stored in a torch dense tensor with shape [num_hid_features]
    :param mode: 'soft' or 'hard'. Default: 'soft'
    :return: one block of wavelet coefficients after shrinkage. The shape will not be changed
    """
    #assert mode in ('soft', 'hard'), 'shrinkage type is invalid'
    if mode == None:
        return x
    if mode == 'soft':
        x = torch.mul(torch.sign(x), (((torch.abs(x) - thr) + torch.abs(torch.abs(x) - thr)) / 2))
    else:
        x = torch.mul(x, (torch.abs(x) > thr))

    return x

def multiScales(x, r, Lev, num_nodes):
    """
    calculate the scales of the high frequency wavelet coefficients, which will be used for wavelet shrinkage.

    :param x: all the blocks of wavelet coefficients, shape [r * Lev * num_nodes, num_hid_features] torch dense tensor
    :param r: an integer
    :param Lev: an integer
    :param num_nodes: an integer which denotes the number of nodes in the graph
    :return: scales stored in a torch dense tensor with shape [(r - 1) * Lev] for wavelet shrinkage
    """
    for block_idx in range((r-1) * Lev + 1):
        if block_idx == 0:
            specEnergy_temp = torch.unsqueeze(torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2), dim=0)
            specEnergy = torch.unsqueeze(torch.tensor(1.0), dim=0).to(x.device)
        else:
            specEnergy = torch.cat((specEnergy,
                                    torch.unsqueeze(torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2), dim=0) / specEnergy_temp))

    assert specEnergy.shape[0] == (r - 1) * Lev + 1, 'something wrong in multiScales'
    return specEnergy

def simpleLambda(x, scale, sigma=1.0):
    """
    De-noising by Soft-thresholding. Author: David L. Donoho

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param scale: the scale of the specific input block of wavelet coefficients, a zero-dimensional torch tensor
    :param sigma: a scalar constant, which denotes the standard deviation of the noise
    :return: thresholds stored in a torch dense tensor with shape [num_hid_features]
    """
    n, m = x.shape
    thr = (math.sqrt(2 * math.log(n)) / math.sqrt(n) * sigma) * torch.unsqueeze(scale, dim=0).repeat(m)

    return thr

class SVDUFGConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, bias=True, activation = None):
        super(SVDUFGConv, self).__init__()
        self.r = r
        self.Lev = Lev
        self.num_nodes = num_nodes
        self.out_features = out_features
        self.act = activation
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
            self.filter = nn.Parameter(torch.Tensor(((r-1) * Lev + 1), num_nodes).cuda())
        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(((r-1) * Lev + 1), num_nodes))
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features).cuda())
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list, U, Vt):
        # d_list is a list of vectors operators generated from Framelets
        # x is a torch dense tensor
        x = torch.matmul(x, self.weight)

        # SVD Decomposition Compositions Filtering
        recons = torch.zeros(self.num_nodes,self.out_features).to(x.device)
        for j in range(d_list.shape[0]):
            recons += torch.matmul(U, ((d_list[j, :] * self.filter[j] * d_list[j, :]).unsqueeze(-1)) * torch.matmul(Vt, x))
        x_shrink = recons

        if self.bias is not None:
            x_shrink += self.bias
        if self.act is not None:
            x_shrink = self.act(x_shrink)
        return x_shrink

class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, r, Lev, num_nodes, dropout_prob=0.5, activation = None):
        super(Net, self).__init__()
        self.GConv1 = SVDUFGConv(num_features, nhid, r, Lev, num_nodes, activation = activation)
        #self.GConv2 = SVDUFGConv(nhid, num_classes, r, Lev, num_nodes)    # This may be used when do denoising
        self.drop1 = nn.Dropout(dropout_prob)
        self.weight = nn.Parameter(torch.Tensor(nhid, num_classes))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, data, d_list, U, Vt):
        x = data  # x has shape [num_nodes, num_input_features]

        x = self.GConv1(x, d_list, U, Vt)
        x = self.drop1(x)
        #x = self.GConv2(x, d_list, U, Vt)
        x = torch.matmul(x, self.weight) + self.bias
        return F.log_softmax(x, dim=1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora_full',
                        help='name of dataset (default: cora_ml): coral_full, citeseer, citeseer_full, amazon_cs, amazon_photo')
    parser.add_argument('--reps', type=int, default=10,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=0.001,
                        help='weight decay (default: 5e-3)')
    parser.add_argument('--nhid', type=int, default=32,
                        help='number of hidden units (default: 16)')
    parser.add_argument('--Lev', type=int, default=2,
                        help='level of transform (default: 2)')
    parser.add_argument('--scale', type=float, default=1.1,
                        help='dilation scale > 1 (default: 2)')
    parser.add_argument('--FrameType', type=str, default='Linear',
                        help='frame type (default: Entropy): Linear, Sigmoid, Entropy')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='dropout probability (default: 0.3)')
    parser.add_argument('--activation', type=str, default='None',
                        help='activation function (default: None): None, elu, sigmoid, relu, tanh')
    parser.add_argument('--shrinkage', type=str, default='soft',
                        help='soft or hard thresholding (default: soft)')
    parser.add_argument('--sigma', type=float, default=0.00001,  #0.005,  #0.0001,
                        help='standard deviation of the noise (default: 1.0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha value in Framelet function (default: 0.5 for Entropy; 20.0 for Sigmoid)')   #newly added parameter
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1000)')
    parser.add_argument('--ExpNum', type=int, default='1',
                        help='The Experiment Number (default: 1)')
    parser.add_argument('--FrequencyNum', type=int, default=100,
                        help='The number of (noise) high frequency components (default: 100)')  # We are not using this.
    args = parser.parse_args()

    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    datapath = './Data/DiGraphData/'
    # Loading and Preparing data
    if args.dataset == 'cora_ml':   # This npz file from https://github.com/flyingtango/DiGCN
        datafileName = 'cora_ml'
        g = load_dataset(datapath + datafileName + '.npz')
        A, X, z = g['A'], g['X'], g['z']
    elif args.dataset == 'cora_full':     # This npz file from  https://github.com/EdisonLeeeee/GraphData
        datafileName = 'cora_full'
        g = load_npz(datapath + datafileName + '.npz')
        A, X, z = g['adj_matrix'], g['node_attr'], g['node_label']
        A = A.astype(np.float32)
    elif args.dataset == 'citeseer_full':
        datafileName = 'citeseer_full'
        g = load_dataset(datapath + datafileName + '.npz')
        A, X, z = g['A'], g['X'], g['z']
    elif args.dataset == 'citeseer':    # This npz file from https://github.com/flyingtango/DiGCN
        datafileName = 'citeseer'
        g = load_dataset(datapath + datafileName + '.npz')
        A, X, z = g['A'], g['X'], g['z']
    elif args.dataset == 'amazon_cs':    # This npz file from  https://github.com/EdisonLeeeee/GraphData
        datafileName = 'amazon_cs'
        g = load_npz(datapath + datafileName + '.npz')
        A, X, z = g['adj_matrix'], g['node_attr'], g['node_label']
        A = A.astype(np.float32)
    elif args.dataset == 'amazon_photo':    # This npz file from  https://github.com/EdisonLeeeee/GraphData
        datafileName = 'amazon_photo'
        g = load_npz(datapath + datafileName + '.npz')
        A, X, z = g['adj_matrix'], g['node_attr'], g['node_label']
        A = A.astype(np.float32)
    else:
        raise Exception('Invalid Dataset')
    del g   # We dont need g

    precomputed_usv = datapath + datafileName + '_usv.npz'
    if os.path.isfile(precomputed_usv):
        usvfiles = np.load(precomputed_usv)
        U = usvfiles['U']
        S = usvfiles['S']
        Vt = usvfiles['Vt']
    else:
        indices = A.nonzero()
        A = sparse.csr_matrix((A.data, (indices[0], indices[1])), shape=(A.shape[0], A.shape[0]))
        #A = (A + sparse.eye(A.shape[0])).todense()
        A_t = torch.tensor((A + sparse.eye(A.shape[0])).todense())
        row_deg_inv_sqrt = A_t.sum(dim=1).clamp(min=1).pow(-0.5)
        col_deg_inv_sqrt = A_t.sum(dim=0).clamp(min=1).pow(-0.5)
        A_t = row_deg_inv_sqrt.unsqueeze(-1) * A_t * col_deg_inv_sqrt.unsqueeze(-2)
        if A.shape[0] > 3000:
            U, S, Vt = linalg.svd(A_t.to(torch.float32))
        else:
            U, S, Vt = linalg.svd(A_t)
        np.savez(precomputed_usv, U = U, S = S, Vt = Vt)
    # Double Precision
    U = U.astype(np.float64)
    S = S.astype(np.float64)
    Vt = Vt.astype(np.float64)

    # Defining the Framelet. We have mirrored all the framelet function for SVD purpose  by x' = \pi - x
    FrameType = args.FrameType
    if FrameType == 'Haar':
        D1 = lambda x: np.cos((np.pi - x) / 2)
        D2 = lambda x: np.sin((np.pi - x) / 2)
        DFilters = [D1, D2]
    elif FrameType == 'Sigmoid':
        alpha = args.alpha  # make sure default value = 20.0
        D1 = lambda x: np.sqrt(1.0 - 1.0 / (1.0 + np.exp(-alpha * ((np.pi - x) / np.pi - 0.5))))
        D2 = lambda x: np.sqrt(1.0 / (1.0 + np.exp(-alpha * ((np.pi - x) / np.pi - 0.5))))
        DFilters = [D1, D2]
    elif FrameType == 'Entropy':
        alpha = args.alpha  # with a default value = 0.5  (can be made a tunable parameter)
        D1 = lambda x: np.sqrt(
            (1 - alpha * 4 * ((np.pi - x) / np.pi) + alpha * 4 * ((np.pi - x) / np.pi) * ((np.pi - x) / np.pi)) * (((np.pi - x) / np.pi) <= 0.5))
        D2 = lambda x: np.sqrt(alpha * 4 * ((np.pi - x) / np.pi) - alpha * 4 * ((np.pi - x) / np.pi) * ((np.pi - x) / np.pi))
        D3 = lambda x: np.sqrt(
            (1 - alpha * 4 * ((np.pi - x) / np.pi) + alpha * 4 * ((np.pi - x) / np.pi) * ((np.pi - x) / np.pi)) * (((np.pi - x) / np.pi) > 0.5))
        DFilters = [D1, D2, D3]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos((np.pi - x) / 2))
        D2 = lambda x: np.sin((np.pi - x)) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin((np.pi - x) / 2))
        DFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos((np.pi - x) / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin((np.pi - x) / 2)), np.cos((np.pi - x) / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin((np.pi - x) / 2) ** 2), np.cos((np.pi - x) / 2))
        D4 = lambda x: np.sin((np.pi - x) / 2) ** 3
        DFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')

    # Preparing SVD-Framelet Matrix
    Lev = args.Lev  # level of transform
    scale = args.scale  # dilation scale
    J = np.log(S[0] / np.pi) / np.log(scale)
    r = len(DFilters)
    d_list = get_SVD_operator(S, DFilters, scale, J, Lev, device)

    '''
    Training Scheme
    '''
    # Hyper-parameter Settings
    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.nhid
    if args.activation == 'None':
        activation = None
    else:
        activation = eval('F.'+ args.activation)   # make the string into a function
    dropout_prb = args.dropout

    # create result matrices
    num_epochs = args.epochs
    num_reps = args.reps
    epoch_loss = dict()
    epoch_acc = dict()
    epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))


    saved_model_val_acc = np.zeros(num_reps)
    saved_model_test_acc = np.zeros(num_reps)

    SaveResultFilename = args.dataset + 'Exp{0:03d}'.format(args.ExpNum)
    ResultCSV = args.dataset + '_AllExp.csv'

    mask = train_test_split(z, 1020, train_examples_per_class=20, val_examples_per_class=None,
                     test_examples_per_class=None, train_size=None, val_size=500, test_size=None)
    num_classes = z.max()+1
    # Here the seed is 1020.   Now we have to re-set seed as we wish
    # print(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data to torch
    U = torch.tensor(U).to(device)
    S = torch.tensor(S).to(device)
    Vt = torch.tensor(Vt).to(device)
    X = torch.tensor(X.todense()).to(torch.float64).to(device)
    z = torch.tensor(z).to(torch.long).to(device)
    train_mask = (torch.tensor(mask['train']) == 1).to(device)
    val_mask = (torch.tensor(mask['val']) == 1).to(device)
    test_mask = (torch.tensor(mask['test']) == 1).to(device)
    cases = []
    cases.append(train_mask)
    cases.append(val_mask)
    cases.append(test_mask)
    namedCases = ['train_mask', 'val_mask', 'test_mask']

    for rep in range(num_reps):
        print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0
        record_test_acc = 0.0

        # initialize the model: setting cutoff to True makes the first layer as hard high-frequency cut-off
        model = Net(X.shape[1], nhid, num_classes, r, Lev, X.shape[0], dropout_prob=dropout_prb, activation = activation).to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # initialize the learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # training
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(X, d_list, U, Vt)
            loss = F.nll_loss(out[train_mask], z[train_mask])
            loss.backward()
            optimizer.step()

            # evaluation mode
            model.eval()
            out = model(X, d_list, U, Vt)
            for i, mask in enumerate(cases):
                pred = out[mask].max(dim=1)[1]
                correct = float(pred.eq(z[mask]).sum().item())
                e_acc = correct / mask.sum().item()
                epoch_acc[namedCases[i]][rep, epoch] = e_acc

                e_loss = F.nll_loss(out[mask], z[mask])
                epoch_loss[namedCases[i]][rep, epoch] = e_loss

            # scheduler.step(epoch_loss['val_mask'][rep, epoch])

            # print out results
            if (epoch + 1) % 1 == 0:
                print('Epoch: {:3d}'.format(epoch + 1),
                   'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                   'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                   'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                   'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                   'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                   'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            # save model   We dont need this on HPC
            if epoch > 10:
               if epoch_acc['val_mask'][rep, epoch] > max_acc:
                   #torch.save(model.state_dict(), SaveResultFilename + '.pth')
                   # print('Epoch: {:3d}'.format(epoch + 1),
                   #      'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                   #      'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                   #      'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                   #      'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                   #      'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                   #      'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))
                   print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                   max_acc = epoch_acc['val_mask'][rep, epoch]
                   record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc, record_test_acc))

    if os.path.isfile(ResultCSV):
        df = pd.read_csv(ResultCSV)
    else:
        outputs_names = {name: type(value).__name__ for (name, value) in args._get_kwargs()}
        outputs_names.update({'Replicate{0:2d}'.format(ii): 'float' for ii in range(1,num_reps+1)})
        outputs_names.update({'Ave_Test_Acc': 'float', 'Test_Acc_std': 'float'})
        df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in outputs_names.items()})

    new_row = {name: value for (name, value) in args._get_kwargs()}
    new_row.update({'Replicate{0:2d}'.format(ii): saved_model_test_acc[ii-1] for ii in range(1,num_reps+1)})
    new_row.update({'Ave_Test_Acc': np.mean(saved_model_test_acc), 'Test_Acc_std': np.std(saved_model_test_acc)})
    df = df.append(new_row, ignore_index=True)
    df.to_csv(ResultCSV, index=False)

    np.savez(SaveResultFilename + '.npz',
             epoch_train_loss=epoch_loss['train_mask'],
             epoch_train_acc=epoch_acc['train_mask'],
             epoch_valid_loss=epoch_loss['val_mask'],
             epoch_valid_acc=epoch_acc['val_mask'],
             epoch_test_loss=epoch_loss['test_mask'],
             epoch_test_acc=epoch_acc['test_mask'],
             saved_model_val_acc=saved_model_val_acc,
             saved_model_test_acc=saved_model_test_acc)

 