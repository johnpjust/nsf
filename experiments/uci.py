import argparse
import json
import numpy as np
import torch
import os
import tensorflow as tf

# from tensorboardX import SummaryWriter
from time import sleep
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from tqdm import tqdm

import data as data_
import nn as nn_
import utils
import datetime
import struct
import glob
import scipy

from experiments import cutils
from nde import distributions, flows, transforms

class parser_:
    pass
args = parser_()
# parser = argparse.ArgumentParser()

# data
# parser.add_argument('--dataset_name', type=str, default='miniboone',
#                     choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'],
#                     help='Name of dataset to use.')
args.dataset_name = 'cifar10'
# parser.add_argument('--train_batch_size', type=int, default=64,
#                     help='Size of batch used for training.')
args.train_batch_size = 64
# parser.add_argument('--val_frac', type=float, default=1.,
#                     help='Fraction of validation set to use.')
args.val_frac = 0.2

# parser.add_argument('--val_batch_size', type=int, default=512,
#                     help='Size of batch used for validation.')
args.val_batch_size=128
# optimization
# parser.add_argument('--learning_rate', type=float, default=3e-4,
#                     help='Learning rate for optimizer.')
args.learning_rate = 3e-4
# parser.add_argument('--num_training_steps', type=int, default=200000,
#                     help='Number of total training steps.')
args.num_training_steps = 200000
# parser.add_argument('--anneal_learning_rate', type=int, default=1,
#                     choices=[0, 1],
#                     help='Whether to anneal the learning rate.')
args.anneal_learning_rate = 1
# parser.add_argument('--grad_norm_clip_value', type=float, default=5.,
#                     help='Value by which to clip norm of gradients.')
args.grad_norm_clip_value = 5.
# flow details
# parser.add_argument('--base_transform_type', type=str, default='rq-autoregressive',
#                     choices=['affine-coupling', 'quadratic-coupling', 'rq-coupling',
#                              'affine-autoregressive', 'quadratic-autoregressive',
#                              'rq-autoregressive'],
#                     help='Type of transform to use between linear layers.')
args.base_transform_type = 'rq-autoregressive'
# parser.add_argument('--linear_transform_type', type=str, default='lu',
#                     choices=['permutation', 'lu', 'svd'],
#                     help='Type of linear transform to use.')
args.linear_transform_type = 'lu'
# parser.add_argument('--num_flow_steps', type=int, default=10,
#                     help='Number of blocks to use in flow.')
args.num_flow_steps = 5
# parser.add_argument('--hidden_features', type=int, default=256,
#                     help='Number of hidden features to use in coupling/autoregressive nets.')
args.hidden_features = 256
# parser.add_argument('--tail_bound', type=float, default=3,
#                     help='Box is on [-bound, bound]^2')
args.tail_bound = 3
# parser.add_argument('--num_bins', type=int, default=8,
#                     help='Number of bins to use for piecewise transforms.')
args.num_bins = 8
# parser.add_argument('--num_transform_blocks', type=int, default=2,
#                     help='Number of blocks to use in coupling/autoregressive nets.')
args.num_transform_blocks=2
# parser.add_argument('--use_batch_norm', type=int, default=0,
#                     choices=[0, 1],
#                     help='Whether to use batch norm in coupling/autoregressive nets.')
args.use_batch_norm = 0
# parser.add_argument('--dropout_probability', type=float, default=0.25,
#                     help='Dropout probability for coupling/autoregressive nets.')
args.dropout_probability = 0
# parser.add_argument('--apply_unconditional_transform', type=int, default=1,
#                     choices=[0, 1],
#                     help='Whether to unconditionally transform \'identity\' '
#                          'features in coupling layer.')
args.apply_unconditional_transform = 1
# logging and checkpoints
# parser.add_argument('--monitor_interval', type=int, default=250,
#                     help='Interval in steps at which to report training stats.')
args.monitor_interval = 25#0
# reproducibility
# parser.add_argument('--seed', type=int, default=1638128,
#                     help='Random seed for PyTorch and NumPy.')
args.seed = 1638128
# args = parser.parse_args()
args.activation = F.tanh
args.stop_cntr = 15
args.step_lim = 5000 ## num of steps after which to record best model

timestamp = str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')
path = os.path.join('checkpoint', '{}_steps{}_baseXfm{}_linXfm{}_h{}_BN{}_Blocks{}_UconXfm{}_{}'.format(
    args.dataset_name,
    args.num_flow_steps, args.base_transform_type, args.linear_transform_type, int(args.hidden_features), int(args.use_batch_norm), args.num_transform_blocks,
    int(args.apply_unconditional_transform),
    timestamp))
try:
    os.mkdir(path)
except:
    pass
run_dir = os.path.join(r'C:\Users\justjo\PycharmProjects\nsf', path)
# SummaryWriter = tf.summary.SummaryWriter

torch.manual_seed(args.seed)
np.random.seed(args.seed)

assert torch.cuda.is_available()
device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# create data
# train_dataset = data_.load_dataset(args.dataset_name, split='train')
# train_loader = data.DataLoader(
#     train_dataset,
#     batch_size=args.train_batch_size,
#     shuffle=True,
#     drop_last=True
# )
# train_generator = data_.batch_generator(train_loader)
# test_batch = next(iter(train_loader)).to(device)
#
# # validation set
# val_dataset = data_.load_dataset(args.dataset_name, split='val', frac=args.val_frac)
# val_loader = data.DataLoader(
#     dataset=val_dataset,
#     batch_size=args.val_batch_size,
#     shuffle=True,
#     drop_last=True
# )
#
# # test set
# test_dataset = data_.load_dataset(args.dataset_name, split='test')
# test_loader = data.DataLoader(
#     dataset=test_dataset,
#     batch_size=args.val_batch_size,
#     shuffle=False,
#     drop_last=False
# )
############## MNIST ####################
# def read_idx(filename):
#     with open(filename, 'rb') as f:
#         zero, data_type, dims = struct.unpack('>HBB', f.read(4))
#         shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
#         return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
#
# dtrain = read_idx(r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/train-images-idx3-ubyte')
# dtrain = dtrain.reshape((dtrain.shape[0],-1))/128. - 1.
# train_idx = np.arange(dtrain.shape[0])
# np.random.shuffle(train_idx)
#
# dtest = read_idx(r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/t10k-images-idx3-ubyte')
# dtest = dtest.reshape((dtest.shape[0],-1))/128. - 1.
#
# # fnames_data = [r'C:\Users\justjo\Downloads\public_datasets/MNIST/train-images.idx3-ubyte', r'C:\Users\justjo\Downloads\public_datasets/MNIST/t10k-images.idx3-ubyte']
# # cont_data = []
# # for f in fnames_data:
# #     cont_data.append(read_idx(f))
# # cont_data = np.concatenate(cont_data)
# cont_data = read_idx(r'C:\Users\justjo\Downloads\public_datasets/MNIST/t10k-images.idx3-ubyte')
# cont_data = cont_data.reshape((cont_data.shape[0],-1))/128. - 1.
# # cont_data = cont_data[np.random.choice(cont_data.shape[0],10000, False), :]
########### CIFAR10 ###############################
fnames_cifar = glob.glob(r'C:\Users\justjo\Downloads\public_datasets\cifar-10-python\cifar-10-batches-py\train\*')
dtrain=[np.load(f, allow_pickle=True, encoding='latin1') for f in fnames_cifar]
dtrain = np.concatenate([a['data'] for a in dtrain])/128. - 1.
train_idx = np.arange(dtrain.shape[0])
np.random.shuffle(train_idx)

dtest = np.load(r'C:\Users\justjo\Downloads\public_datasets\cifar-10-python\cifar-10-batches-py\test\test_batch', allow_pickle=True, encoding='latin1')
dtest = dtest['data']/128. - 1.

cont_data = scipy.io.loadmat(r'C:\Users\justjo\Downloads\public_datasets\SVHN.mat')
cont_data = np.moveaxis(cont_data['X'],3,0)
cont_data = np.reshape(cont_data, (cont_data.shape[0],-1))/128. - 1.
# cont_data = cont_data[np.random.choice(cont_data.shape[0],10000, False), :]
#################################################

train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(dtrain[train_idx[:-int(args.val_frac*dtrain.shape[0])]]).float())
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    drop_last=True
)
train_generator = data_.batch_generator(train_loader)
test_batch = next(iter(train_loader))[0].to(device)

val_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(dtrain[train_idx[-int(args.val_frac*dtrain.shape[0]):]]).float())
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True)
val_loader = data.DataLoader(
    dataset=val_dataset,
    batch_size=args.val_batch_size,
    shuffle=True,
    drop_last=True
)
val_generator = data_.batch_generator(val_loader)

test_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(dtest).float())
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False)
test_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=args.val_batch_size,
    shuffle=False,
    drop_last=False
)
test_generator = data_.batch_generator(test_loader)

cont_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(cont_data).float())
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False)
cont_loader = data.DataLoader(
    dataset=cont_dataset,
    batch_size=args.val_batch_size,
    shuffle=False,
    drop_last=False
)
cont_generator = data_.batch_generator(cont_loader)

features = test_batch.shape[1]
# features = train_dataset.dim

def create_linear_transform():
    if args.linear_transform_type == 'permutation':
        return transforms.RandomPermutation(features=features)
    elif args.linear_transform_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.LULinear(features, identity_init=True)
        ])
    elif args.linear_transform_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.SVDLinear(features, num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError


def create_base_transform(i):
    if args.base_transform_type == 'affine-coupling':
        return transforms.AffineCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=args.activation,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            )
        )
    elif args.base_transform_type == 'quadratic-coupling':
        return transforms.PiecewiseQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=args.activation,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            apply_unconditional_transform=args.apply_unconditional_transform
        )
    elif args.base_transform_type == 'rq-coupling':
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=args.activation,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            apply_unconditional_transform=args.apply_unconditional_transform
        )
    elif args.base_transform_type == 'affine-autoregressive':
        return transforms.MaskedAffineAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=args.activation,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    elif args.base_transform_type == 'quadratic-autoregressive':
        return transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=args.activation,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    elif args.base_transform_type == 'rq-autoregressive':
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=args.activation,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    else:
        raise ValueError

def create_transform():
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(),
            create_base_transform(i)
        ]) for i in range(args.num_flow_steps)
    ] + [
        create_linear_transform()
    ])
    return transform

# create model
distribution = distributions.StandardNormal((features,))
transform = create_transform()
flow = flows.Flow(transform, distribution).to(device)

n_params = utils.get_num_parameters(flow)
print('There are {} trainable parameters in this model.'.format(n_params))

# create optimizer
optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate)
if args.anneal_learning_rate:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, 0)
else:
    scheduler = None

# create summary writer and write to log directory
# timestamp = cutils.get_timestamp()
# if cutils.on_cluster():
#     timestamp += '||{}'.format(os.environ['SLURM_JOB_ID'])
# log_dir = os.path.join(path, args.dataset_name, timestamp)
# while True:
#     try:
#         # writer = SummaryWriter(log_dir=log_dir, max_queue=20)
writer = tf.summary.create_file_writer(run_dir, max_queue=20)
writer.set_as_default()

#         break
#     except FileExistsError:
#         sleep(5)
# filename = os.path.join(log_dir, 'config.json')
# with open(filename, 'w') as file:
#     json.dump(vars(args), file)
with open(os.path.join(run_dir, 'args.json'), 'w') as f:
    json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

tbar = tqdm(range(args.num_training_steps))
best_val_score = -1e10
torch.cuda.empty_cache()
stop_cntr = 0
for step in tbar:
    flow.train()
    if args.anneal_learning_rate:
        scheduler.step(step)
    optimizer.zero_grad()

    batch = next(train_generator)[0].to(device)
    log_density = flow.log_prob(batch)
    loss = - torch.mean(log_density)
    loss.backward()
    if args.grad_norm_clip_value is not None:
        clip_grad_norm_(flow.parameters(), args.grad_norm_clip_value)
    optimizer.step()

    tf.summary.scalar(name='loss', data=loss.item(), step=step)

    ## option #1 for val monitoring
    flow.eval()
    with torch.no_grad():
        val_batch = next(val_generator)[0].to(device)
        log_density_val = flow.log_prob(val_batch[0].to(device).detach())
        mean_log_density_val = torch.mean(log_density_val).detach()
        running_val_log_density = mean_log_density_val.cpu().numpy()

        test_batch = next(test_generator)[0].to(device)
        log_density_test = flow.log_prob(test_batch[0].to(device).detach())
        mean_log_density_test = torch.mean(log_density_test).detach()
        running_test_log_density = mean_log_density_test.cpu().numpy()

        cont_batch = next(cont_generator)[0].to(device)
        log_density_cont = flow.log_prob(cont_batch[0].to(device).detach())
        mean_log_density_cont = torch.mean(log_density_cont).detach()
        running_cont_log_density = mean_log_density_cont.cpu().numpy()

    if running_val_log_density > best_val_score:
        best_val_score = running_val_log_density
        stop_cntr = 0
        if step > args.step_lim:
            model_holder = flow.cpu().state_dict().copy()
            flow.cuda().state_dict()
    else:
        stop_cntr += 1
        if stop_cntr > args.stop_cntr:
            break

    # ## option #2 for val monitoring
    # if (step + 1) % args.monitor_interval == 0:
    #     flow.eval()
    #
    #     with torch.no_grad():
    #         # compute validation score
    #         running_val_log_density = 0
    #         for val_batch in val_loader:
    #             log_density_val = flow.log_prob(val_batch[0].to(device).detach())
    #             mean_log_density_val = torch.mean(log_density_val).detach()
    #             running_val_log_density += mean_log_density_val.cpu().numpy()
    #         running_val_log_density /= len(val_loader)
    #
    #     ####### save best model  #### don't use if want to run faster...or wait until reaching a certain best score
    #     if running_val_log_density > best_val_score:
    #         best_val_score = running_val_log_density
    #         stop_cntr = 0
    #         if step > args.step_lim:
    #             model_holder = flow.cpu().state_dict().copy()
    #             flow.cuda().state_dict()
    #         with torch.no_grad():
    #             running_test_log_density = 0
    #             for test_batch in test_loader:
    #                 log_density_test = flow.log_prob(test_batch[0].to(device).detach())
    #                 mean_log_density_test = torch.mean(log_density_test).detach()
    #                 running_test_log_density += mean_log_density_test.cpu().numpy()
    #             running_test_log_density /= len(test_loader)
    #             running_cont_log_density = 0
    #             for cont_batch in cont_loader:
    #                 log_density_cont = flow.log_prob(cont_batch[0].to(device).detach())
    #                 mean_log_density_cont = torch.mean(log_density_cont).detach()
    #                 running_cont_log_density += mean_log_density_cont.cpu().numpy()
    #             running_cont_log_density /= len(cont_loader)
    #     else:
    #         stop_cntr+= 1
    #         if stop_cntr > args.stop_cntr:
    #             break
        #     # path_ = os.path.join(cutils.get_checkpoint_root(),
        #     #                     '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
        #     path_ = os.path.join(run_dir,
        #                         '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
        #     torch.save(flow.state_dict(), path_)

        # compute reconstruction
        # with torch.no_grad():
        #     test_batch_noise = flow.transform_to_noise(test_batch)
        #     test_batch_reconstructed, _ = flow._transform.inverse(test_batch_noise)
        # errors = test_batch - test_batch_reconstructed
        # max_abs_relative_error = torch.abs(errors / test_batch).max()
        # average_abs_relative_error = torch.abs(errors / test_batch).mean()
        # tf.summary.scalar('max-abs-relative-error',
        #                   max_abs_relative_error.cpu().numpy(), step=step)
        # tf.summary.scalar('average-abs-relative-error',
        #                   average_abs_relative_error.cpu().numpy(), step=step)

        # summaries = {
        #     'val': running_val_log_density.item(),
        #     'best-val': best_val_score.item(),
        #     'max-abs-relative-error': max_abs_relative_error.item(),
        #     'average-abs-relative-error': average_abs_relative_error.item()
        # }
        summaries = {
            'val': running_val_log_density,
            'best-val': best_val_score,
            'test': running_test_log_density,
            'cont_data': running_cont_log_density
            # 'max-abs-relative-error': max_abs_relative_error,
            # 'average-abs-relative-error': average_abs_relative_error
        }
        for summary, value in summaries.items():
            tf.summary.scalar(name=summary, data=value, step=step)


####### load best val model
# path = os.path.join(cutils.get_checkpoint_root(),
#                     '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
flow.load_state_dict(model_holder)
path_ = os.path.join(run_dir,
                    '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
torch.save(model_holder, path_)
# flow.load_state_dict(torch.load(path_))
# flow.eval()
# calculate log-likelihood on test set
with torch.no_grad():
    log_likelihood = torch.Tensor([])
    for batch in tqdm(test_loader):
        log_density = flow.log_prob(batch[0].to(device))
        log_likelihood = torch.cat([
            log_likelihood,
            log_density
        ])
# calculate log-likelihood on contrastive set
with torch.no_grad():
    log_likelihood_cont = torch.Tensor([])
    for batch in tqdm(cont_loader):
        log_density = flow.log_prob(batch[0].to(device))
        log_likelihood_cont = torch.cat([
            log_likelihood_cont,
            log_density
        ])

path_ = os.path.join(run_dir, '{}-{}-log-likelihood.npy'.format(
    args.dataset_name,
    args.base_transform_type
))
np.save(path_, utils.tensor2numpy(log_likelihood))
mean_log_likelihood = log_likelihood.mean()
std_log_likelihood = log_likelihood.std()

# save log-likelihood
s = 'Final score for {}: {:.2f} +- {:.2f}'.format(
    args.dataset_name.capitalize(),
    mean_log_likelihood.item(),
    2 * std_log_likelihood.item() / np.sqrt(len(test_dataset))
)
print(s)
filename = os.path.join(run_dir, 'test-results.txt')
with open(filename, 'w') as file:
    file.write(s)
