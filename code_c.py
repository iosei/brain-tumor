import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torchvision
# import tkinter
from torchvision import datasets, transforms, models
import torch
import os
import random
import time
from torchvision.datasets.utils import download_url
from torchvision.datasets import \
    ImageFolder  # ImageFolder arranges images into root/label/picture.png (each class of image is stored in a different folder)
import torchvision.transforms as tt
from torch.utils.data import random_split
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, cifar100_iid, cifar100_noniid, emnist_iid, \
    emnist_noniid, fruit_iid, fruit_noniid, fashionMnist_iid, fashionMnist_noniid, cars_iid, cars_noniid, caltech_iid, \
    caltech_noniid, svhn_iid, svhn_noniid, usps_iid, usps_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, VGGEmnist, VGGCifar, CNNEmnist, CompCNNMnist, CompCNNEmnist, \
    CNNCifar100, CompCNNFruit, CNNFruit, CNNSvhn, CNNUsps, CompCNNSvhn, CompCNNUsps, CNNfashionMnist, \
    CompCNNFashionMnist  # ResnetMnist #VGGMnist
from models.Fed import FedAvg
from models.test import test_img
from models.train_accuracy import train_img
import pandas as pd
from torch import nn
from efficientnet_pytorch import EfficientNet
# from models.RESNET18 import Block, ResNet
import warnings

warnings.filterwarnings("ignore")
import opendatasets as od
from torch.utils.data import DataLoader, Dataset

# from torchvision.datasets import ImageFolder
# dataloader is used to load data into the program


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        # print(len(dataset_train))
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'usps':
        dataset_train = datasets.USPS('../data/usps/', train=True, download=True, transform=transforms.ToTensor())
        dataset_test = datasets.USPS('../data/usps/', train=False, download=True, transform=transforms.ToTensor())
        if args.iid:
            dict_users = usps_iid(dataset_train, args.num_users)
        else:
            dict_users = usps_noniid(dataset_train, args.num_users)
    elif args.dataset == 'svhn':
        trans_svhn = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()])
        dataset_trains = datasets.SVHN('../data/svhn/', download=True, transform=trans_svhn)
        # transforms.Grayscale(1)
        # print(len(dataset_train))
        test_size = 13257
        train_size = len(dataset_trains) - test_size
        dataset_train, dataset_test = random_split(dataset_trains, [train_size, test_size])
        # sample users
        if args.iid:
            dict_users = svhn_iid(dataset_train, args.num_users)
        else:
            dict_users = svhn_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashionMnist':
        trans_fashion = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('../data/fashionMnist/', train=True, download=True,
                                              transform=trans_fashion)
        dataset_test = datasets.FashionMNIST('../data/fashionMnist/', train=False, download=True,
                                             transform=trans_fashion)
        # transforms.Grayscale(1)
        print(len(dataset_train))
        print(len(dataset_test))
        # transforms.Resize(32),
        # sample users
        if args.iid:
            dict_users = fashionMnist_iid(dataset_train, args.num_users)
        else:
            dict_users = fashionMnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'car':
        trans_cars = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()])
        dataset_train = datasets.StanfordCars('../data/cars/', split='train', download=True, transform=trans_cars)
        dataset_test = datasets.StanfordCars('../data/cars/', split='test', download=True, transform=trans_cars)
        # transforms.Grayscale(1)
        print(len(dataset_train))
        print(dataset_test.size)
        # transforms.Resize(32),
        # sample users
        if args.iid:
            dict_users = cars_iid(dataset_train, args.num_users)
        else:
            dict_users = cars_noniid(dataset_train, args.num_users)

    elif args.dataset == 'vgg_mnist':
        trans_data1 = transforms.Compose([transforms.Grayscale(3), transforms.Resize(224), transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])  # TRANSFORMING IMAGE TO TENSOR
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_data1)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_data1)
        # sample users
        if args.iid:  # Non-iid if input is not given at the terminal
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'effnet_mnist':
        trans_data1 = transforms.Compose([transforms.Grayscale(3), transforms.Resize(224), transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (
                                              0.3081,))])  # TRANSFORMING IMAGE TO TENSOR T.Resize(size=(224,224))
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_data1)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_data1)
        # sample users
        if args.iid:  # Non-iid if input is not given at the terminal
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'resnet_mnist':
        trans_data1 = transforms.Compose([transforms.Grayscale(3), transforms.Resize(224), transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (
                                              0.3081,))])  # TRANSFORMING IMAGE TO TENSOR T.Resize(size=(224,224))
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_data1)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_data1)
        # sample users
        if args.iid:  # Non-iid if input is not given at the terminal
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'emnist':
        trans_data = transforms.Compose([transforms.Resize(224),
                                         lambda img: transforms.functional.rotate(img, -90),
                                         lambda img: transforms.functional.hflip(img),
                                         transforms.ToTensor()
                                         ])
        dataset_train = datasets.EMNIST(root='EMNIST/processed/training.pt', split='byclass', train=True, download=True,
                                        transform=trans_data)
        dataset_test = datasets.EMNIST(root='EMNIST/processed/test.pt', split='byclass', train=False, download=True,
                                       transform=trans_data)
        if args.iid:
            dict_users = emnist_iid(dataset_train, args.num_users)
        else:
            dict_users = emnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'vgg_cifar':
        trans_data1 = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_data1)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_data1)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')

    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar100)
        if args.iid:
            dict_users = cifar100_iid(dataset_train, args.num_users)
        else:
            # exit('Error: only consider IID setting in CIFAR10')
            dict_users = cifar100_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            # exit('Error: only consider IID setting in CIFAR10')
            dict_users = cifar_noniid(dataset_train, args.num_users)

    elif args.dataset == 'fruit':
        # train_loaders, test_loaders, train_all_dataset = fruit_loader(args)
        # return train_loaders, test_loaders, train_all_dataset
        TRANSFORM_IMG = transforms.Compose([transforms.Grayscale(3),
                                            transforms.Resize((100, 100)),
                                            transforms.ToTensor(),
                                            ])
        '''
        dataset_train = ImageFolder(
            '/home/eben/federated-learningV2/data/fruits/fruits-360-original-size/fruits-360-original-size/Training',
            transform=TRANSFORM_IMG)
        dataset_test = ImageFolder(
            '/home/eben/federated-learningV2/data/fruits/fruits-360-original-size/fruits-360-original-size/Test',
            transform=TRANSFORM_IMG)
        '''
        dataset_train = ImageFolder(
            '/home/eben/federated-learningV2/data/fruits/fruits-360_dataset/fruits-360/Training',
            transform=TRANSFORM_IMG)
        dataset_test = ImageFolder('/home/eben/federated-learningV2/data/fruits/fruits-360_dataset/fruits-360/Test',
                                   transform=TRANSFORM_IMG)
        if args.iid:
            dict_users = fruit_iid(dataset_train, args.num_users)
        else:
            dict_users = fruit_noniid(dataset_train, args.num_users)

    elif args.dataset == 'caltech':
        trans_caltech = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.Caltech256('../data/caltech', train=True, download=True, transform=trans_caltech)
        dataset_test = datasets.Caltech256('../data/caltech', train=False, download=True, transform=trans_caltech)
        if args.iid:
            dict_users = caltech_iid(dataset_train, args.num_users)
        else:
            # exit('Error: only consider IID setting in CIFAR10')
            dict_users = caltech_noniid(dataset_train, args.num_users)

    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fashionMnist':
        net_glob = CNNfashionMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'emnist':
        net_glob = CNNEmnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fruit':
        net_glob = CNNFruit(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'svhn':
        net_glob = CNNSvhn(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'usps':
        net_glob = CNNUsps(args=args).to(args.device)  # change to usps
    elif args.model == 'resnet' and args.dataset == 'resnet_mnist':
        net_glob = models.resnet18(pretrained=True).to(args.device)
        num_ftrs = net_glob.fc.in_features
        net_glob.fc = nn.Linear(num_ftrs, 10)
        # net_glob = ResNet18(img_channels=3, num_classes=10):
    elif args.model == 'cnn' and args.dataset == 'cifar100':
        net_glob = CNNCifar100(args=args).to(args.device)
    elif args.model == 'compcnn' and args.dataset == 'mnist':
        net_glob = CompCNNMnist(args=args).to(args.device)
    elif args.model == 'compcnn' and args.dataset == 'fashionMnist':
        net_glob = CompCNNFashionMnist(args=args).to(args.device)
    elif args.model == 'compcnn' and args.dataset == 'svhn':
        net_glob = CompCNNSvhn(args=args).to(args.device)
    elif args.model == 'compcnn' and args.dataset == 'usps':
        net_glob = CompCNNUsps(args=args).to(args.device)
    elif args.model == 'compcnn' and args.dataset == 'emnist':
        net_glob = CompCNNEmnist(args=args).to(args.device)
    elif args.model == 'compcnn' and args.dataset == 'fruit':
        net_glob = CompCNNFruit(args=args).to(args.device)
    # elif args.model == 'resnet' and args.dataset == 'mnist':
    #   net_glob = ResnetMnist(args=args).to(args.device)
    elif args.model == 'effnet' and args.dataset == 'effnet_mnist':
        # net_glob = EfficientNet.from_pretrained('efficientnet-b0').to(args.device)
        net_glob = EfficientNet.from_name('efficientnet-b0').to(args.device)
        # Freeze weights
        # for param in  net_glob.parameters():
        # param.requires_grad = False
        in_features = net_glob._fc.in_features
        # Defining Dense top layers after the convolutional layers
        net_glob._fc = nn.Sequential(
            nn.BatchNorm1d(num_features=in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.4),
            nn.Linear(128, 10),

        )
    elif args.model == 'effnet' and args.dataset == 'emnist':
        # net_glob = EfficientNet.from_pretrained('efficientnet-b0').to(args.device)
        net_glob = EfficientNet.from_name('efficientnet-b0').to(args.device)
        # Freeze weights
        for param in net_glob.parameters():
            param.requires_grad = False
        in_features = net_glob._fc.in_features
        # Defining Dense top layers after the convolutional layers
        net_glob._fc = nn.Sequential(
            nn.BatchNorm1d(num_features=in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.4),
            nn.Linear(128, 62),
        )
    elif args.model == 'vgg' and args.dataset == 'vgg_cifar':
        # net_glob = VGGCifar(args=args).to(args.device)
        net_glob = torchvision.models.vgg11_bn(pretrained=True).to(args.device)  # remove torchvision
        last_layer = nn.Linear(4096, args.num_classes)
        net_glob.classifier[6] = last_layer
    elif args.model == 'vgg' and args.dataset == 'vgg_mnist':
        # net_glob = VGGMnist(args=args).to(args.device)
        net_glob = models.vgg11_bn(pretrained=True).to(args.device)
        last_layer = nn.Linear(4096, args.num_classes)
        net_glob.classifier[6] = last_layer
        # net_glob.load_state_dict(torch.load("../input/vgg11bn/vgg11_bn.pth", map_location=device))
    elif args.model == 'vgg' and args.dataset == 'emnist':
        # net_glob = VGGEmnist(args=args).to(args.device)
        net_glob = models.vgg11_bn(pretrained=True).to(args.device)
        last_layer = nn.Linear(4096, args.num_classes)
        net_glob.classifier[6] = last_layer
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # directory = os.getcwd()
    original_model = copy.deepcopy(net_glob).to(args.device)
    net_glob = net_glob.to(args.device)
    # print(directory)
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    glob_train_loss_acc_list = []
    store_avg_sum_weight_div = []
    clients_divergence = []
    clients_div = []
    global_param = []
    clients_loss = []
    rounds_with_update = []
    # sum_weight_difference = 0
    start_time = time.time()  # Timing training
    store_sum_weight_difference = []
    optim_state = False
    global_optim_state = False
    global_direction_list = []
    sum_first_four = 0
    sum_last_four = 0
    averages_list = []

    all_global_gradients = []
    updated_global_gradients = []
    round_global_gradient = None
    global_gradient = None
    sum_gradients = torch.tensor(0., device=args.device)
    sum_gradients1 = torch.tensor(0., device=args.device)
    clients_weights_update = [user for user in range(args.num_users)]

    first_n_rounds = 4
    threshold_round = 5
    threshold_scale_factor = 2
    last_n_rounds = 4
    initial_threshold_state = False
    avg_grads_ns = None

    fl_round = 0
    while fl_round < args.epochs:  # args.epochs

        sum_weight_difference = torch.tensor(0., device=args.device)
        sum_weight_div = torch.tensor(0., device=args.device)
        global_weight_sum = torch.zeros(200, 784, device=args.device)
        final_global_sum = torch.tensor(0., device=args.device)

        loss_locals = []

        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if fl_round > 0.0:
            temp_v = store_sum_weight_difference[fl_round - 1]
        else:
            temp_v = 0

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        clients_gradients = []

        for idx in idxs_users:

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, gradients = local.train2(copy.deepcopy(net_glob).to(args.device), temp_v, optim_state)

            clients_gradients.append(gradients)
            clients_weights_update[idx] = copy.deepcopy(w)

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # get the record the round gradients
        round_global_gradient = FedAvg(clients_gradients)  # gradient of global model
        all_global_gradients.append(round_global_gradient)  # list
        w_glob = FedAvg(w_locals)  # calculating fedavg on local weights w_glob

        # when current run is greater or equal to the threshold_round  and thresh_state not achieved
        if ((fl_round + 1) > threshold_round) and initial_threshold_state==False:
            print("Checking if condition for gradient direction satisfied")
            gradients_first_n_rounds = all_global_gradients[0:first_n_rounds]
            gradients_last_n_rounds = all_global_gradients[-last_n_rounds:]
            # calculate averge of first_n and last_n gradients
            avg_gradients_first_n_rounds = FedAvg(gradients_first_n_rounds)
            avg_gradients_last_n_rounds = FedAvg(gradients_last_n_rounds)

            # calculate standard deviation
            std_dev = 0
            for key in avg_gradients_first_n_rounds.keys():
                # find std of each layer of the two avgs
                tensors_two = torch.tensor(
                    [torch.sum(avg_gradients_last_n_rounds[key]), torch.sum(avg_gradients_first_n_rounds[key])])
                sum_gradients = torch.std(tensors_two, unbiased=False)
                std_dev = std_dev + sum_gradients

            # confirming the direction of the gradients
            if std_dev.item() <= 0.50:  # defining first std threshold
                initial_threshold_state = True
                # calculate the avg of first_n and last_n together
                avg_grads_ns = FedAvg([avg_gradients_first_n_rounds, avg_gradients_last_n_rounds])
                print("Gradient direction satisfied, leaving checking loop condition \n")
            else:
                #reinitialize all parameters and start round training again to get ideal gradient direction. Take note of fl_round variable and continue
                initial_threshold_state = False
                start_epoch = 0
                loss_locals = []
                all_global_gradients = []
                store_sum_weight_difference = []
                clients_loss = []
                global_param = []
                clients_divergence = []
                loss_train = []
                glob_train_loss_acc_list = []
                rounds_with_update = []
                clients_weights_update = [user for user in range(args.num_users)]
                net_glob = copy.deepcopy(original_model).to(args.device)
                threshold_round = threshold_round * threshold_scale_factor
                print("Gradient direction not satisfied, checking for ", str(threshold_round), " threshold rounds")
                print("Continue training for more rounds and recompute standard deviation using an increased number of preliminary rounds \n")
                fl_round = 0
                continue

        # Starting updating global model based on initial threshold check computation check else skip
        if initial_threshold_state == True:
            # calculate deviation for global model and avg_grads_ns
            global_deviations = 0
            global_gradient = copy.deepcopy(round_global_gradient)
            for key in avg_grads_ns.keys():
                # find std of each layer of the two avgs
                global_param_diff = torch.sum(global_gradient[key] - avg_grads_ns[key])
                global_sum_gradients = global_deviations + global_param_diff
                global_deviations = global_param_diff + sum_gradients

            if global_deviations <= 1.00:  # checking global gradient threshold condition
                # update global weights
                print("Updating global model at round ", str(fl_round + 1))
                net_glob.load_state_dict(w_glob)  # updating global model
                rounds_with_update.append(fl_round + 1)
                global_optim_state =False
            else:
                print("No global model update, will implement loss penalization")
                global_optim_state = True

        # These actions run irrespective of all condtions check but some parameters or variable are initialized based on above conditions
        # Round metrics
        # print global loss for each round
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('\nRound {:3d}, Average loss {:.3f}'.format(fl_round+1, loss_avg))
        loss_train.append([fl_round + 1, loss_avg])
        df = pd.DataFrame(loss_train, columns=['round', 'loss'])
        #df.to_csv("mine_global_loss_cnn_mlp_iid_sup2_lasts_100e.csv", index=False)

        # update global model with no condition check if round is less than or equal to threshold_round
        if ((fl_round + 1) <= threshold_round):
            net_glob.load_state_dict(w_glob)
            rounds_with_update.append(fl_round+1)

        # accuracy of global model
        net_glob.eval()
        acc_global_train, loss_global_train = train_img(net_glob, dataset_train, args)
        glob_train_loss_acc_list.append([fl_round + 1, loss_global_train, acc_global_train.item()])
        df = pd.DataFrame(glob_train_loss_acc_list, columns=['round', 'loss', 'accuracy'])
        #df.to_csv("mine_global_accuracy_mlp_mnist_iid_sup2_lasts_100e.csv", index=False)

        # calculating the norm of global model parameters for each round
        norm_globalM_parameters = 0
        for key in w_glob.keys():
            norm_globalM_parameters += torch.norm(w_glob[key])
        global_param.append([fl_round + 1, norm_globalM_parameters.cpu().numpy()])
        df = pd.DataFrame(global_param, columns=['round', 'global_model_parameters'])
        # df.to_csv("mine_global model parameter_mlp_mnist_iid_sup4_lasts.csv", index=False)

        # Computing loss for each client after every round
        for client_idx, client_loss in enumerate(loss_locals):
            clients_loss.append([fl_round + 1, client_idx + 1, client_loss])
            df = pd.DataFrame(clients_loss, columns=['round', 'client_name', 'loss'])
            #df.to_csv("mine_clients_losses_mlp_mnist_iid_sup2_lasts_100e.csv", index=False)

        # Divergence btn clients' parameters and global parameters
        for idx, client in enumerate(range(0, len(w_locals))):
            for k in w_glob.keys():
                w_div = torch.norm(torch.sub(w_locals[client][k].float(), w_glob[k].float())) / torch.norm(
                    w_locals[client][k].float())
            clients_divergence.append([fl_round + 1, idx + 1, w_div.cpu().numpy()])
            df = pd.DataFrame(clients_divergence, columns=['round', 'client_index', 'parameter divergence'])
            #df.to_csv("mine_clients_parameter_divergence_mlp_mnist_iid_sup2_lasts_100e.csv", index=False)
            # df.to_csv("mine_clients_parameter_divergence_delete.csv", index=False)

        # Tracking clients weight difference for loss penalization
        for client in range(0, len(w_locals)):  # w_locals is a list
            for k in w_glob.keys():
                weight_difference = torch.pow(torch.norm(torch.sub(w_locals[client][k], w_glob[k])), 2)
                sum_weight_difference = sum_weight_difference + weight_difference
        store_sum_weight_difference.append(sum_weight_difference)

        if store_sum_weight_difference[fl_round] < store_sum_weight_difference[fl_round - 1]:
            print("No loss penalization needed")
            optim_state = False
        else:
            print("loss penalization needed")
            optim_state = True

        if global_optim_state == True:
            print("Overriding loss penalization condition with loss penalization")
            optim_state = True

        fl_round = fl_round + 1

    print("Training time in seconds: ", time.time() - start_time)
    print("len_all_global_gradients: ", len(all_global_gradients))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # Save the global model
    # torch.save(net_glob.state_dict(), "mine_cnn_Svhn_iid_sup1_last2.pth")
    # print(store_avg_sum_weight_div)
    # print(store_sum_weight_difference)

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
