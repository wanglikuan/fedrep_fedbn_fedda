
import torch
from torch import nn
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
from FLAlgorithms.trainmodel.models import *
import copy

import torch.nn.functional as F
from scipy import linalg


class Server:
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate ,beta, lamda,
                 num_glob_iters, local_epochs, optimizer,num_users, times):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.times = times

        self.model_list = []
        for _ in range(self.num_users):
            self.model_list.append(CNNTarget().to(torch.device("cuda:0"))) #Net #CNNCifar #CNNTarget

        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        #self.send_parameters()

    def avg_dual_att(self, epsilon=1.0, rho=0.1): #代替 aggregate_parameters 进行参数聚合
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
        for idx_local_user, local_user in enumerate(self.selected_users):

            w_server = copy.deepcopy(local_user.model.state_dict())
            w_clients = []
            for idx_user, user in enumerate(self.selected_users):
                w_clients.append(copy.deepcopy(user.model.state_dict()))

            w_next = copy.deepcopy(w_server)
            att = {}

            for k in w_server.keys(): #k代表server model参数中的一层
                w_next[k] = torch.zeros_like(w_server[k])
                att[k] = torch.zeros(len(w_clients)) #att[k]中有clients数量的元素，存储 每一个client中的k层 与 server的k层距离（也就是权重）

            for k in w_next.keys():
                most_similar = 1000.0
                for i in range(0, len(w_clients)):
                    att[k][i] = torch.from_numpy(np.array(linalg.norm(w_server[k].cpu() - w_clients[i][k].cpu())))#用linalg计算 server的k层 与 第i个client的k层 的距离
                    #att[k][i] = loss_fn(w_server[k].cpu(), w_clients[i][k].cpu()) #MSE_loss    
                    print('type(att[k][i].item())',type(att[k][i].item()))                
                    if (att[k][i].item() < most_similar) & (i != idx_local_user):
                        most_similar = att[k][i]
                att[k][idx_local_user] = most_similar*0.1

            for i, k in enumerate(w_next.keys()): #将att[k]中的元素进行1/x的处理，更改的地方：1. model_list 中的model；2.是否进行1/x处理；3.算距离的算法
                att[k] = F.softmax((1/att[k]), dim=0)

            # for i, k in enumerate(w_next.keys()):
            #     att[k] = F.softmax(att[k], dim=0)

            #对应git上的update rules
            for k in w_next.keys(): #按model中的层数进行每一次循环，k代表model中的一层
                att_weight = torch.zeros_like(w_server[k])
                for i in range(0, len(w_clients)):
                    att_weight += torch.mul(w_server[k] - w_clients[i][k], att[k][i]) #此时 att_weight = clients与server距离求和 * 权重；att[k][i] = 权重alpha_c
                print('att in avg_dual_att:', att)
                w_next[k] = w_server[k] - torch.mul(att_weight, epsilon) #w_next = w^(t+1); w_server = w^(t); 
            #self.model.load_state_dict(w_next)
            self.model_list[idx_local_user].load_state_dict(w_next)

        return None

    # def avg_dual_att(self, epsilon=1.0, rho=0.1): #代替 aggregate_parameters 进行参数聚合
    #     #TO DO
    #     w_server = copy.deepcopy(self.model.state_dict())
    #     w_clients = []
    #     for idx_user, user in enumerate(self.selected_users):
    #         w_clients.append(copy.deepcopy(user.model.state_dict()))


    #     w_next = copy.deepcopy(w_server)
    #     att = {}
    #     #att_warm = {}
    #     for k in w_server.keys(): #k代表server model参数中的一层
    #         w_next[k] = torch.zeros_like(w_server[k])
    #         att[k] = torch.zeros(len(w_clients)) #att[k]中有clients数量的元素，存储 每一个client中的k层 与 server的k层距离（也就是权重）

    #     for k in w_next.keys():
    #         for i in range(0, len(w_clients)):
    #             att[k][i] = torch.from_numpy(np.array(linalg.norm(w_server[k].cpu() - w_clients[i][k].cpu())))#用linalg计算 server的k层 与 第i个client的k层 的距离
    #         #sw_diff = w_server[k].cpu() - warm_server[k].cpu()
    #         #att_warm[k] = torch.FloatTensor(np.array(linalg.norm(sw_diff)))

    #     #warm_tensor = torch.FloatTensor([v for k, v in att_warm.items()])
    #     #layer_w = F.softmax(warm_tensor, dim=0)

    #     for i, k in enumerate(w_next.keys()):
    #         att[k] = F.softmax(att[k], dim=0)
    #         #att_warm[k] = layer_w[i]

    #     #对应git上的update rules
    #     for k in w_next.keys(): #按model中的层数进行每一次循环，k代表model中的一层
    #         att_weight = torch.zeros_like(w_server[k])
    #         for i in range(0, len(w_clients)):
    #             att_weight += torch.mul(w_server[k] - w_clients[i][k], att[k][i]) #此时 att_weight = clients与server距离求和 * 权重；att[k][i] = 权重alpha_c

    #         print('att in avg_dual_att:', att)
    #         #att_weight += torch.mul(w_server[k] - warm_server[k], rho*att_warm[k]) #warm_server = w_Q

    #         w_next[k] = w_server[k] - torch.mul(att_weight, epsilon) #w_next = w^(t+1); w_server = w^(t); 

    #     self.model.load_state_dict(w_next)
    #     return None

    def send_att_parameters(self): #代替 send_parameters 进行参数分发
        #TO DO
        assert (self.users is not None and len(self.users) > 0)
        for idx, user in enumerate(self.users):
            #user.set_parameters(self.model)            
            user.set_parameters(self.model_list[idx])
        return None

    ######## origin code ########
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self,user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio


    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()
        
        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b"+ "_" + str(self.local_epochs)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc_per) != 0 &  len(self.rs_train_acc_per) & len(self.rs_train_loss_per)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.close()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()  
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ",train_loss)

    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()  
        stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()  
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)
