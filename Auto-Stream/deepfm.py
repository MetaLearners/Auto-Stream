import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class Network(nn.Module):

    def __init__(self, embedding_dim, sparse_cat_num, dense_dim, hidden_size):
        '''
        embedding_dim (int): embedding output dimension of sparse features. 
        sparse_cat_num (list of int): number of categories in each sparse feature. 
        dense_dim (int): dimension of dense features. 
        hidden_size (list of int): hidden layer dimension of DNN part. 
        '''
        super(Network, self).__init__()

        self.embedding_dim = embedding_dim
        self.sparse_cat_num = sparse_cat_num
        self.dense_dim = dense_dim
        self.hidden_size = hidden_size

        #self.first_order_embed = nn.ModuleList([nn.Embedding(cat_num, 1) for cat_num in sparse_cat_num])
        self.second_order_embed = nn.ModuleList([nn.Embedding(cat_num, self.embedding_dim) for cat_num in sparse_cat_num])

        #self.lr_dense = nn.Linear(dense_dim, 1, bias=False)
        #self.lr_bias = nn.Parameter(torch.randn(1))

        self.fc = nn.ModuleList()
        for i, size in enumerate(hidden_size):
            if (i == 0):
                self.fc.append(nn.Linear(embedding_dim * len(sparse_cat_num) + dense_dim, size))
                #self.fc.append(nn.Linear(embedding_dim * len(sparse_cat_num), size))
            else:
                self.fc.append(nn.Linear(hidden_size[i - 1], size))
        self.dnn_output_layer = nn.Linear(hidden_size[-1], 1)


    def forward(self, X_sparse, X_dense):
        '''
        X_sparse (np.array): sparse input features of shape [sample_num, sparse_feature_dim]
        X_dense (np.array): dense input features of shape [sample_num, dense_feature_dim]
        '''
        X_sparse_tensor = torch.tensor(X_sparse, dtype=torch.long).cuda()
        X_dense_tensor = torch.tensor(X_dense, dtype=torch.float).cuda()
        # compute embedding
        embeddings = [embed(X_sparse_tensor[:, i]) for i, embed in enumerate(self.second_order_embed)]

        # FM part
        '''
        lr_sparse = sum([embed(X_sparse_tensor[:, i]) for i, embed in enumerate(self.first_order_embed)])
        lr_dense = self.lr_dense(X_dense_tensor)
        lr_output = lr_sparse + lr_dense
        #lr_output = lr_sparse + lr_dense + self.lr_bias
        #lr_input = torch.cat([X_sparse_one_hot, X_dense_tensor], dim=1)
        #lr_output = self.lr(lr_input)
        #embedding_3D = torch.stack(embeddings, dim=2)
        sum_square = sum(embeddings).pow(2).sum(1)
        square_sum = sum([emb.pow(2).sum(1) for emb in embeddings])
        second_order_term = (0.5 * (sum_square - square_sum)).unsqueeze(1)
        fm_output = lr_output + second_order_term
        '''
        # DNN part
        if (len(embeddings) != 0):
            embedding_2D = torch.cat(embeddings, dim=1)
            #X_fc = embedding_2D
            X_fc = torch.cat([embedding_2D, X_dense_tensor], dim=1)
        else:
            X_fc = X_dense_tensor
        for layer in self.fc:
            X_fc = F.relu(layer(X_fc))
        dnn_output = self.dnn_output_layer(X_fc)

        #final_output = torch.sigmoid(fm_output + dnn_output)
        final_output = torch.sigmoid(dnn_output)
        return final_output


class DeepFM:

    def __init__(self, embedding_dim, sparse_cat_num, dense_dim, hidden_size):

        self.net = Network(embedding_dim, sparse_cat_num, dense_dim, hidden_size).cuda()
        self.loss = F.binary_cross_entropy
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)


    def fit(self, X_sparse, X_dense, y, epoch=10, batch_size=1024, 
            use_validation=True, validation_ratio=0.1, early_stop_epoch=20):

        if use_validation:
            X_sparse_train, X_sparse_valid, X_dense_train, X_dense_valid, y_train, y_valid = train_test_split(
                X_sparse, X_dense, y, test_size=validation_ratio, random_state=0, stratify=y)
        else:
            X_sparse_train, X_dense_train, y_train = X_sparse, X_dense, y

        index = np.arange(y_train.shape[0])
        batch_num = math.ceil(y_train.shape[0] / batch_size)
        logloss_history, logloss_best, no_improvement_epoch = [], 100., 0
        for i in range(epoch):
            np.random.seed(i)
            np.random.shuffle(index)
            X_sparse_train, X_dense_train, y_train = X_sparse_train[index], X_dense_train[index], y_train[index]

            for j in range(batch_num):
                batch_start = j * batch_size
                batch_end = (j + 1) * batch_size
                X_sparse_batch = X_sparse_train[batch_start:batch_end]
                X_dense_batch = X_dense_train[batch_start:batch_end]
                y_train_batch = y_train[batch_start:batch_end]
                self.optimizer.zero_grad()
                outputs = self.net(X_sparse_batch, X_dense_batch)
                loss = self.loss(outputs.squeeze(1), torch.tensor(y_train_batch, dtype=torch.float).cuda())
                loss.backward()
                self.optimizer.step()
            
            if use_validation:
                auc, logloss = self.evaluate(X_sparse_valid, X_dense_valid, y_valid)
                print ('=== epoch %d, validtion auc = %.4f, loss = %.4f ===' %(i, auc, logloss))
                logloss_history.append(logloss)
                if (logloss < logloss_best):
                    logloss_best = logloss
                    no_improvement_epoch = 0
                else:
                    no_improvement_epoch += 1
                if no_improvement_epoch >= early_stop_epoch:
                    print ('early stop at epoch %d' %(i))
                    break


    def evaluate(self, X_sparse, X_dense, y, batch_size=1024):

        with torch.no_grad():
            pred = []
            batch_num = math.ceil(X_sparse.shape[0] / batch_size)
            for i in range(batch_num):
                X_sparse_batch = X_sparse[(i * batch_size):((i + 1) * batch_size)]
                X_dense_batch = X_dense[(i * batch_size):((i + 1) * batch_size)]
                y_pred = self.net(X_sparse_batch, X_dense_batch)
                pred.append(y_pred)
            pred = torch.cat(pred, 0).squeeze(1)
            loss = float(self.loss(pred, torch.tensor(y, dtype=torch.float).cuda()))
            pred = pred.cpu().detach().numpy()
            auc = roc_auc_score(y, pred)
        return auc, loss


    def predict(self, X_sparse, X_dense, batch_size=1024):

        with torch.no_grad():
            pred = []
            batch_num = math.ceil(X_sparse.shape[0] / batch_size)
            for i in range(batch_num):
                X_sparse_batch = X_sparse[(i * batch_size):((i + 1) * batch_size)]
                X_dense_batch = X_dense[(i * batch_size):((i + 1) * batch_size)]
                y = self.net(X_sparse_batch, X_dense_batch).cpu().detach().numpy().ravel()
                pred.append(y)
            pred = np.concatenate(pred)
        return pred



if __name__ == '__main__':
    
    sample_num = 2000
    sparse_cat_num = [10, 5]
    X_sparse_1 = np.random.choice(10, sample_num).reshape(-1, 1)
    X_sparse_2 = np.random.choice(5, sample_num).reshape(-1, 1)
    X_sparse = np.concatenate([X_sparse_1, X_sparse_2], axis=1)
    X_dense = np.random.random([sample_num, 2])
    y_train = np.random.choice(2, sample_num)

    net = DeepFM(8, sparse_cat_num, 2, [32, 32])
    net.fit(X_sparse, X_dense, y_train)
    net.predict(X_sparse, X_dense)
