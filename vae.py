# /**
#  * @author Shashank Gupta
#  * @email s.gupta2@uva.nl
#  * @create date 2021-05-27 17:29:29
#  * @modify date 2021-05-27 17:29:29
#  * @desc Matrix Factorization code 
#  */


import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from utils import rank_metrics
from torch.optim import Adam
import numpy as np
import torch.distributions as td
import itertools
import operator
import itertools as it
from power_spherical.power_spherical import HypersphericalUniform, MarginalTDistribution, PowerSpherical

# Define VAE with user and item features Model
class VAEBase(LightningModule):
    '''
    Variational AE based MF with user and item covariates as input
    '''
    def __init__(self, num_users, num_items, lf_dim, lf_dim1, lf_dim2,\
                user_feats, item_feats, \
                lr=1e-4, topk=5, reg=1e-5, ips=False):
        super().__init__()
        self.automatic_optimization = False
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=lf_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=lf_dim)
        # nn.init.xavier_normal_(self.user_embedding.weight)
        # nn.init.xavier_normal_(self.item_embedding.weight)
        # Initialize user and item embedding from user and item covariates
        self.user_embedding.weight.data.copy_(torch.from_numpy(user_feats))
        self.item_embedding.weight.data.copy_(torch.from_numpy(item_feats))
        #freeze user and item embedding matrix
        #self.user_embedding.weight.requires_grad = False
        #self.item_embedding.weight.requires_grad = False
        # network hyper-params
        
        self.user_feat_dim = user_feats.shape[1]
        self.item_feat_dim = item_feats.shape[1]
        # projection layers for click input and feat inputs
        
        self.item_feat_map = nn.Linear(self.item_feat_dim, lf_dim1)
        self.user_feat_map = nn.Linear(self.user_feat_dim, lf_dim1)

        self.item_feat_map1 = nn.Linear(lf_dim1, 2 * lf_dim2)
        self.user_feat_map1 = nn.Linear(lf_dim1, 2 * lf_dim2)

        self.activation = nn.Tanh()
        #self.output = nn.Sigmoid()
        self.lr = lr
        self.topk = topk
        #self.loss = nn.BCEWithLogitsLoss()
        self.reg = reg
        # MF with or w/o IPS
        self.ips = ips
        

    def l2_regularize(self, array):
        return torch.sum(array ** 2.0)

    def forward(self, uid, pid):
        # get user and item feat vectors from respective embedding matrices
        user_embedded = self.user_embedding(uid)
        item_embedded = self.item_embedding(pid)
        
        user_embedding = self.activation(self.user_feat_map(user_embedded))
        item_embedding = self.activation(self.item_feat_map(item_embedded))

        user_embedding = self.activation(self.user_feat_map1(user_embedding))
        item_embedding = self.activation(self.item_feat_map1(item_embedding))

        mu_u, logvar_u = torch.chunk(user_embedding, chunks=2, dim=1)
        mu_i, logvar_i = torch.chunk(item_embedding, chunks=2, dim=1)

        std_u = logvar_u.exp().pow(0.5)
        std_i = logvar_i.exp().pow(0.5)

        q_z_u = td.normal.Normal(mu_u, std_u)
        q_z_i = td.normal.Normal(mu_i, std_i)

        z_u = q_z_u.rsample()
        z_i = q_z_i.rsample()
        ui_factor = torch.sum(z_u * z_i, dim=-1)
        #pred = ui_factor + user_bias + item_bias + self.bias
        pred = ui_factor #+ self.bias
        #logit = self.output(pred)
        return pred, z_u, z_i, q_z_i, q_z_u, mu_u, mu_i

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        EPS = 1e-7
        opt = self.optimizers()
        uid, pid, click, prop, user_click_vec, item_click_vec = batch
        logits, z_u, z_i, q_z_i, q_z_u, _, _ = self(uid, pid)
        
        logits = torch.sigmoid(logits)
        if not self.ips:
            ce_loss = click * torch.log(logits + EPS) + (1 - click) * torch.log(1 - logits)
        else:
            ce_loss = (click/prop) * torch.log(logits + EPS) + (1 - click/prop) * torch.log(1 - logits + EPS)
        ce_loss = -torch.mean(ce_loss)
        p_z_u = td.normal.Normal(torch.zeros_like(q_z_u.loc), torch.ones_like(q_z_u.scale))
        KLD_u = td.kl_divergence(q_z_u, p_z_u).sum()

        p_z_i = td.normal.Normal(torch.zeros_like(q_z_i.loc), torch.ones_like(q_z_i.scale))
        KLD_i = td.kl_divergence(q_z_i, p_z_i).sum()
        #loss = self.loss(logits, click) + self.reg * (reg_user + reg_item)
        loss = ce_loss + KLD_u + KLD_i
        self.log("training loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        uid, pid, click, prop, user_click_vec, item_click_vec, all_pids = batch
        batch_size = uid.shape[0]
        n_items = all_pids.shape[1]
        # Generate ranking score over all items for each user in the validation set
        uid_flat = uid.unsqueeze(dim=-1).expand(uid.shape[0], all_pids.shape[1]).flatten()
        pid_flat = all_pids.flatten()
        pred, z_u, z_i, q_z_i, q_z_u, mu_u, mu_i = self(uid_flat, pid_flat)
        # get predictions for users in the batch and all items using mean of the variational distrs
        user_all_scores = torch.sum(mu_u * mu_i, dim=-1).reshape(batch_size,  n_items)
        #user_all_scores = (user_embeddings @ all_item_embeddings.T) #+  self.bias
        user_item_ranking = torch.argsort(user_all_scores, dim=1, descending=True)[:, :self.topk]
        # Create dictionary to store (uid, dcgs) pairs
        dcg_dict = {}
        val_rank = (user_item_ranking == torch.unsqueeze(pid, dim=1)).int()
        val_rank = val_rank.detach().numpy()
        #map_val = rank_metrics.mean_average_precision(val_rank)
        map_val = list(map(lambda x: rank_metrics.average_precision(x), val_rank))
        map_dict = list(zip(uid.detach().numpy(), map_val, prop.detach().numpy()))
        ndcg_val = list(map(lambda x: rank_metrics.dcg_at_k(x, 5), val_rank))
        dcg_dict = list(zip(uid.detach().numpy(), ndcg_val, prop.detach().numpy() ))
        prop_dict = list(zip(uid.detach().numpy(), prop.detach().numpy()))
        #logits = self(uid, pid)
        #val_loss = F.cross_entropy(y_hat, y)
        log_metric = {'map_val': map_dict, 'ndcg_val': dcg_dict, 'prop_batch': prop_dict}
        #self.log("ndcg_val", ndcg_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log_metric

    def _reduce_metric(self, outputs):
        '''
        For input of form [(uid, uid_metric_Val)], reduce
        '''
        map_final, ndcg_final = [], []
        user_maps = [map_list['map_val'] for map_list in outputs]
        user_maps_flat = list(itertools.chain(*user_maps))
        for key, group in itertools.groupby(sorted(user_maps_flat), operator.itemgetter(0)):
            map_per_user = 0.0
            for item in group:
                map_per_user += item[1]
            map_final.append(map_per_user)

        user_ndcg = [map_list['ndcg_val'] for map_list in outputs]
        user_ndcg_flat = list(itertools.chain(*user_ndcg))
        for key, group in itertools.groupby(sorted(user_ndcg_flat), operator.itemgetter(0)):
            ndcg_per_user = 0.0
            for item in group:
                ndcg_per_user += item[1]
            ndcg_final.append(ndcg_per_user)

        return np.mean(map_final), np.mean(ndcg_final)

    def _reduce_metric_val(self, outputs):
        '''
        For input of form [(uid, uid_metric_Val)], reduce
        '''
        map_final, ndcg_final = [], []
        user_maps = [map_list['map_val'] for map_list in outputs]
        user_maps_flat = list(itertools.chain(*user_maps))
        for key, group in itertools.groupby(sorted(user_maps_flat), operator.itemgetter(0)):
            map_per_user = 0.0
            prop_per_user = 0.0
            for item in group:
                map_per_user   += item[1]/item[2]
                prop_per_user  += 1/item[2]
            #try:
            map_final.append(map_per_user/prop_per_user)
            #except:
            #map_final.append(0.)

        user_ndcg = [map_list['ndcg_val'] for map_list in outputs]
        user_ndcg_flat = list(itertools.chain(*user_ndcg))
        for key, group in itertools.groupby(sorted(user_ndcg_flat), operator.itemgetter(0)):
            ndcg_per_user = 0.0
            prop_per_user = 0.0
            for item in group:
                ndcg_per_user += item[1]/item[2]
                prop_per_user += 1/item[2]
            #try:
            ndcg_final.append(ndcg_per_user/prop_per_user)
            #except:
            #    ndcg_final.append(0.)

        return np.mean(map_final), np.mean(ndcg_final)

    def validation_epoch_end(self, outputs):
        map_val, ndcg_val = self._reduce_metric_val(outputs)
        self.log("ptl/val_ndcg", ndcg_val)
        self.log("ptl/val_map", map_val)

    def test_step(self, batch, batch_idx):
        uid, pid, click, prop, user_click_vec, item_click_vec, all_pids = batch
        batch_size = uid.shape[0]
        n_items = all_pids.shape[1]
        # Generate ranking score over all items for each user in the validation set
        uid_flat = uid.unsqueeze(dim=-1).expand(uid.shape[0], all_pids.shape[1]).flatten()
        pid_flat = all_pids.flatten()
        pred, z_u, z_i, q_z_i, q_z_u, mu_u, mu_i = self(uid_flat, pid_flat)
        # get predictions for users in the batch and all items using mean of the variational distrs
        user_all_scores = torch.sum(mu_u * mu_i, dim=-1).reshape(batch_size,  n_items)
        user_item_ranking = torch.argsort(user_all_scores, dim=-1, descending=True)[:, :self.topk]
        test_rank = (user_item_ranking == torch.unsqueeze(pid, dim=1)).int()
        test_rank = test_rank.detach().numpy()
        map_test = list(map(lambda x: rank_metrics.average_precision(x), test_rank))
        map_dict = list(zip(uid.detach().numpy(), map_test))
        ndcg_test = list(map(lambda x: rank_metrics.dcg_at_k(x, 5), test_rank))
        dcg_dict = list(zip(uid.detach().numpy(), ndcg_test))
        #logits = self(uid, pid)
        #val_loss = F.cross_entropy(y_hat, y)
        log_metric = {'map_val': map_dict, 'ndcg_val': dcg_dict, 'batch_lens': len(test_rank)}
        #self.log("ndcg_val", ndcg_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log_metric

    def test_epoch_end(self, outputs):
        map_val, ndcg_val = self._reduce_metric(outputs)
        self.log("test/ndcg_test", ndcg_val)
        self.log("test/map_test", map_val)


# Define MF Model
class VAE(LightningModule):
    '''
    Variational AE based MF with user and item covariates as input
    '''
    def __init__(self, num_users, num_items, lf_dim,\
                user_feats, item_feats, lf_item_click, lf_item_feat,\
                lf_user_click, lf_user_feat,\
                dropout_prob=0.5, lr=1e-3, topk=100, reg=1e-5, ips=False):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=lf_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=lf_dim)
        # Initialize user and item embedding from user and item covariates
        self.user_embedding.weight.data.copy_(torch.from_numpy(user_feats))
        self.item_embedding.weight.data.copy_(torch.from_numpy(item_feats))
        # freeze user and item embedding matrix
        self.user_embedding.weight.requires_grad = False
        self.item_embedding.weight.requires_grad = False
        # network hyper-params
        # embedding dim for item click feat. vector 
        self.lf_item_click = lf_item_click
        self.lf_item_feat = lf_item_feat
        self.lf_user_click = lf_user_click
        self.lf_user_feat = lf_user_feat
        self.dropout = dropout_prob
        self.user_feat_dim = user_feats.shape[1]
        self.item_feat_dim = item_feats.shape[1]
        # projection layers for click input and feat inputs
        self.item_click_map = nn.Linear(num_users, self.lf_item_click)
        self.user_click_map = nn.Linear(num_items, self.lf_user_click)
        self.item_feat_map = nn.Linear(self.item_feat_dim, self.lf_item_feat)
        self.user_feat_map = nn.Linear(self.user_feat_dim, self.lf_user_feat)
        self.activation = nn.Tanh()
        #self.output = nn.Sigmoid()
        self.lr = lr
        self.topk = topk
        #self.loss = nn.BCEWithLogitsLoss()
        self.reg = reg
        # MF with or w/o IPS
        self.ips = ips

    def l2_regularize(self, array):
        return torch.sum(array ** 2.0)

    def forward(self, uid, pid, user_click_vec, item_click_vec):
        # get user and item feat vectors from respective embedding matrices
        user_embedded = self.user_embedding(uid)
        item_embedded = self.item_embedding(pid)
        user_click_norm = F.normalize(user_click_vec)
        item_click_norm = F.normalize(item_click_vec)
        user_click_norm = self.dropout(user_click_norm)
        item_click_norm = self.dropout(item_click_norm)
        user_click_norm = self.activation(self.user_click_map(user_click_norm))
        item_click_norm = self.activation(self.item_click_map(item_click_norm))
        # mapping 

        ui_factor = torch.sum(user_embedded * item_embedded, dim=-1)
        pred = ui_factor + user_bias + item_bias + self.bias
        #pred = ui_factor #+ self.bias
        #logit = self.output(pred)
        return pred

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        EPS = 1e-7
        uid, pid, click, prop, user_click_vec, item_click_vec = batch
        logits = self(uid, pid)
        reg_user = self.l2_regularize(self.bias_user.weight) 
        reg_item = self.l2_regularize(self.bias_item.weight) 
        logits = torch.sigmoid(logits)
        if not self.ips:
            ce_loss = click * torch.log(logits + EPS) + (1 - click) * torch.log(1 - logits)
        else:
            ce_loss = (click/prop) * torch.log(logits + EPS) + (1 - click/prop) * torch.log(1 - logits + EPS)
        ce_loss = -torch.mean(ce_loss)
        #loss = self.loss(logits, click) + self.reg * (reg_user + reg_item)
        loss = ce_loss + self.reg * (reg_user + reg_item)
        self.log("training loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        uid, pid, click, prop = batch
        # Generate ranking score over all items for each user in the validation set
        user_embeddings = self.user_embedding(uid)
        user_bias = self.bias_user(uid)
        all_item_bias = self.bias_item.weight
        all_item_embeddings = self.item_embedding.weight
        user_all_scores = (user_embeddings @ all_item_embeddings.T) + user_bias + all_item_bias.T + self.bias
        #user_all_scores = (user_embeddings @ all_item_embeddings.T) #+  self.bias
        user_item_ranking = torch.argsort(user_all_scores, dim=1, descending=True)[:, :self.topk]
        val_rank = (user_item_ranking == torch.unsqueeze(pid, dim=1)).int()
        val_rank = val_rank.detach().numpy()
        #map_val = rank_metrics.mean_average_precision(val_rank)
        map_val = np.sum(list(map(lambda x: rank_metrics.average_precision(x), val_rank)))
        ndcg_val = np.sum(list(map(lambda x: rank_metrics.dcg_at_k(x, 5), val_rank)))
        #logits = self(uid, pid)
        #val_loss = F.cross_entropy(y_hat, y)
        log_metric = {'map_val': map_val, 'ndcg_val': ndcg_val, 'batch_lens': len(val_rank)}
        #self.log("ndcg_val", ndcg_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log_metric

    def validation_epoch_end(self, outputs):
        map_val = np.stack([x['map_val'] for x in outputs]).sum()/ np.stack([x['batch_lens'] for x in outputs]).sum()
        ndcg_val = np.stack([x['ndcg_val'] for x in outputs]).sum()/ np.stack([x['batch_lens'] for x in outputs]).sum()
        self.log("ptl/val_ndcg", ndcg_val)
        self.log("ptl/val_map", map_val)

    def test_step(self, batch, batch_idx):
        uid, pid, click, prop = batch
        #print(click)
        # Generate ranking score over all items for each user in the validation set
        user_embeddings = self.user_embedding(uid)
        user_bias = self.bias_user(uid)
        all_item_bias = self.bias_item.weight
        all_item_embeddings = self.item_embedding.weight
        #user_all_scores = (user_embeddings @ all_item_embeddings.T)
        user_all_scores = (user_embeddings @ all_item_embeddings.T) + user_bias + all_item_bias.T + self.bias
        user_item_ranking = torch.argsort(user_all_scores, dim=-1, descending=True)[:, :self.topk]
        test_rank = (user_item_ranking == torch.unsqueeze(pid, dim=1)).int()
        test_rank = test_rank.detach().numpy()
        map_test = np.sum(list(map(lambda x: rank_metrics.average_precision(x), test_rank)))
        ndcg_test = np.sum(list(map(lambda x: rank_metrics.dcg_at_k(x, 5), test_rank)))
        #logits = self(uid, pid)
        #val_loss = F.cross_entropy(y_hat, y)
        log_metric = {'map_test': map_test, 'ndcg_test': ndcg_test, 'batch_lens': len(test_rank)}
        #self.log("Test Metrics", {'map_test': map_test, 'ndcg_val': ndcg_test}, on_step=False, on_epoch=True, prog_bar=None, logger=True)
        return log_metric

    def test_epoch_end(self, outputs):
        map_val = np.stack([x['map_test'] for x in outputs]).sum()/ np.stack([x['batch_lens'] for x in outputs]).sum()
        ndcg_val = np.stack([x['ndcg_test'] for x in outputs]).sum()/ np.stack([x['batch_lens'] for x in outputs]).sum()
        self.log("test/ndcg_test", ndcg_val)
        self.log("test/map_test", map_val)


# Define VAE with user and item Click vectors
class VAEClick(LightningModule):
    '''
    Variational AE based MF with user and item covariates as input
    '''
    def __init__(self, num_users, num_items, \
                lfs_user, lfs_item, lf_pos,
                lr=3 * 1e-3, clip=0.05, topk=5, ips=False, gpu=True):
        '''
        args:
            lfs_user: User-side n/w dims 
            lfs_item: Item-side n/w dims
            lf_pos: Latent dimensions of the posteriors on user and item side
        '''
        super().__init__()
        self.automatic_optimization = False
        # torch.autograd.detect_anomaly
        # torch.autograd.set_detect_anomaly(True)

        # Initialize posterior params for users and items randomly
        self.theta = torch.randn(num_users, lf_pos) 
        self.beta = torch.randn(num_items, lf_pos) 
        nn.init.xavier_normal_(self.theta)
        nn.init.xavier_normal_(self.beta)
        if gpu:
            self.theta = self.theta.to('cuda')
            self.beta = self.beta.to('cuda')
        # store mu's for user and items
        self.mu_u = torch.zeros(num_users, lf_pos) 
        self.mu_i = torch.zeros(num_items, lf_pos) 
        if gpu:
            self.mu_u = self.mu_u.to('cuda')
            self.mu_i = self.mu_i.to('cuda')
        # Define user-side network
        self.user_encoder = nn.Sequential()
        for i in range(0, len(lfs_user)):
            # for first layer, inp dim is #items
            if i == 0:
                self.user_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(num_items, lfs_user[i]),
                )
            else:
                self.user_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(lfs_user[i-1], lfs_user[i]),
                )
            self.user_encoder.add_module("act{}".format(i), nn.Tanh())
        print(self.user_encoder)
        # Get user side variational distribution params - mu and logsigma
        self.user_var = nn.Linear(lfs_user[-1], 2 * lf_pos)  # [mu, logsigma]

        # Define item-side network
        self.item_encoder = nn.Sequential()
        for i in range(0, len(lfs_item)):
            # for first layer, inp dim is #users
            if i == 0:
                self.item_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(num_users, lfs_item[i]),
                )
            else:
                self.item_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(lfs_item[i-1], lfs_item[i]),
                )
            self.item_encoder.add_module("act{}".format(i), nn.Tanh())

        # Get item side variational distribution params - mu and logsigma
        self.item_var = nn.Linear(lfs_item[i], 2 * lf_pos)  # [mu, logsigma]
        # Collect all user and item side params
        self.user_params = it.chain(self.user_encoder.parameters(), \
                            self.user_var.parameters())
        
        self.item_params = it.chain(self.item_encoder.parameters(), \
                            self.item_var.parameters())

        self.lr = lr
        self.dropout = nn.Dropout(0.5)
        self.topk = topk
        self.clip = torch.tensor(clip)
        self.clip.requires_grad = False
        #self.loss = nn.BCEWithLogitsLoss()
        #self.reg = reg
        # MF with or w/o IPS
        self.ips = ips

    def configure_optimizers(self):
        u_opt = torch.optim.Adam(self.user_params, lr=self.lr)
        i_opt = torch.optim.Adam(self.item_params, lr=self.lr)
        return u_opt, i_opt

    def _l2_regularize(self, array):
        return torch.sum(array ** 2.0)

    def _forward_user(self, u_vec):
        # # Forward pass for users in the batch
        #user_click_norm = F.normalize(u_vec)
        #user_click_norm = self.dropout(user_click_norm)
        
        user_click_norm = u_vec
        user_embedding = self.user_encoder(user_click_norm.float())
        user_var = self.user_var(user_embedding)

        mu_u, logvar_u = torch.chunk(user_var, chunks=2, dim=1)
        std_u = torch.exp(0.5 * logvar_u)
        #std_u = F.sigmoid(logvar_u)
        # Defining the posterior dist
        qz_u = td.normal.Normal(mu_u, std_u)
        # Drawing one sample from the posterior
        theta = qz_u.rsample()
        # Generate Prediction over all items for users in the batch - [bs, num_items]
        #theta = theta.to(self.device)
        #self.beta = self.beta.to(self.device)
        pred = (theta @ self.beta.T)
        return theta, qz_u, pred, mu_u, std_u, logvar_u

    def _forward_item(self, i_vec):
        # Forward pass for items in the batch
        #item_click_norm = F.normalize(i_vec)
        #item_click_norm = self.dropout(item_click_norm)
        
        item_click_norm = i_vec
        item_embedding = self.item_encoder(item_click_norm.float())
        item_var = self.item_var(item_embedding)

        mu_i, logvar_i = torch.chunk(item_var, chunks=2, dim=1)
        std_i = torch.exp(0.5 * logvar_i)
        #std_i = F.sigmoid(logvar_i)
        # Defining the posterior dist
        qz_i = td.normal.Normal(mu_i, std_i)
        # Drawing one sample from the posterior
        beta = qz_i.rsample()
        # Generate Prediction over all users for users in the batch - [bs, num_users]
        #beta = beta.to(self.device)
        #self.theta = self.theta.to(self.device)
        pred = (beta @ self.theta.T)
        return beta, qz_i, pred, mu_i, std_i, logvar_i

    def training_step(self, batch, batch_idx):
        EPS = 1e-7
        u_opt, i_opt = self.optimizers()

        # Optimizing for users
        uid, pid, click, prop, user_click_vec, item_click_vec, user_prop, item_prop = batch
        prop = torch.maximum(prop, self.clip)
        user_prop = torch.maximum(user_prop, self.clip)
        item_prop = torch.maximum(item_prop, self.clip)
        user_click_vec = user_click_vec.squeeze(1)
        item_click_vec = item_click_vec.squeeze(2)
        #user_prop = user_prop.squeeze(1)
        #item_prop = item_prop.squeeze(2)
        theta, qz_u, pred, mu_u, std_u, logvar_u = self._forward_user(user_click_vec)
        
        logits = torch.sigmoid(pred)
        if not self.ips:
            ce_loss = user_click_vec * torch.log(logits + EPS) + (1 - user_click_vec) * torch.log(1 - logits + EPS)
            ce_loss = torch.sum(ce_loss, axis=-1)
        else:
            #norm_factor = torch.sum(user_prop, dim=-1)
            ce_loss = (user_click_vec/user_prop) * torch.log(logits + EPS) +\
                 (1 - user_click_vec/user_prop) * torch.log(1 - logits + EPS)
            #norm_factor = torch.sum(1/user_prop, axis=-1)
            ce_loss = torch.clip(ce_loss, -100, 100)
            ce_loss = torch.sum(ce_loss, axis=-1)
            
        ce_loss = -torch.mean(ce_loss)
        pz_u = td.normal.Normal(torch.zeros_like(qz_u.loc), torch.ones_like(qz_u.scale))
        KL = torch.mean(torch.sum(0.5 * (-logvar_u + torch.exp(logvar_u) + mu_u ** 2 - 1), dim=1))
        KL_u = KL.detach().clone()
        #KLD_u = td.kl_divergence(qz_u, pz_u).sum()
        loss_u = ce_loss +  KL
        u_opt.zero_grad()
        self.manual_backward(loss_u)
        u_opt.step()
        theta, qz_u, pred, mu_u, std_u, _  = self._forward_user(user_click_vec)
        self.theta.data[uid] = theta.data
        self.mu_u.data[uid] = mu_u.data

        # Optimizing for items
        beta, qz_i, pred, mu_i, std_i, logvar_i = self._forward_item(item_click_vec)
        logits = torch.sigmoid(pred)
        if not self.ips:
            ce_loss = item_click_vec * torch.log(logits + EPS) + (1 - item_click_vec) * torch.log(1 - logits + EPS)
            ce_loss = torch.sum(ce_loss, axis=-1)
        else:
            #norm_factor = torch.sum(item_prop, dim=-1)
            ce_loss = (item_click_vec/item_prop) * torch.log(logits + EPS) +\
                     (1 - item_click_vec/item_prop) * torch.log(1 - logits + EPS)
            #norm_factor = torch.sum(1/item_prop, axis=-1)
            ce_loss = torch.clip(ce_loss, -100, 100)
            ce_loss = torch.sum(ce_loss, axis=-1)
        ce_loss = -torch.mean(ce_loss)
        pz_i = td.normal.Normal(torch.zeros_like(qz_i.loc), torch.ones_like(qz_i.scale))
        KL = torch.mean(torch.sum(0.5 * (-logvar_i + torch.exp(logvar_i) + mu_i ** 2 - 1), dim=1))
        #KLD_i = td.kl_divergence(qz_i, pz_i).sum()
        KL_i = KL.detach().clone()
        loss_i = ce_loss + KL
        i_opt.zero_grad()
        self.manual_backward(loss_i)
        i_opt.step()
        beta, qz_i, pred, mu_i, std_i, _ = self._forward_item(item_click_vec)
        self.beta.data[pid] = beta.data
        self.mu_i.data[pid] = mu_i.data
        # Logging
        self.log_dict({'user_loss': loss_u, 'item_loss': loss_i, 'kl_user': KL_u, 'kl_item': KL_i},\
                         on_step=False, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        uid, pid, click, prop, user_click_vec, item_click_vec, _, _ = batch
        user_click_vec = user_click_vec.squeeze(1)
        item_click_vec = item_click_vec.squeeze(2)
        mu_u = self.mu_u[uid]
        #mu_u = mu_u.to(self.device)
        #self.mu_i = self.mu_i.to(self.device)
        # get predictions for users in the batch and all items using mean of the variational distrs
        user_all_scores = (mu_u @ self.mu_i.T) #+  self.bias
        log_metric = {}
        # Get metrics at cut-offs - {1, 3, 5}
        for k in [1,3,5]:
            user_item_ranking = torch.argsort(user_all_scores, dim=1, descending=True)[:, :k]
            # Create dictionary to store (uid, dcgs) pairs
            dcg_dict = {}
            val_rank = (user_item_ranking == pid.unsqueeze(1)).int()
            val_rank = val_rank.cpu().detach().numpy()
            #map_val = rank_metrics.mean_average_precision(val_rank)
            # get validation metrics at cut-off - {1, 3, 5}
            map_val = list(map(lambda x: rank_metrics.average_precision(x), val_rank))
            map_dict = list(zip(uid.cpu().detach().numpy(), map_val, prop.cpu().detach().numpy()))
            ndcg_val = list(map(lambda x: rank_metrics.dcg_at_k(x, k), val_rank))
            dcg_dict = list(zip(uid.cpu().detach().numpy(), ndcg_val, prop.cpu().detach().numpy() ))
            prop_dict = list(zip(uid.cpu().detach().numpy(), prop.cpu().detach().numpy()))
            log_metric['map_val_%s'%str(k)] = map_dict
            log_metric['ndcg_val_%s'%str(k)] = dcg_dict
            log_metric['prop_batch'] = prop_dict
        #log_metric = {'map_val': map_dict, 'ndcg_val': dcg_dict, 'prop_batch': prop_dict}
        #self.log("ndcg_val", ndcg_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log_metric

    def _reduce_metric(self, outputs):
        '''
        For input of form [(uid, uid_metric_Val)], reduce
        '''
        metric_vals = {}
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                metric_final = []
                user_metric = [map_list[metric + '_%s'%str(k)] for map_list in outputs]
                user_metric_flat = list(itertools.chain(*user_metric))
                for key, group in itertools.groupby(sorted(user_metric_flat), operator.itemgetter(0)):
                    metric_per_user = 0.0
                    prop_per_user = 0.0
                    for item in group:
                        metric_per_user += item[1]
                    #try:
                    metric_final.append(metric_per_user)
                    #except:
                    #    ndcg_final.append(0.)
                metric_vals[metric + '_%s'%str(k)] = np.mean(metric_final)
        return metric_vals

    def _reduce_metric_val(self, outputs):
        '''
        For input of form [(uid, uid_metric_Val)], reduce
        '''
        metric_vals = {}
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                metric_final = []
                user_metric = [map_list[metric + '_%s'%str(k)] for map_list in outputs]
                user_metric_flat = list(itertools.chain(*user_metric))
                for key, group in itertools.groupby(sorted(user_metric_flat), operator.itemgetter(0)):
                    metric_per_user = 0.0
                    prop_per_user = 0.0
                    for item in group:
                        metric_per_user += item[1]/item[2]
                        prop_per_user += 1/item[2]
                    #try:
                    metric_final.append(metric_per_user/prop_per_user)
                    #except:
                    #    ndcg_final.append(0.)
                metric_vals[metric + '_%s'%str(k)] = np.mean(metric_final)
        return metric_vals

    def validation_epoch_end(self, outputs):
        metric_vals = self._reduce_metric_val(outputs)
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                self.log("ptl/" + metric + '_%s'%str(k), metric_vals[metric + '_%s'%str(k)])

    def test_step(self, batch, batch_idx):
        uid, pid, click, prop, user_click_vec, item_click_vec, _, _ = batch
        user_click_vec = user_click_vec.squeeze(1)
        item_click_vec = item_click_vec.squeeze(2)
        mu_u = self.mu_u[uid]
        #mu_u = mu_u.to(self.device)
        #self.mu_i = self.mu_i.to(self.device)
        # get predictions for users in the batch and all items using mean of the variational distrs
        user_all_scores = (mu_u @ self.mu_i.T) #+  self.bias
        log_metric = {}
        for k in [1,3,5]:
            user_item_ranking = torch.argsort(user_all_scores, dim=-1, descending=True)[:, :k]
            test_rank = (user_item_ranking == pid.unsqueeze(1)).int()
            test_rank = test_rank.cpu().detach().numpy()
            map_test = list(map(lambda x: rank_metrics.average_precision(x), test_rank))
            map_dict = list(zip(uid.cpu().detach().numpy(), map_test))
            ndcg_test = list(map(lambda x: rank_metrics.dcg_at_k(x, 5), test_rank))
            dcg_dict = list(zip(uid.cpu().detach().numpy(), ndcg_test))
            #logits = self(uid, pid)
            #val_loss = F.cross_entropy(y_hat, y)
            #log_metric = {'map_val': map_dict, 'ndcg_val': dcg_dict, 'batch_lens': len(test_rank)}
            log_metric['map_val_%s'%str(k)] = map_dict
            log_metric['ndcg_val_%s'%str(k)] = dcg_dict
            log_metric['batch_lens'] = len(test_rank)
        #self.log("ndcg_val", ndcg_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log_metric

    def test_epoch_end(self, outputs):
        metric_vals = self._reduce_metric(outputs)
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                self.log("test/" + metric + '_%s'%str(k), metric_vals[metric + '_%s'%str(k)])


class PowerVAEClick(LightningModule):
    '''
    Variational AE with Power Spherical distribution based MF with user and item covariates as input
    '''
    def __init__(self, num_users, num_items, \
                lfs_user, lfs_item, lf_pos,
                lr=3 * 1e-3, clip=0.1, topk=5, scale=400., ips=False, gpu=True):
        '''
        args:
            lfs_user: User-side n/w dims 
            lfs_item: Item-side n/w dims
            lf_pos: Latent dimensions of the posteriors on user and item side
        '''
        super().__init__()
        self.automatic_optimization = False
        #self.device = gpu
        # torch.autograd.detect_anomaly
        # torch.autograd.set_detect_anomaly(True)

        # Initialize posterior params for users and items randomly
        self.theta = torch.randn(num_users, lf_pos) 
        self.beta = torch.randn(num_items, lf_pos) 
        nn.init.xavier_normal_(self.theta)
        nn.init.xavier_normal_(self.beta)
        if gpu:
            self.theta = self.theta.to('cuda')
            self.beta = self.beta.to('cuda')
        # store mu's for user and items
        self.mu_u = torch.zeros(num_users, lf_pos) 
        self.mu_i = torch.zeros(num_items, lf_pos) 
        if gpu:
            self.mu_u = self.mu_u.to('cuda')
            self.mu_i = self.mu_i.to('cuda')
        # Define user-side network
        self.user_encoder = nn.Sequential()
        for i in range(0, len(lfs_user)):
            # for first layer, inp dim is #items
            if i == 0:
                self.user_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(num_items, lfs_user[i]),
                )
            else:
                self.user_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(lfs_user[i-1], lfs_user[i]),
                )
            self.user_encoder.add_module("act{}".format(i), nn.Tanh())
        print(self.user_encoder)
        # Get user side variational distribution params - mu and logsigma
        self.user_var = nn.Linear(lfs_user[-1], lf_pos)  # [mu, logsigma]

        # Define item-side network
        self.item_encoder = nn.Sequential()
        for i in range(0, len(lfs_item)):
            # for first layer, inp dim is #users
            if i == 0:
                self.item_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(num_users, lfs_item[i]),
                )
            else:
                self.item_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(lfs_item[i-1], lfs_item[i]),
                )
            self.item_encoder.add_module("act{}".format(i), nn.Tanh())

        # Get item side variational distribution params - mu and logsigma
        self.item_var = nn.Linear(lfs_item[i], lf_pos)  # [mu, logsigma]
        # Collect all user and item side params
        self.user_params = it.chain(self.user_encoder.parameters(), \
                            self.user_var.parameters())
        
        self.item_params = it.chain(self.item_encoder.parameters(), \
                            self.item_var.parameters())

        self.lr = lr
        self.dropout = nn.Dropout(0.1)
        self.topk = topk
        self.clip = torch.tensor(clip)
        self.clip.requires_grad = False
        self.scale = torch.tensor(scale)#.to(self.device)
        self.scale.requires_grad = False
        #self.loss = nn.BCEWithLogitsLoss()
        #self.reg = reg
        # MF with or w/o IPS
        self.ips = ips

    def configure_optimizers(self):
        u_opt = torch.optim.Adam(self.user_params, lr=self.lr)
        i_opt = torch.optim.Adam(self.item_params, lr=self.lr)
        return u_opt, i_opt

    def _l2_regularize(self, array):
        return torch.sum(array ** 2.0)

    def _forward_user(self, u_vec):
        # # Forward pass for users in the batch
        # user_click_norm = F.normalize(u_vec)
        # user_click_norm = self.dropout(user_click_norm)
        
        user_click_norm = u_vec
        user_embedding = self.user_encoder(user_click_norm.float())
        user_var = self.user_var(user_embedding)

        mu_u = user_var
        #mu_u = F.normalize(mu_u, p=2, dim=1)
        #std_u = F.sigmoid(logvar_u)
        # Defining the posterior dist
        self.scale = self.scale.to(self.device)
        qz_u = PowerSpherical(mu_u, self.scale)#.to(self.device)
        # Drawing one sample from the posterior
        theta = qz_u.rsample()
        # Generate Prediction over all items for users in the batch - [bs, num_items]
        #theta = theta.to(self.device)
        #self.beta = self.beta.to(self.device)
        pred = (theta @ self.beta.T)
        return theta, qz_u, pred, mu_u

    def _forward_item(self, i_vec):
        # Forward pass for items in the batch
        # item_click_norm = F.normalize(i_vec)
        # item_click_norm = self.dropout(item_click_norm)
        
        item_click_norm = i_vec
        item_embedding = self.item_encoder(item_click_norm.float())
        item_var = self.item_var(item_embedding)

        mu_i = item_var
        #mu_i = F.normalize(mu_i, p=2, dim=1)
        #std_i = F.sigmoid(logvar_i)
        # Defining the posterior dist
        self.scale = self.scale.to(self.device)
        qz_i = PowerSpherical(mu_i, self.scale)
        # Drawing one sample from the posterior
        beta = qz_i.rsample()
        # Generate Prediction over all users for users in the batch - [bs, num_users]
        #beta = beta.to(self.device)
        #self.theta = self.theta.to(self.device)
        pred = (beta @ self.theta.T)
        return beta, qz_i, pred, mu_i

    def training_step(self, batch, batch_idx):
        EPS = 1e-7
        u_opt, i_opt = self.optimizers()

        # Optimizing for users
        uid, pid, click, prop, user_click_vec, item_click_vec, user_prop, item_prop = batch
        prop = torch.maximum(prop, self.clip)
        user_prop = torch.maximum(user_prop, self.clip)
        item_prop = torch.maximum(item_prop, self.clip)
        user_click_vec = user_click_vec.squeeze(1)
        item_click_vec = item_click_vec.squeeze(2)
        #user_prop = user_prop.squeeze(1)
        #item_prop = item_prop.squeeze(2)
        theta, qz_u, pred, mu_u = self._forward_user(user_click_vec)
        
        logits = torch.sigmoid(pred)
        if not self.ips:
            ce_loss = user_click_vec * torch.log(logits + EPS) + (1 - user_click_vec) * torch.log(1 - logits + EPS)
            ce_loss = torch.sum(ce_loss, axis=-1)
        else:
            #norm_factor = torch.sum(user_prop, dim=-1)
            ce_loss = (user_click_vec/user_prop) * torch.log(logits + EPS) +\
                 (1 - user_click_vec/user_prop) * torch.log(1 - logits + EPS)
            #norm_factor = torch.sum(1/user_prop, axis=-1)
            ce_loss = torch.clip(ce_loss, -150, 150)
            ce_loss = torch.sum(ce_loss, axis=-1)
            
        ce_loss = -torch.mean(ce_loss)
        #pz_u = td.normal.Normal(torch.zeros_like(qz_u.loc), torch.ones_like(qz_u.scale))
        #KL = torch.mean(torch.sum(0.5 * (-logvar_u + torch.exp(logvar_u) + mu_u ** 2 - 1), dim=1))
        #KL_u = KL.detach().clone()
        #KLD_u = td.kl_divergence(qz_u, pz_u).sum()
        loss_u = ce_loss #+  KL
        u_opt.zero_grad()
        self.manual_backward(loss_u)
        u_opt.step()
        theta, qz_u, pred, mu_u  = self._forward_user(user_click_vec)
        self.theta.data[uid] = theta.data
        self.mu_u.data[uid] = mu_u.data

        # Optimizing for items
        beta, qz_i, pred, mu_i = self._forward_item(item_click_vec)
        logits = torch.sigmoid(pred)
        if not self.ips:
            ce_loss = item_click_vec * torch.log(logits + EPS) + (1 - item_click_vec) * torch.log(1 - logits + EPS)
            ce_loss = torch.sum(ce_loss, axis=-1)
        else:
            #norm_factor = torch.sum(item_prop, dim=-1)
            ce_loss = (item_click_vec/item_prop) * torch.log(logits + EPS) +\
                     (1 - item_click_vec/item_prop) * torch.log(1 - logits + EPS)
            #norm_factor = torch.sum(1/item_prop, axis=-1)
            ce_loss = torch.clip(ce_loss, -150, 150)
            ce_loss = torch.sum(ce_loss, axis=-1)
        ce_loss = -torch.mean(ce_loss)
        #pz_i = td.normal.Normal(torch.zeros_like(qz_i.loc), torch.ones_like(qz_i.scale))
        #KL = torch.mean(torch.sum(0.5 * (-logvar_i + torch.exp(logvar_i) + mu_i ** 2 - 1), dim=1))
        #KLD_i = td.kl_divergence(qz_i, pz_i).sum()
        #KL_i = KL.detach().clone()
        loss_i = ce_loss #+ KL
        i_opt.zero_grad()
        self.manual_backward(loss_i)
        i_opt.step()
        beta, qz_i, pred, mu_i = self._forward_item(item_click_vec)
        self.beta.data[pid] = beta.data
        self.mu_i.data[pid] = mu_i.data
        # Logging
        self.log_dict({'user_loss': loss_u, 'item_loss': loss_i},\
                         on_step=False, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        uid, pid, click, prop, user_click_vec, item_click_vec, _, _ = batch
        user_click_vec = user_click_vec.squeeze(1)
        item_click_vec = item_click_vec.squeeze(2)
        mu_u = self.mu_u[uid]
        #mu_u = mu_u.to(self.device)
        #self.mu_i = self.mu_i.to(self.device)
        # get predictions for users in the batch and all items using mean of the variational distrs
        user_all_scores = (mu_u @ self.mu_i.T) #+  self.bias
        log_metric = {}
        # Get metrics at cut-offs - {1, 3, 5}
        for k in [1,3,5]:
            user_item_ranking = torch.argsort(user_all_scores, dim=1, descending=True)[:, :k]
            # Create dictionary to store (uid, dcgs) pairs
            dcg_dict = {}
            val_rank = (user_item_ranking == pid.unsqueeze(1)).int()
            val_rank = val_rank.cpu().detach().numpy()
            #map_val = rank_metrics.mean_average_precision(val_rank)
            # get validation metrics at cut-off - {1, 3, 5}
            map_val = list(map(lambda x: rank_metrics.average_precision(x), val_rank))
            map_dict = list(zip(uid.cpu().detach().numpy(), map_val, prop.cpu().detach().numpy()))
            ndcg_val = list(map(lambda x: rank_metrics.dcg_at_k(x, k), val_rank))
            dcg_dict = list(zip(uid.cpu().detach().numpy(), ndcg_val, prop.cpu().detach().numpy() ))
            prop_dict = list(zip(uid.cpu().detach().numpy(), prop.cpu().detach().numpy()))
            log_metric['map_val_%s'%str(k)] = map_dict
            log_metric['ndcg_val_%s'%str(k)] = dcg_dict
            log_metric['prop_batch'] = prop_dict
        #log_metric = {'map_val': map_dict, 'ndcg_val': dcg_dict, 'prop_batch': prop_dict}
        #self.log("ndcg_val", ndcg_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log_metric

    def _reduce_metric(self, outputs):
        '''
        For input of form [(uid, uid_metric_Val)], reduce
        '''
        metric_vals = {}
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                metric_final = []
                user_metric = [map_list[metric + '_%s'%str(k)] for map_list in outputs]
                user_metric_flat = list(itertools.chain(*user_metric))
                for key, group in itertools.groupby(sorted(user_metric_flat), operator.itemgetter(0)):
                    metric_per_user = 0.0
                    prop_per_user = 0.0
                    for item in group:
                        metric_per_user += item[1]
                    #try:
                    metric_final.append(metric_per_user)
                    #except:
                    #    ndcg_final.append(0.)
                metric_vals[metric + '_%s'%str(k)] = np.mean(metric_final)
        return metric_vals

    def _reduce_metric_val(self, outputs):
        '''
        For input of form [(uid, uid_metric_Val)], reduce
        '''
        metric_vals = {}
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                metric_final = []
                user_metric = [map_list[metric + '_%s'%str(k)] for map_list in outputs]
                user_metric_flat = list(itertools.chain(*user_metric))
                for key, group in itertools.groupby(sorted(user_metric_flat), operator.itemgetter(0)):
                    metric_per_user = 0.0
                    prop_per_user = 0.0
                    for item in group:
                        metric_per_user += item[1]/item[2]
                        prop_per_user += 1/item[2]
                    #try:
                    metric_final.append(metric_per_user/prop_per_user)
                    #except:
                    #    ndcg_final.append(0.)
                metric_vals[metric + '_%s'%str(k)] = np.mean(metric_final)
        return metric_vals

    def validation_epoch_end(self, outputs):
        metric_vals = self._reduce_metric_val(outputs)
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                self.log("ptl/" + metric + '_%s'%str(k), metric_vals[metric + '_%s'%str(k)])

    def test_step(self, batch, batch_idx):
        uid, pid, click, prop, user_click_vec, item_click_vec, _, _ = batch
        user_click_vec = user_click_vec.squeeze(1)
        item_click_vec = item_click_vec.squeeze(2)
        mu_u = self.mu_u[uid]
        #mu_u = mu_u.to(self.device)
        #self.mu_i = self.mu_i.to(self.device)
        # get predictions for users in the batch and all items using mean of the variational distrs
        user_all_scores = (mu_u @ self.mu_i.T) #+  self.bias
        log_metric = {}
        for k in [1,3,5]:
            user_item_ranking = torch.argsort(user_all_scores, dim=-1, descending=True)[:, :k]
            test_rank = (user_item_ranking == pid.unsqueeze(1)).int()
            test_rank = test_rank.cpu().detach().numpy()
            map_test = list(map(lambda x: rank_metrics.average_precision(x), test_rank))
            map_dict = list(zip(uid.cpu().detach().numpy(), map_test))
            ndcg_test = list(map(lambda x: rank_metrics.dcg_at_k(x, 5), test_rank))
            dcg_dict = list(zip(uid.cpu().detach().numpy(), ndcg_test))
            #logits = self(uid, pid)
            #val_loss = F.cross_entropy(y_hat, y)
            #log_metric = {'map_val': map_dict, 'ndcg_val': dcg_dict, 'batch_lens': len(test_rank)}
            log_metric['map_val_%s'%str(k)] = map_dict
            log_metric['ndcg_val_%s'%str(k)] = dcg_dict
            log_metric['batch_lens'] = len(test_rank)
        #self.log("ndcg_val", ndcg_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log_metric

    def test_epoch_end(self, outputs):
        metric_vals = self._reduce_metric(outputs)
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                self.log("test/" + metric + '_%s'%str(k), metric_vals[metric + '_%s'%str(k)])

# Define Importance Weighted AE with click vectors
class IWAEClick(LightningModule):
    '''
    Importance Weighted AE with user and item click vectors as input
    '''
    def __init__(self, num_users, num_items, \
                lfs_user, lfs_item, lf_pos,
                lr=3 * 1e-3, topk=5, reg=1e-5, k=16, ips=False, gpu=True):
        '''
        args:
            lfs_user: User-side n/w dims 
            lfs_item: Item-side n/w dims
            lf_pos: Latent dimensions of the posteriors on user and item side
        '''
        super().__init__()
        self.automatic_optimization = False
        # torch.autograd.detect_anomaly
        # torch.autograd.set_detect_anomaly(True)

        # Initialize posterior params for users and items randomly
        self.theta = torch.randn(num_users, lf_pos) 
        self.beta = torch.randn(num_items, lf_pos) 
        nn.init.xavier_normal_(self.theta)
        nn.init.xavier_normal_(self.beta)
        if gpu:
            self.theta = self.theta.to('cuda')
            self.beta = self.beta.to('cuda')
        # store mu's for user and items
        self.mu_u = torch.zeros(num_users, lf_pos) 
        self.mu_i = torch.zeros(num_items, lf_pos) 
        if gpu:
            self.mu_u = self.mu_u.to('cuda')
            self.mu_i = self.mu_i.to('cuda')
        # Define user-side network
        self.user_encoder = nn.Sequential()
        for i in range(0, len(lfs_user)):
            # for first layer, inp dim is #items
            if i == 0:
                self.user_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(num_items, lfs_user[i]),
                )
            else:
                self.user_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(lfs_user[i-1], lfs_user[i]),
                )
            self.user_encoder.add_module("act{}".format(i), nn.Tanh())
        print(self.user_encoder)
        # Get user side variational distribution params - mu and logsigma
        self.user_var = nn.Linear(lfs_user[-1], 2 * lf_pos)  # [mu, logsigma]

        # Define item-side network
        self.item_encoder = nn.Sequential()
        for i in range(0, len(lfs_item)):
            # for first layer, inp dim is #users
            if i == 0:
                self.item_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(num_users, lfs_item[i]),
                )
            else:
                self.item_encoder.add_module(
                    "fc{}".format(i),
                    nn.Linear(lfs_item[i-1], lfs_item[i]),
                )
            self.item_encoder.add_module("act{}".format(i), nn.Tanh())

        # Get item side variational distribution params - mu and logsigma
        self.item_var = nn.Linear(lfs_item[i], 2 * lf_pos)  # [mu, logsigma]
        # Collect all user and item side params
        self.user_params = it.chain(self.user_encoder.parameters(), \
                            self.user_var.parameters())
        
        self.item_params = it.chain(self.item_encoder.parameters(), \
                            self.item_var.parameters())

        self.lr = lr
        self.dropout = nn.Dropout(0.5)
        self.topk = topk
        #self.loss = nn.BCEWithLogitsLoss()
        self.reg = reg
        # MF with or w/o IPS
        self.ips = ips
        # num of MC samples from the posterior
        self.k = k

    def configure_optimizers(self):
        u_opt = torch.optim.Adam(self.user_params, lr=self.lr)
        i_opt = torch.optim.Adam(self.item_params, lr=self.lr)
        return u_opt, i_opt

    def _l2_regularize(self, array):
        return torch.sum(array ** 2.0)

    def _forward_user(self, u_vec, mode='train'):
        # # Forward pass for users in the batch
        #user_click_norm = F.normalize(u_vec)
        #user_click_norm = self.dropout(user_click_norm)
        
        user_click_norm = u_vec
        user_embedding = self.user_encoder(user_click_norm.float())
        user_var = self.user_var(user_embedding)

        mu_u, logvar_u = torch.chunk(user_var, chunks=2, dim=1)
        std_u = torch.exp(0.5 * logvar_u)
        #std_u = F.sigmoid(logvar_u)
        # Defining the posterior dist
        qz_u = td.normal.Normal(mu_u, std_u)
        pz_u = td.normal.Normal(torch.zeros_like(qz_u.loc), torch.ones_like(qz_u.scale))
        # get k samples from prior 
        
        if mode == 'train':
            # Drawing multiple sample from the posterior
            thetas = [qz_u.rsample() for i in range(self.k)]
            preds = [torch.sigmoid(theta @ self.beta.T) for theta in thetas]
            #thetas = torch.stack(thetas)
            # Generate Prediction over all items for users in the batch - [bs, num_items]
            preds = torch.stack(preds)
        else:
            thetas = qz_u.rsample()
            preds = torch.sigmoid(thetas @ self.beta.T)
        
        #theta = theta.to(self.device)
        #self.beta = self.beta.to(self.device)
        return thetas, qz_u, preds, mu_u, std_u, logvar_u

    def _forward_item(self, i_vec, mode='train'):
        # Forward pass for items in the batch
        #item_click_norm = F.normalize(i_vec)
        #item_click_norm = self.dropout(item_click_norm)
        
        item_click_norm = i_vec
        item_embedding = self.item_encoder(item_click_norm.float())
        item_var = self.item_var(item_embedding)

        mu_i, logvar_i = torch.chunk(item_var, chunks=2, dim=1)
        std_i = torch.exp(0.5 * logvar_i)
        #std_i = F.sigmoid(logvar_i)
        # Defining the posterior dist
        qz_i = td.normal.Normal(mu_i, std_i)
        if mode == 'train':
            # Drawing multiple sample from the posterior
            betas = [qz_i.rsample() for i in range(self.k)]
            preds = [torch.sigmoid(beta @ self.theta.T) for beta in betas]
            #betas = torch.stack(betas)
            # Generate Prediction over all users for users in the batch - [bs, num_users]
            #beta = beta.to(self.device)
            #self.theta = self.theta.to(self.device)
            preds = torch.stack(preds)
        else:
            betas = qz_i.rsample()
            preds = torch.sigmoid(betas @ self.theta.T)
        
        return betas, qz_i, preds, mu_i, std_i, logvar_i

    
    def training_step(self, batch, batch_idx):
        # Define the forward pass for importnce weighted AE
        EPS = 1e-7
        u_opt, i_opt = self.optimizers()

        # Optimizing for users
        uid, pid, click, prop, user_click_vec, item_click_vec, user_prop, item_prop = batch
        user_click_vec = user_click_vec.squeeze(1)
        item_click_vec = item_click_vec.squeeze(2)
        # make a copy of user vec for inference
        user_vec_inf = user_click_vec.detach().clone()
        item_vec_inf = item_click_vec.detach().clone()
        #user_prop = user_prop.squeeze(1)
        #item_prop = item_prop.squeeze(2)
        thetas, qz_u, pred, mu_u, std_u, logvar_u = self._forward_user(user_click_vec)
        pz_u = td.normal.Normal(torch.zeros_like(qz_u.loc), torch.ones_like(qz_u.scale))
        # get k samples from prior 
        log_pzu = [pz_u.log_prob(theta) for theta in thetas]
        log_pzu = torch.stack(log_pzu)
        user_click_vec = user_click_vec.unsqueeze(0).repeat(self.k, 1, 1)
        user_prop = user_prop.unsqueeze(0).repeat(self.k, 1, 1)
        # prior log-pdf
        log_pzu = torch.sum(log_pzu, dim=-1)
        # prior log-pdf
        log_pzx = [qz_u.log_prob(theta) for theta in thetas]
        log_pzx = torch.stack(log_pzx)
        log_pzx = torch.sum(log_pzx, dim=-1)
        if not self.ips:
            log_pcui = user_click_vec * torch.log(pred + EPS) + (1 - user_click_vec) * torch.log(1 - pred + EPS)
            log_pcui = torch.sum(log_pcui, axis=-1)
            log_lik = log_pcui + log_pzu - log_pzx
            log_lik = torch.logsumexp(log_lik, dim=0)  - np.log(self.k)

        else:
            #norm_factor = torch.sum(user_prop, dim=-1)
            log_pcui = (user_click_vec/user_prop) * torch.log(pred + EPS) + (1 - (user_click_vec/user_prop)) * torch.log(1 - pred + EPS)
            log_pcui = torch.sum(log_pcui, axis=-1)
            log_lik = log_pcui + log_pzu - log_pzx
            log_lik = torch.logsumexp(log_lik, dim=0)  - np.log(self.k)
            
        loss_u = -torch.mean(log_lik)
        u_opt.zero_grad()
        self.manual_backward(loss_u)
        u_opt.step()
        theta, qz_u, pred, mu_u, std_u, _  = self._forward_user(user_vec_inf, mode='inference')
        self.theta.data[uid] = theta.data
        self.mu_u.data[uid] = mu_u.data

        # Optimizing for items
        betas, qz_i, pred, mu_i, std_i, logvar_i = self._forward_item(item_click_vec)
        pz_i = td.normal.Normal(torch.zeros_like(qz_i.loc), torch.ones_like(qz_i.scale))
        # get k samples from prior 
        log_pzi = [pz_i.log_prob(beta) for beta in betas]
        log_pzi = torch.stack(log_pzi)
        item_click_vec = item_click_vec.unsqueeze(0).repeat(self.k, 1, 1)
        item_prop = item_prop.unsqueeze(0).repeat(self.k, 1, 1)
        # prior log-pdf
        log_pzi = torch.sum(log_pzi, dim=-1)
        # prior log-pdf
        log_pzx = [qz_i.log_prob(beta) for beta in betas]
        log_pzx = torch.stack(log_pzx)
        log_pzx = torch.sum(log_pzx, dim=-1)
        if not self.ips:
            log_pcui = item_click_vec * torch.log(pred + EPS) + (1 - item_click_vec) * torch.log(1 - pred + EPS)
            log_pcui = torch.sum(log_pcui, axis=-1)
            log_lik = log_pcui + log_pzi - log_pzx
            log_lik = torch.logsumexp(log_lik, dim=0)  - np.log(self.k)

        else:
            #norm_factor = torch.sum(user_prop, dim=-1)
            log_pcui = (item_click_vec/item_prop) * torch.log(pred + EPS) + (1 - (item_click_vec/item_prop)) * torch.log(1 - pred + EPS)
            log_pcui = torch.sum(log_pcui, axis=-1)
            log_lik = log_pcui + log_pzi - log_pzx
            log_lik = torch.logsumexp(log_lik, dim=0)  - np.log(self.k)
        #KLD_i = td.kl_divergence(qz_i, pz_i).sum()
        loss_i = -torch.mean(log_lik)
        i_opt.zero_grad()
        self.manual_backward(loss_i)
        i_opt.step()
        beta, qz_i, pred, mu_i, std_i, _ = self._forward_item(item_vec_inf, mode='inference')
        self.beta.data[pid] = beta.data
        self.mu_i.data[pid] = mu_i.data
        # Logging
        self.log_dict({'user_loss': loss_u, 'item_loss': loss_i}, on_step=False, on_epoch=True,logger=True)

    def validation_step(self, batch, batch_idx):
        uid, pid, click, prop, user_click_vec, item_click_vec, _, _ = batch
        user_click_vec = user_click_vec.squeeze(1)
        item_click_vec = item_click_vec.squeeze(2)
        mu_u = self.mu_u[uid]
        #mu_u = mu_u.to(self.device)
        #self.mu_i = self.mu_i.to(self.device)
        # get predictions for users in the batch and all items using mean of the variational distrs
        user_all_scores = (mu_u @ self.mu_i.T) #+  self.bias
        user_item_ranking = torch.argsort(user_all_scores, dim=1, descending=True)[:, :self.topk]
        # Create dictionary to store (uid, dcgs) pairs
        dcg_dict = {}
        val_rank = (user_item_ranking == pid.unsqueeze(1)).int()
        val_rank = val_rank.cpu().detach().numpy()
        #map_val = rank_metrics.mean_average_precision(val_rank)
        map_val = list(map(lambda x: rank_metrics.average_precision(x), val_rank))
        map_dict = list(zip(uid.cpu().detach().numpy(), map_val, prop.cpu().detach().numpy()))
        ndcg_val = list(map(lambda x: rank_metrics.dcg_at_k(x, 5), val_rank))
        dcg_dict = list(zip(uid.cpu().detach().numpy(), ndcg_val, prop.cpu().detach().numpy() ))
        prop_dict = list(zip(uid.cpu().detach().numpy(), prop.cpu().detach().numpy()))
        #logits = self(uid, pid)
        #val_loss = F.cross_entropy(y_hat, y)
        log_metric = {'map_val': map_dict, 'ndcg_val': dcg_dict, 'prop_batch': prop_dict}
        #self.log("ndcg_val", ndcg_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log_metric

    def _reduce_metric(self, outputs):
        '''
        For input of form [(uid, uid_metric_Val)], reduce
        '''
        map_final, ndcg_final = [], []
        user_maps = [map_list['map_val'] for map_list in outputs]
        user_maps_flat = list(itertools.chain(*user_maps))
        for key, group in itertools.groupby(sorted(user_maps_flat), operator.itemgetter(0)):
            map_per_user = 0.0
            for item in group:
                map_per_user += item[1]
            map_final.append(map_per_user)

        user_ndcg = [map_list['ndcg_val'] for map_list in outputs]
        user_ndcg_flat = list(itertools.chain(*user_ndcg))
        for key, group in itertools.groupby(sorted(user_ndcg_flat), operator.itemgetter(0)):
            ndcg_per_user = 0.0
            for item in group:
                ndcg_per_user += item[1]
            ndcg_final.append(ndcg_per_user)

        return np.mean(map_final), np.mean(ndcg_final)

    def _reduce_metric_val(self, outputs):
        '''
        For input of form [(uid, uid_metric_Val)], reduce
        '''
        map_final, ndcg_final = [], []
        user_maps = [map_list['map_val'] for map_list in outputs]
        user_maps_flat = list(itertools.chain(*user_maps))
        for key, group in itertools.groupby(sorted(user_maps_flat), operator.itemgetter(0)):
            map_per_user = 0.0
            prop_per_user = 0.0
            for item in group:
                map_per_user   += item[1]/item[2]
                prop_per_user  += 1/item[2]
            #try:
            map_final.append(map_per_user/prop_per_user)
            #except:
            #map_final.append(0.)

        user_ndcg = [map_list['ndcg_val'] for map_list in outputs]
        user_ndcg_flat = list(itertools.chain(*user_ndcg))
        for key, group in itertools.groupby(sorted(user_ndcg_flat), operator.itemgetter(0)):
            ndcg_per_user = 0.0
            prop_per_user = 0.0
            for item in group:
                ndcg_per_user += item[1]/item[2]
                prop_per_user += 1/item[2]
            #try:
            ndcg_final.append(ndcg_per_user/prop_per_user)
            #except:
            #    ndcg_final.append(0.)

        return np.mean(map_final), np.mean(ndcg_final)

    def validation_epoch_end(self, outputs):
        map_val, ndcg_val = self._reduce_metric_val(outputs)
        self.log("ptl/val_ndcg", ndcg_val)
        self.log("ptl/val_map", map_val)

    def test_step(self, batch, batch_idx):
        uid, pid, click, prop, user_click_vec, item_click_vec, _, _ = batch
        user_click_vec = user_click_vec.squeeze(1)
        item_click_vec = item_click_vec.squeeze(2)
        mu_u = self.mu_u[uid]
        #mu_u = mu_u.to(self.device)
        #self.mu_i = self.mu_i.to(self.device)
        # get predictions for users in the batch and all items using mean of the variational distrs
        user_all_scores = (mu_u @ self.mu_i.T) #+  self.bias
        user_item_ranking = torch.argsort(user_all_scores, dim=-1, descending=True)[:, :self.topk]
        test_rank = (user_item_ranking == pid.unsqueeze(1)).int()
        test_rank = test_rank.cpu().detach().numpy()
        map_test = list(map(lambda x: rank_metrics.average_precision(x), test_rank))
        map_dict = list(zip(uid.cpu().detach().numpy(), map_test))
        ndcg_test = list(map(lambda x: rank_metrics.dcg_at_k(x, 5), test_rank))
        dcg_dict = list(zip(uid.cpu().detach().numpy(), ndcg_test))
        #logits = self(uid, pid)
        #val_loss = F.cross_entropy(y_hat, y)
        log_metric = {'map_val': map_dict, 'ndcg_val': dcg_dict, 'batch_lens': len(test_rank)}
        #self.log("ndcg_val", ndcg_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log_metric

    def test_epoch_end(self, outputs):
        map_val, ndcg_val = self._reduce_metric(outputs)
        self.log("test/ndcg_test", ndcg_val)
        self.log("test/map_test", map_val)


# Define MF Model
class MF(LightningModule):
    '''
    Logistic MF model w/o bias
    '''
    def __init__(self, num_users, num_items, lf_dim, lr=3*1e-3, topk=5, reg=2*1e-5, clip=0.05, ips=False):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=lf_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=lf_dim)
        
        self.bias_user = nn.Embedding(num_users, 1)
        self.bias_item = nn.Embedding(num_items, 1)
        # Initialization stabalizes the training
        # default init. statergy is randomly sampling from normal(0,1), which initializes most of the weights to 0 in expectation.
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        #nn.init.uniform_(self.bias_user.weight, -1.0, 1.0)
        #nn.init.uniform_(self.bias_item.weight, -1.0, 1.0)
        self.bias = nn.Parameter(torch.ones(1))
        #self.output = nn.Sigmoid()
        self.lr = lr
        self.topk = topk
        self.loss = nn.BCEWithLogitsLoss()
        self.reg = reg
        self.clip = torch.tensor(clip)
        self.clip.requires_grad = False
        # MF with or w/o IPS
        self.ips = ips

    def l2_regularize(self, array):
        return torch.sum(array ** 2.0)

    def forward(self, uid, pid):
        user_embedded = self.user_embedding(uid)
        item_embedded = self.item_embedding(pid)
        user_bias = self.bias_user(uid).squeeze()
        item_bias = self.bias_item(pid).squeeze()
        ui_factor = torch.sum(user_embedded * item_embedded, dim=-1)
        #pred = ui_factor + user_bias + item_bias + self.bias
        pred = ui_factor #+ self.bias
        #logit = self.output(pred)
        return pred

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        EPS = 1e-7
        uid, pid, click, prop, _, _, _, _ = batch
        prop = torch.maximum(prop, self.clip)
        logits = self(uid, pid)
        #reg_user_bias = self.l2_regularize(self.bias_user.weight) 
        #reg_item_bias = self.l2_regularize(self.bias_item.weight) 
        reg_user_embed = self.l2_regularize(self.user_embedding.weight) 
        reg_item_bias = self.l2_regularize(self.item_embedding.weight) 
        logits = torch.sigmoid(logits)
        if not self.ips:
            ce_loss = click * torch.log(logits + EPS) + (1 - click) * torch.log(1 - logits)
        else:
            ce_loss = (click/prop) * torch.log(logits + EPS) + (1 - click/prop) * torch.log(1 - logits + EPS)
        ce_loss = -torch.mean(ce_loss)
        #loss = self.loss(logits, click) + self.reg * (reg_user + reg_item)
        loss = ce_loss + self.reg * (reg_user_embed + reg_item_bias)
        self.log("training loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        uid, pid, click, prop, _, _, _, _ = batch
        #prop = torch.maximum(prop, self.clip)
        # Generate ranking score over all items for each user in the validation set
        user_embeddings = self.user_embedding(uid)
        user_bias = self.bias_user(uid)
        all_item_bias = self.bias_item.weight
        all_item_embeddings = self.item_embedding.weight
        #user_all_scores = (user_embeddings @ all_item_embeddings.T) + user_bias + all_item_bias.T + self.bias
        user_all_scores = (user_embeddings @ all_item_embeddings.T) #+  self.bias
        log_metric = {}
        # Get metrics at cut-offs - {1, 3, 5}
        for k in [1,3,5]:
            user_item_ranking = torch.argsort(user_all_scores, dim=1, descending=True)[:, :k]
            # Create dictionary to store (uid, dcgs) pairs
            dcg_dict = {}
            val_rank = (user_item_ranking == pid.unsqueeze(1)).int()
            val_rank = val_rank.cpu().detach().numpy()
            #map_val = rank_metrics.mean_average_precision(val_rank)
            # get validation metrics at cut-off - {1, 3, 5}
            map_val = list(map(lambda x: rank_metrics.average_precision(x), val_rank))
            map_dict = list(zip(uid.cpu().detach().numpy(), map_val, prop.cpu().detach().numpy()))
            ndcg_val = list(map(lambda x: rank_metrics.dcg_at_k(x, k), val_rank))
            dcg_dict = list(zip(uid.cpu().detach().numpy(), ndcg_val, prop.cpu().detach().numpy() ))
            prop_dict = list(zip(uid.cpu().detach().numpy(), prop.cpu().detach().numpy()))
            log_metric['map_val_%s'%str(k)] = map_dict
            log_metric['ndcg_val_%s'%str(k)] = dcg_dict
            log_metric['prop_batch'] = prop_dict
        #log_metric = {'map_val': map_dict, 'ndcg_val': dcg_dict, 'prop_batch': prop_dict}
        #self.log("ndcg_val", ndcg_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log_metric

    def _reduce_metric(self, outputs):
        '''
        For input of form [(uid, uid_metric_Val)], reduce
        '''
        metric_vals = {}
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                metric_final = []
                user_metric = [map_list[metric + '_%s'%str(k)] for map_list in outputs]
                user_metric_flat = list(itertools.chain(*user_metric))
                for key, group in itertools.groupby(sorted(user_metric_flat), operator.itemgetter(0)):
                    metric_per_user = 0.0
                    prop_per_user = 0.0
                    for item in group:
                        metric_per_user += item[1]
                    #try:
                    metric_final.append(metric_per_user)
                    #except:
                    #    ndcg_final.append(0.)
                metric_vals[metric + '_%s'%str(k)] = np.mean(metric_final)
        return metric_vals

    def _reduce_metric_val(self, outputs):
        '''
        For input of form [(uid, uid_metric_Val)], reduce
        '''
        metric_vals = {}
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                metric_final = []
                user_metric = [map_list[metric + '_%s'%str(k)] for map_list in outputs]
                user_metric_flat = list(itertools.chain(*user_metric))
                for key, group in itertools.groupby(sorted(user_metric_flat), operator.itemgetter(0)):
                    metric_per_user = 0.0
                    prop_per_user = 0.0
                    for item in group:
                        metric_per_user += item[1]/item[2]
                        prop_per_user += 1/item[2]
                    #try:
                    metric_final.append(metric_per_user/prop_per_user)
                    #except:
                    #    ndcg_final.append(0.)
                metric_vals[metric + '_%s'%str(k)] = np.mean(metric_final)
        return metric_vals

    def validation_epoch_end(self, outputs):
        metric_vals = self._reduce_metric_val(outputs)
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                self.log("ptl/" + metric + '_%s'%str(k), metric_vals[metric + '_%s'%str(k)])

    def test_step(self, batch, batch_idx):
        uid, pid, click, prop, _, _, _, _ = batch
        #print(click)
        # Generate ranking score over all items for each user in the validation set
        user_embeddings = self.user_embedding(uid)
        user_bias = self.bias_user(uid)
        all_item_bias = self.bias_item.weight
        all_item_embeddings = self.item_embedding.weight
        user_all_scores = (user_embeddings @ all_item_embeddings.T)
        #user_all_scores = (user_embeddings @ all_item_embeddings.T) + user_bias + all_item_bias.T + self.bias
        log_metric = {}
        for k in [1,3,5]:
            user_item_ranking = torch.argsort(user_all_scores, dim=-1, descending=True)[:, :k]
            test_rank = (user_item_ranking == pid.unsqueeze(1)).int()
            test_rank = test_rank.cpu().detach().numpy()
            map_test = list(map(lambda x: rank_metrics.average_precision(x), test_rank))
            map_dict = list(zip(uid.cpu().detach().numpy(), map_test))
            ndcg_test = list(map(lambda x: rank_metrics.dcg_at_k(x, 5), test_rank))
            dcg_dict = list(zip(uid.cpu().detach().numpy(), ndcg_test))
            #logits = self(uid, pid)
            #val_loss = F.cross_entropy(y_hat, y)
            #log_metric = {'map_val': map_dict, 'ndcg_val': dcg_dict, 'batch_lens': len(test_rank)}
            log_metric['map_val_%s'%str(k)] = map_dict
            log_metric['ndcg_val_%s'%str(k)] = dcg_dict
            log_metric['batch_lens'] = len(test_rank)
        #self.log("ndcg_val", ndcg_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log_metric

    def test_epoch_end(self, outputs):
        metric_vals = self._reduce_metric(outputs)
        for metric in ['map_val', 'ndcg_val']:
            for k in [1,3,5]:
                self.log("test/" + metric + '_%s'%str(k), metric_vals[metric + '_%s'%str(k)])


    

if __name__ == '__main__':
    n_users = 100
    n_movies = 100
    n_factors = 10
    model = RecommenderV1(n_users, n_movies, n_factors, n_factors, 0.001)
    model.summary()
    for trainable_var in model.trainable_variables:

        print(trainable_var.shape)
    
        

    