import json
import pandas as pd
import shapely
import area
import numpy as np
import random
import torch
from zipfile import ZipFile
from math import sqrt, sin, cos, pi, asin
from ast import literal_eval
import itertools
from sklearn.model_selection import KFold

from importlib.machinery import SourceFileLoader

path = 'deepgravity/models/od_models.py'
od = SourceFileLoader('od', path).load_module()


def df_to_dict(df):
    split = df.to_dict(orient='split')
    keys = split['index']
    values = split['data']
    return {k: v for k, v in zip(keys, values)}


def get_features_ffnn(oa_origin, oa_destination, oa2features, oa2centroid, df, distances, k):
    if df == 'deepgravity':
        dist_od = earth_distance(
            oa2centroid[oa_origin], oa2centroid[oa_destination])
        return oa2features[oa_origin] + oa2features[oa_destination] + [dist_od]

    elif df == 'deepgravity_knn':
        return oa2features[oa_origin] + oa2features[oa_destination] + distances[oa_origin] + distances[oa_destination]
    else:
        dist_od = earth_distance(
            oa2centroid[oa_origin], oa2centroid[oa_destination])
        return [np.log(oa2features[oa_origin])] + [np.log(oa2features[oa_destination])] + [dist_od]


def get_flow(oa_origin, oa_destination, o2d2flow):
    try:
        return o2d2flow[oa_origin][oa_destination]
    except KeyError:
        return 0


def get_destinations(oa, size_train_dest, all_locs_in_train_region, o2d2flow, frac_true_dest=0.5):
    try:
        true_dests_all = list(o2d2flow[oa].keys())
    except KeyError:
        true_dests_all = []
    size_true_dests = min(
        int(size_train_dest * frac_true_dest), len(true_dests_all))
    size_fake_dests = size_train_dest - size_true_dests

    true_dests = np.random.choice(
        true_dests_all, size=size_true_dests, replace=False)
    fake_dests_all = list(set(all_locs_in_train_region) - set(true_dests))
    fake_dests = np.random.choice(
        fake_dests_all, size=size_fake_dests, replace=False)

    dests = np.concatenate((true_dests, fake_dests))
    np.random.shuffle(dests)
    return dests


def split_train_test_sets(oas, fraction_train):
    n = len(oas)
    dim_train = int(n * fraction_train)

    random.shuffle(oas)
    train_locs = oas[:dim_train]
    test_locs = oas[dim_train:]

    return train_locs, test_locs


class NN_MultinomialRegression(od.NN_OriginalGravity):

    def __init__(self, dim_input, dim_hidden, df, dropout_p=0.35,  device=torch.device("cpu")):
        super(od.NN_OriginalGravity, self).__init__(dim_input, device=device)

        self.df = df
        self.device = device
        p = dropout_p

        self.linear1 = torch.nn.Linear(dim_input, dim_hidden)
        self.relu1 = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(p)

        self.linear2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu2 = torch.nn.LeakyReLU()
        self.dropout2 = torch.nn.Dropout(p)

        self.linear3 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu3 = torch.nn.LeakyReLU()
        self.dropout3 = torch.nn.Dropout(p)

        self.linear4 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu4 = torch.nn.LeakyReLU()
        self.dropout4 = torch.nn.Dropout(p)

        self.linear5 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu5 = torch.nn.LeakyReLU()
        self.dropout5 = torch.nn.Dropout(p)

        self.linear6 = torch.nn.Linear(dim_hidden, dim_hidden // 2)
        self.relu6 = torch.nn.LeakyReLU()
        self.dropout6 = torch.nn.Dropout(p)

        self.linear7 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu7 = torch.nn.LeakyReLU()
        self.dropout7 = torch.nn.Dropout(p)

        self.linear8 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu8 = torch.nn.LeakyReLU()
        self.dropout8 = torch.nn.Dropout(p)

        self.linear9 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu9 = torch.nn.LeakyReLU()
        self.dropout9 = torch.nn.Dropout(p)

        self.linear10 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu10 = torch.nn.LeakyReLU()
        self.dropout10 = torch.nn.Dropout(p)

        self.linear11 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu11 = torch.nn.LeakyReLU()
        self.dropout11 = torch.nn.Dropout(p)

        self.linear12 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu12 = torch.nn.LeakyReLU()
        self.dropout12 = torch.nn.Dropout(p)

        self.linear13 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu13 = torch.nn.LeakyReLU()
        self.dropout13 = torch.nn.Dropout(p)

        self.linear14 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu14 = torch.nn.LeakyReLU()
        self.dropout14 = torch.nn.Dropout(p)

        self.linear15 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu15 = torch.nn.LeakyReLU()
        self.dropout15 = torch.nn.Dropout(p)

        self.linear_out = torch.nn.Linear(dim_hidden // 2, 1)

    def forward(self, vX):
        lin1 = self.linear1(vX)
        h_relu1 = self.relu1(lin1)
        drop1 = self.dropout1(h_relu1)

        lin2 = self.linear2(drop1)
        h_relu2 = self.relu2(lin2)
        drop2 = self.dropout2(h_relu2)

        lin3 = self.linear3(drop2)
        h_relu3 = self.relu3(lin3)
        drop3 = self.dropout3(h_relu3)

        lin4 = self.linear4(drop3)
        h_relu4 = self.relu4(lin4)
        drop4 = self.dropout4(h_relu4)

        lin5 = self.linear5(drop4)
        h_relu5 = self.relu5(lin5)
        drop5 = self.dropout5(h_relu5)

        lin6 = self.linear6(drop5)
        h_relu6 = self.relu6(lin6)
        drop6 = self.dropout6(h_relu6)

        lin7 = self.linear7(drop6)
        h_relu7 = self.relu7(lin7)
        drop7 = self.dropout7(h_relu7)

        lin8 = self.linear8(drop7)
        h_relu8 = self.relu8(lin8)
        drop8 = self.dropout8(h_relu8)

        lin9 = self.linear9(drop8)
        h_relu9 = self.relu9(lin9)
        drop9 = self.dropout9(h_relu9)

        lin10 = self.linear10(drop9)
        h_relu10 = self.relu10(lin10)
        drop10 = self.dropout10(h_relu10)

        lin11 = self.linear11(drop10)
        h_relu11 = self.relu11(lin11)
        drop11 = self.dropout11(h_relu11)

        lin12 = self.linear12(drop11)
        h_relu12 = self.relu12(lin12)
        drop12 = self.dropout12(h_relu12)

        lin13 = self.linear13(drop12)
        h_relu13 = self.relu13(lin13)
        drop13 = self.dropout13(h_relu13)

        lin14 = self.linear14(drop13)
        h_relu14 = self.relu14(lin14)
        drop14 = self.dropout14(h_relu14)

        lin15 = self.linear15(drop14)
        h_relu15 = self.relu15(lin15)
        drop15 = self.dropout15(h_relu15)

        out = self.linear_out(drop15)
        return out

    def get_features(self, oa_origin, oa_destination, oa2features, oa2centroid, df):
        return get_features_ffnn(oa_origin, oa_destination, oa2features, oa2centroid, df)


param_grid = {
    'lr': [0.001, 0.0001],
    'dropout_rate': [0.3, 0.5],
    'hidden_layers': [2, 3]
}


def train_and_evaluate(train_loader, test_loader, epochs, lr, dropout_rate, hidden_layers, seed, dim_input, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = NN_MultinomialRegression(
        dim_input, dim_hidden=512, df='deepgravity', dropout_p=dropout_rate, device=device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, data_temp in enumerate(train_loader):
            b_data = data_temp[0]
            b_target = data_temp[1]
            optimizer.zero_grad()
            loss = 0.0
            for data, target in zip(b_data, b_target):
                output = model(data)
                loss += model.loss(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch: {epoch}, Loss: {running_loss / len(train_loader)}')

    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_accuracy = 0.0
        n_origins = 0

        for batch_idx, data_temp in enumerate(test_loader):
            b_data = data_temp[0]
            b_target = data_temp[1]
            for data, target in zip(b_data, b_target):
                output = model(data)
                test_loss += model.loss(output, target).item()
                cpc = model.get_cpc(data, target)
                test_accuracy += cpc
                n_origins += 1

        test_loss /= n_origins
        test_accuracy /= n_origins

    return model, optimizer, test_loss, test_accuracy


def hyperparameter_tuning(train_data, train_dataset_args, args, dim_input, device, dgd):
    best_params = None
    best_cpc = float('-inf')

    for params in itertools.product(param_grid['lr'], param_grid['dropout_rate'], param_grid['hidden_layers']):
        lr, dropout_rate, hidden_layers = params
        print(
            f"Evaluating with params: lr={lr}, dropout_rate={dropout_rate}, hidden_layers={hidden_layers}")

        kfold = KFold(n_splits=3, shuffle=True, random_state=args.seed)
        cpc_scores = []

        for train_idx, val_idx in kfold.split(train_data):
            train_subset = torch.utils.data.Subset(
                dgd.FlowDataset(train_data, **train_dataset_args), train_idx)
            val_subset = torch.utils.data.Subset(
                dgd.FlowDataset(train_data, **train_dataset_args), val_idx)

            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=args.batch_size)
            val_loader = torch.utils.data.DataLoader(
                val_subset, batch_size=args.test_batch_size)

            _, _, _, cpc = train_and_evaluate(
                train_loader, val_loader, args.epochs, lr, dropout_rate, hidden_layers, args.seed, dim_input, device)
            cpc_scores.append(cpc)

        avg_cpc_score = np.mean(cpc_scores)
        print(f"Avg CPC for params {params}: {avg_cpc_score}")

        if avg_cpc_score > best_cpc:
            best_cpc = avg_cpc_score
            best_params = params

    return best_params


def evaluate():
    loc2cpc_numerator = {}

    model.eval()
    with torch.no_grad():
        for data_temp in test_loader:
            b_data = data_temp[0]
            b_target = data_temp[1]
            ids = data_temp[2]
            for id, data, target in zip(ids, b_data, b_target):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model.forward(data)
                cpc = model.get_cpc(data, target, numerator_only=True)
                loc2cpc_numerator[id[0]] = cpc
    edf = pd.DataFrame.from_dict(loc2cpc_numerator, columns=['cpc_num'], orient='index').reset_index().rename(
        columns={'index': 'locID'})
    oa2tile = {oa: t for t, v in tileid2oa2features2vals.items()
               for oa in v.keys()}

    def cpc_from_num(edf, oa2tile, o2d2flow):
        print(edf.head())
        edf['tile'] = edf['locID'].apply(lambda x: oa2tile[x])
        edf['tot_flow'] = edf['locID'].apply(lambda x: sum(
            o2d2flow[x].values()) if x in o2d2flow else 1e-6)
        cpc_df = pd.DataFrame(edf.groupby('tile').apply(
            lambda x: x['cpc_num'].sum() / 2 / x['tot_flow'].sum()),
            columns=['cpc']).reset_index()
        return cpc_df

    cpc_df = cpc_from_num(edf, oa2tile, o2d2flow)

    average_cpc = cpc_df['cpc'].mean()
    cpc_stdev = cpc_df['cpc'].std()

    print(
        f'Average CPC of test tiles: {average_cpc:.4f}  stdev: {cpc_stdev:.4f}')

    fname = 'deepgravity/results/tile2cpc_{}_{}.csv'.format(
        model_type, args.dataset)

    cpc_df.to_csv(fname, index=False)
