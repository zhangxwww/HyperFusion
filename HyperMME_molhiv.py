import numpy as np
from ogb.graphproppred import Evaluator
from sklearn.metrics.pairwise import pairwise_distances


evaluator = Evaluator('ogbg-molhiv')

def eval(y_true, y_pred):
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['rocauc']

all_res = []
all_val_res = []
for run in range(10):
    deepauc_test_pred = np.load(f'scores_molhiv/DeepAUC_test_pred_{run}.npy')
    deepauc_test_true = np.load(f'scores_molhiv/DeepAUC_test_true_{run}.npy')
    deepauc_val_pred = np.load(f'scores_molhiv/DeepAUC_val_pred_{run}.npy')
    deepauc_val_true = np.load(f'scores_molhiv/DeepAUC_val_true_{run}.npy')


    hig_test_pred = np.load(f'scores_molhiv/HIG_test_pred_{run}.npy')
    hig_test_true = np.load(f'scores_molhiv/HIG_test_true_{run}.npy')
    hig_val_pred = np.load(f'scores_molhiv/HIG_val_pred_{run}.npy')
    hig_val_true = np.load(f'scores_molhiv/HIG_val_true_{run}.npy')


    yougraph_test_pred = np.load(f'scores_molhiv/YouGraph_test_pred_{run}.npy')
    yougraph_test_true = np.load(f'scores_molhiv/YouGraph_test_true_{run}.npy')
    yougraph_val_pred = np.load(f'scores_molhiv/YouGraph_val_pred_{run}.npy')
    yougraph_val_true = np.load(f'scores_molhiv/YouGraph_val_true_{run}.npy')



    deepauc_test_pred = deepauc_test_pred.squeeze()
    hig_test_pred = hig_test_pred.squeeze()
    yougraph_test_pred = yougraph_test_pred.squeeze()

    deepauc_val_pred = deepauc_val_pred.squeeze()
    hig_val_pred = hig_val_pred.squeeze()
    yougraph_val_pred = yougraph_val_pred.squeeze()


    test_pred = np.stack((deepauc_test_pred, hig_test_pred, yougraph_test_pred))
    val_pred = np.stack((deepauc_val_pred, hig_val_pred, yougraph_val_pred))

    d = pairwise_distances(test_pred, metric='cosine')
    H = np.clip(d, 0, 1)

    A = H @ H.T
    test_pred = A @ test_pred
    test_pred = test_pred.sum(0, keepdims=True).T
    val_pred = A @ val_pred
    val_pred = val_pred.sum(0, keepdims=True).T

    test_true = deepauc_test_true
    val_true = deepauc_val_true

    res = eval(test_true, test_pred)
    all_res.append(res)
    val_res  = eval(val_true, val_pred)
    all_val_res.append(val_res)

print(f'{np.mean(all_res)}, {np.std(all_res)}')
print(f'{np.mean(all_val_res)}, {np.std(all_val_res)}')
