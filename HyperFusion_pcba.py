import numpy as np
from ogb.graphproppred import Evaluator
from sklearn.metrics.pairwise import pairwise_distances


evaluator = Evaluator('ogbg-molpcba')

def eval(y_true, y_pred):
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['ap']

test_all_res = []
val_all_res = []
for run in range(10):
    CRaWI_pred_test = np.load(f'scores_pcba/CRaWI_y_pred_test_{run}.npy')
    CRaWI_true_test = np.load(f'scores_pcba/CRaWI_y_true_test_{run}.npy')
    CRaWI_pred_val = np.load(f'scores_pcba/CRaWI_y_pred_val_{run}.npy')
    CRaWI_true_val = np.load(f'scores_pcba/CRaWI_y_true_val_{run}.npy')


    GINAK_pred_test = np.load(f'scores_pcba/GINAK_y_pred_test_{run}.npy')
    GINAK_true_test = np.load(f'scores_pcba/GINAK_y_true_test_{run}.npy')
    GINAK_pred_val = np.load(f'scores_pcba/GINAK_y_pred_val_{run}.npy')
    GINAK_true_val = np.load(f'scores_pcba/GINAK_y_true_val_{run}.npy')

    PHC_pred_test = np.load(f'scores_pcba/PHC_y_pred_test_{run}.npy')
    PHC_true_test = np.load(f'scores_pcba/PHC_y_true_test_{run}.npy')
    PHC_pred_val = np.load(f'scores_pcba/PHC_y_pred_val_{run}.npy')
    PHC_true_val = np.load(f'scores_pcba/PHC_y_true_val_{run}.npy')


    CRaWI_pred_test = CRaWI_pred_test#.squeeze()
    GINAK_pred_test = GINAK_pred_test#.squeeze()
    PHC_pred_test = PHC_pred_test#.squeeze()


    pred_test = np.stack((CRaWI_pred_test, GINAK_pred_test, PHC_pred_test), axis=0)#.transpose((1, 2, 0))  # 3 n 128
    pred_val = np.stack((CRaWI_pred_val, GINAK_pred_val, PHC_pred_val), axis=0)#.transpose((1, 2, 0))  # 3 n 128

    test_feat = pred_test.reshape((3, -1))
    val_feat = pred_val.reshape((3, -1))
    d = pairwise_distances(test_feat, metric='cosine')
    H = np.clip(d, 0, 1)

    A = H @ H.T
    pred_test = A @ test_feat
    pred_test = pred_test.reshape((3, -1, 128))
    pred_test = pred_test.mean(0)
    pred_val = A @ val_feat
    pred_val = pred_val.reshape((3, -1, 128))
    pred_val = pred_val.mean(0)


    true_test = CRaWI_true_test
    true_val = CRaWI_true_val

    test_res = eval(true_test, pred_test)
    val_res = eval(true_val, pred_val)
    test_all_res.append(test_res)
    val_all_res.append(val_res)


print(f'{np.mean(test_all_res)}, {np.std(test_all_res)}')
print(f'{np.mean(val_all_res)}, {np.std(val_all_res)}')
