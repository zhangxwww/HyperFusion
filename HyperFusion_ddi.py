import numpy as np
from ogb.linkproppred import Evaluator
from sklearn.metrics.pairwise import pairwise_distances

evaluator = Evaluator(name='ogbl-ddi')

def hits(evaluator, pos_val, neg_val, pos_test, neg_test):
    k = 20
    evaluator.K = k
    val_hits = evaluator.eval({
        'y_pred_pos': pos_val,
        'y_pred_neg': neg_val,
    })['hits@20']
    test_hits = evaluator.eval({
        'y_pred_pos': pos_test,
        'y_pred_neg': neg_test,
    })['hits@20']

    return val_hits, test_hits


root_dir = 'scores_ddi'

val_res, test_res = [], []

for run in range(10):

    agdn_val_pos = np.load(f'{root_dir}/agdn_val_pos_{run}.npy')
    agdn_val_neg = np.load(f'{root_dir}/agdn_val_neg_{run}.npy')
    agdn_test_pos = np.load(f'{root_dir}/agdn_test_pos_{run}.npy')
    agdn_test_neg = np.load(f'{root_dir}/agdn_test_neg_{run}.npy')


    e2n_val_pos = np.load(f'{root_dir}/E2N_val_pos_{run}.npy')
    e2n_val_neg = np.load(f'{root_dir}/E2N_val_neg_{run}.npy')
    e2n_test_pos = np.load(f'{root_dir}/E2N_test_pos_{run}.npy')
    e2n_test_neg = np.load(f'{root_dir}/E2N_test_neg_{run}.npy')

    psg_val_pos = np.load(f'{root_dir}/psg_val_pos_{run}.npy')
    psg_val_neg = np.load(f'{root_dir}/psg_val_neg_{run}.npy')
    psg_test_pos = np.load(f'{root_dir}/psg_test_pos_{run}.npy')
    psg_test_neg = np.load(f'{root_dir}/psg_test_neg_{run}.npy')

    val_pos = np.stack((agdn_val_pos, e2n_val_pos, psg_val_pos))
    val_neg = np.stack((agdn_val_neg, e2n_val_neg, psg_val_neg))
    test_pos = np.stack((agdn_test_pos, e2n_test_pos, psg_test_pos))
    test_neg = np.stack((agdn_test_neg, e2n_test_neg, psg_test_neg))

    d1 = pairwise_distances(val_pos, metric='cosine')
    d2 = pairwise_distances(val_neg, metric='cosine')
    d3 = pairwise_distances(test_pos, metric='cosine')
    d4 = pairwise_distances(test_neg, metric='cosine')

    d = [d1, d2, d3, d4]
    H = np.zeros((3, 4))

    for idx, dd in enumerate(d):
        where = np.argwhere((dd > 0) & (dd < 0.1))
        for w in where:
            for i in w:
                H[i, idx] = 1

    A = H @ H.T
    val_pos = A @ val_pos
    val_neg = A @ val_neg
    test_pos = A @ test_pos
    test_neg = A @ test_neg

    final_val_pos = val_pos.sum(0)
    final_val_neg = val_neg.sum(0)
    final_test_pos = test_pos.sum(0)
    final_test_neg = test_neg.sum(0)

    v_res, t_res = hits(
                evaluator,
                final_val_pos,
                final_val_neg,
                final_test_pos,
                final_test_neg)
    val_res.append(v_res)
    test_res.append(t_res)


print(f'Test: {np.mean(test_res)} ± {np.std(test_res)}')
print(f'Valid: {np.mean(val_res)} ± {np.std(val_res)}')