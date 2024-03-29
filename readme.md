# HyperFusion: Multi-Model Ensemble on Hypergraph

## Requirements

```
numpy=1.22.3
sklearn=1.3.2
ogb=1.3.6
```

## Reproduce

### Data preparation

First, the predictions of the bases models should be prepared.

The directory structure is as follows:

```
# for ogbl-collab and ogbl-ddi dataset:
|-- scores_{dataset}
    |-- {model_name_1}_val_pos_{run}.npy
    |-- {model_name_1}_val_neg_{run}.npy
    |-- {model_name_1}_test_pos_{run}.npy
    |-- {model_name_1}_test_neg_{run}.npy
    |-- {model_name_2}_val_pos_{run}.npy
    |-- {model_name_2}_val_neg_{run}.npy
    |-- {model_name_2}_test_pos_{run}.npy
    |-- {model_name_2}_test_neg_{run}.npy
    |-- ...

# for ogbg-molhiv and ogbg-molpcba dataset:
|-- scores_{dataset}
    |-- {model_name_1}_val_pred_{run}.npy
    |-- {model_name_1}_val_true_{run}.npy
    |-- {model_name_1}_test_pred_{run}.npy
    |-- {model_name_1}_test_true_{run}.npy
    |-- ...
```
### Run the python script

```bash
python HyperFusion_collab.py
python HyperFusion_ddi.py
python HyperFusion_molhiv.py
python HyperFusion_molpcba.py
```
