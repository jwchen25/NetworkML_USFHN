# NetworkML Course Project

## Note
- Title: Predicting University Rankings on the US Faculty Hiring Networks via GNNs
- Author: Junwu Chen
- Group 25
### All experiment results shown in the report are in `exp_results` folder


## create python env for this project
```
conda env create -f environment.yml
```

## process USFHN to get 4 subsets
```
cd dataset
python preprocessing.py
```

## train the all models on the all subsets
```
bash scripts/run_all.sh
```

## train the models using certain graph conv type on all subsets
```
# bash scripts/train.sh [CONV TYPE]
bash scripts/train.sh gcn
```

## plot all the figures of the report
```
python report_plots.py
```