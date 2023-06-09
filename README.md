# NetworkML Course Project


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