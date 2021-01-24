# DBA

## Requirements
Pytorch

## Prepare the dataset
### LOAN dataset:

- download the raw dataset into dir `./utils` 
- preprocess the dataset. 

```
cd ./utils
bash process_loan_data.sh
```

### Tiny-imagenet dataset:

- download the dataset into dir `./utils` 
- reformat the dataset.
```
cd ./utils
bash process_tiny_data.sh
```

### Others:
MNIST and CIFAR10 will be automatically download.

## Backdoor Attack
Modfiy the params.yaml file to conduct different experiment.
### Attack A-M vs. A-S  
Params:
- generation_epoch  
single-shot: [ different epochs for different attacker ]  
multi-shot: [ same epoch for attackers ]
- baseline  
single-shot: false; multi-shot: true
- eta  
single-shot: 0.1; multi-shot: 1
- poison_epochs  
single-shot: [ single epoch for each attacker ]  
multi-shot: [ multiple epochs for attackers ]

### Distributed vs. Centralized  
Params:
- adversary_list  
  distributed attackers: [ multiple attackers ]  
  centralized attacker: [ single attacker ]
- poison_epochs  
single-shot: X_poison_epochs  
multi-shot: 0_poison_epochs

### Trigger Generation  
  Our method for trigger generation is incorporated in `gen.py`  
  Set `is_generated: true` to apply our scheme.

## Reproduce experiments

We can use Visdom to monitor the training process.  
```
python -m visdom.server -p 8098
```

Run experiments for different datasets:
```
python main.py --params utils/X.yaml
```
`X` = `mnist_params`, `cifar_params`,`tiny_params` or `loan_params`. Parameters can be changed in those yaml files to reproduce our experiments.



