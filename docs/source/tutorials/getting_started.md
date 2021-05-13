## Getting Started

- **Source Code:**   `$ git clone` this repo and install the Python dependencies from `requirements.txt`
- **Dataset** Download the dataset by filling out the
   form [here](https://docs.google.com/forms/d/10Nke6m8MvCxP7hoJQ_k-mtiejbXtE0RliX9w_8pooLQ/edit).

This document provides a brief intro of the usage of this repo.


### Inference with Pre-trained Models

1. Pick a model configuration from `utils.config.py` file.
2. We provide `img_trainer.py` that can be used to train and evaluate the models. 

```bash
usage: img_trainer.py [-h] [-m MODE]

optional arguments:
  -h, --help            show this help message and exit
  
  -m MODE, --mode MODE  {'train', 'test'} (default: 'test')
                        
  -n MODEL_NAME, --model_name MODEL_NAME  (default: 'my_model')
  
  -a ARCHITECTURE, --architecture ARCHITECTURE {'resnet18', 'resnet50', 'resnet101'} (default: 'resnet18')
```
Run the file `img_trainer.py` in test mode:
```
python img_trainer.py -m test
```
The configs are made for training as well as evaluation, therefore we need to specify these arguments.

### Training & Evaluation

The script `img_trainer.py` is used to train the models as well. A reference command for training the model is shown below
```
python img_trainer.py -m train -a resnet50
```


For detailed option summary, see `./train_net.py -h`.
