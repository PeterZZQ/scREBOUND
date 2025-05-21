# scREBOUND test script
scREBOUND is a pretrained single-cell foundation model that generate cell embedding for multi-purpose downstream tasks.

## Dependency
```
python >= 3.12
anndata >= 0.11.4
sklearn >= 1.5.2
numpy >= 2.0.1
scanpy >= 1.11.1
pytorch == 2.5.0+cu124
flash-attn == 2.7.4.post1
```

## Directory
* `./batch_encoding/` stores the batch preprocessing script for training data 
* `./script_evaluation/` stores evaluation script of scREBOUND and baseline models
* `./script_train/` stores the training scripts for scREBOUND
* `./src/` stores the source code of scREBOUND

## Usage
```bash
python script_infer.py --config [config_file.yaml]
```

The `config_file.yaml` stores the configuration of the inference, which include the key parameters:
* seed: the random seed
* device: running device
* model_dir: the directory of the model state dict
* data_dir: the directory of the evaluation dataset
* output_dir: the output directory
See `config_pancreas.ymal` for the example setting on pancreas evaluation dataset.
