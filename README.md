
## Training Fragment Injection Module
The lightweight fragment injection module is the only part that requires training in $f$-RAG.<br>
We provide the [data](https://docs.google.com/uc?export=download&id=1zWM5WY0mQEFUB0xIg4D7Ba_e4oifR2i7) to train the model and evaluate the results. Download and place the `data` folder in this directory.<br>

To train the module from scratch, first run the following command to preprocess the data:
```bash
python preprocess.py
```
We provide a partially preprocessed data file `data/zinc250k_train.csv` for ease of use. To preprocess the data from scratch, delete this file before running the preprocessing.

Then, run the following command to train the module:
```bash
python fusion/trainer/train.py \
    --dataset data/zinc250k \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 128 \
    --save_strategy epoch \
    --num_train_epochs 8 \
    --learning_rate 1e-4
```
We used a single NVIDIA GeForce RTX 3090 GPU to train the module.

## Running PMO Experiments (Section 4.1)
The folder `mol_opt` contains the code to run the experiments on the practical molecular optimization (PMO) benchmark and is based on the official [benchmark codebase](https://github.com/wenhao-gao/mol_opt).<br>
First run the following command to construct an initial fragment vocabulary:
```bash
python get_vocab.py pmo
```

Then, run the following command to run the experiments:
```bash
cd exps/pmo
python run.py -o ${oracle_name} -s ${seed}
```
You can adjust hyperparameters in `exps/pmo/main/f_rag/hparams.yaml`.

Run the following command to evaluate the generated molecules:
```bash
python eval.py ${file}
```

## Running Docking Experiments (Section 4.2)
The folder `dock` contains the code to run the experiments on the docking score optimization tasks.<br>
Before running the experiments, place the trained fragment injection module `model.safetensors` under the folder `dock`.
First run the following command to construct an initial fragment vocabulary:
```bash
python get_vocab.py dock
```

Then, run the following command to run the experiments:
```bash
cd exps/dock
python run.py -o ${oracle_name} -s ${seed}
```
You can adjust hyperparameters in `exps/dock/hparams.yaml`.

Run the following command to evaluate the generated molecules:
```bash
python eval.py ${file}
```


