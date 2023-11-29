# Gen_model_with_GCN

## Requirements

General

- Python (verified on 3.8.16)
- CUDA (verified on 11.7)

Python Packages

- see requirements.txt

```bash
conda create -n gengcn python=3.8
conda activate gengcn
pip install -r requirements.txt
```

### Data Format

Data folder contains four files (The detailed preprocessing steps refer to `Event Datasets Preprocessing`):

```text
data/text2tree/one_ie_ace2005_subtype
├── event.schema
├── test.json
├── train.json
└── val.json
```

train/val/test.json are data files, and each line is a JSON instance.
Each JSON instance contains `text` and `event` fields, in which `text` is plain text, and `event` is event linearized form.

```text
{"text": "He also owns a television and a radio station and a newspaper .", "event": "<extra_id_0>  <extra_id_1>"}
{"text": "' ' For us the United Natgions is the key authority '' in resolving the Iraq crisis , Fischer told reporters opn arrival at the EU meeting .", "event": "<extra_id_0> <extra_id_0> Meet meeting <extra_id_0> Entity EU <extra_id_1> <extra_id_1> <extra_id_1>"}
```

Note:
- Use the extra character of T5 as the structure indicators, such as `<extra_id_0>`, `<extra_id_1>`, etc.
- `event.schema` is the event schema file for building the trie of constrained decoding.
It contains three lines: the first line is event type name list, the second line is event role name list, the third line is type-to-role dictionary.

  ```text
  ["Declare-Bankruptcy", "Convict", ...]
  ["Plaintiff", "Target", ...]
  {"End-Position": ["Place", "Person", "Entity"], ...}
  ```

###DYGIEPP ACE05 
####OneIE ACE05+ ERE

After data preprocessing and we get the following data files:

```text
 $ tree data/raw_data/
data/raw_data/
├── ace05-EN
│   ├── dev.oneie.json
│   ├── test.oneie.json
│   └── train.oneie.json
├── dyiepp_ace2005
│   ├── dev.json
│   ├── test.json
│   └── train.json
└── ERE-EN
    ├── dev.oneie.json
    ├── test.oneie.json
    └── train.oneie.json
```

We then convert the above data files to tree format.
The following scripts generate the corresponding data folder in `data/text2tree`.
The conversion will automatically generate `train/dev/test` JSON files and `event.schema` file.

```bash
bash scripts/processing_data.bash
```

```text
data/text2tree
├── dyiepp_ace2005_subtype
│   ├── event.schema
│   ├── test.json
│   ├── train.json
│   └── val.json
├── dyiepp_ace2005_subtype_span
│   ├── event.schema
│   ├── test.json
│   ├── train.json
│   └── val.json
├── one_ie_ace2005_subtype
│   ├── event.schema
│   ├── test.json
│   ├── train.json
│   └── val.json
├── one_ie_ace2005_subtype_span
│   ├── event.schema
│   ├── test.json
│   ├── train.json
│   └── val.json
├── one_ie_ere_en_subtype
│   ├── event.schema
│   ├── test.json
│   ├── train.json
│   └── val.json
└── one_ie_ere_en_subtype_span
    ├── event.schema
    ├── test.json
    ├── train.json
    └── val.json
```
### Model Training

Training scripts as follows:

- `run_seq2seq_gcn.bash`:  model training script, output to the screen directly.
- `run_seq2seq_with_pretrain.bash`: Model training script for curriculum learning, which contains substructure learning and full structure learning.

The command for the training is as follows (see bash scripts and Python files for the corresponding command-line
arguments):

```bash
bash run_seq2seq_gcn.bash -d 0 -f tree -m t5-base --label_smoothing 0 -l 1e-4 --lr_scheduler linear --warmup_steps 2000 -b 16 -i one_ie_ace2005_subtype
```
```bash
bash run_seq2seq_gcn.bash -d 0 -f tree -m t5-base --label_smoothing 0 -l 1e-4 --lr_scheduler linear --warmup_steps 2000 -b 8 -i one_ie_ere_en_subtype
```
```bash
bash run_seq2seq_with_pretrain.bash -d 0 -f tree -m t5_large --label_smoothing 0 -l 5e-5 --lr_scheduler linear --warmup_steps 2000 -b 8 -i one_ie_ere_en_subtype
```
```bash
bash run_seq2seq_with_pretrain.bash -d 0 -f tree -m t5_large --label_smoothing 0 -l 5e-5 --lr_scheduler linear --warmup_steps 2000 -b 8 -i dyiepp_ace2005_subtype
```
--max_train_samples=4300

- `-i` means dataset.
- `-d` refers to the GPU device id.
- `-m t5-base` refers to using T5-base.******but we load local model for training, so -m only use for substructure learning and as a symbol. To change model,we need to change raw codes.******
- Currently, constrained decoding algorithms shoud be changed to support bart-model.For bart,more changes are needed if you want.

Trained models are saved in the `models/` folder.


