# Parameter-Efficient Prompt Tuning Makes Generalized and Calibrated Neural Text Retrievers

Parameter-efficient learning can boost cross-domain and cross-topic generalization and calibration. [[Paper]](https://arxiv.org/pdf/2207.07087.pdf)

<p align="center">
  <img src="https://github.com/THUDM/P-Tuning-v2/blob/main/PT-Retrieval/figures/PT-Retrieval.png?raw=true" alt="PT-Retrieval"/ width="600px">
</p>

## Setup
Create the python environment via `Anaconda3`:
```bash
conda create -n pt-retrieval -y python=3.8.13
conda activate pt-retrieval
```

Install the necessary python packages. Change `cudatoolkit` version according to your environment (`11.3` in our experiment).

```bash
conda install -n pt-retrieval pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

(Optional) To train with Adapter, install Adapter-transformers

```bash
pip install -U adapter-transformers
```

## Data

### OpenQA

We use the preprocessed data provided by DPR which can be downloaded from the cloud using `download_data.py`. One needs to specify the resource name to be downloaded. Run `python download_data.py'`to see all options.

```bash
python data/download_data.py \
	--resource {key from download_data.py's RESOURCES_MAP} 
```

The keys of the five datasets and retrieval corpus we used in our experiments are:

- `data.retriever.nq` and `data.retriever.qas.nq`
- `data.retriever.trivia` and `data.retriever.qas.trivia`
- `data.retriever.squad1` and `data.retriever.qas.squad1`
- `data.retriever.webq` and `data.retriever.qas.webq`
- `data.retriever.curatedtre` and `data.retriever.qas.curatedtrec`
- `data.wikipedia_split.psgs_w100`

**NOTE**: The resource name matching is prefix-based. So if you need to download all training data, just use ` --resource data.retriever`.

### BEIR

Run the evaluation script for DPR on BEIR and the script will download the dataset automatically. The decompressed data will be saved at `./beir_eval/datasets`.

Or you can download the data manually via: 

```bash
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$dataset.zip
```

### OAG-QA

OAG-QA is the largest public topic-specific passage retrieval dataset, which consists of 17,948 unique queries from 22 scientific disciplines and 87 fine-grained topics.
The queries and reference papers are collected from professional forums such as [Zhihu](https://zhihu.com) and [StackExchange](https://stackexchange.com), and mapped to papers in the [Open Academic Graph](https://www.aminer.cn/oag-2-1) with rich meta-information (e.g. abstract, field-of-study (FOS)).

Download OAG-QA from the link: [OAG-QA](https://drive.google.com/file/d/1jEAzWq_J0cXz1pvT9nQ2s2ooGdcR8r6-/view?usp=sharing), and unzip it to `./data/oagqa-topic-v2`


## Training

### DPR

Five training modes are supported in the code and the corresponding training scripts are provided:

- Fine-tuning
- P-tuning v2
- BitFit
- Adapter
- Lester et al. & P-tuning

Run training scripts in `./run_scripts`, for example:

```bash
bash run_scripts/run_train_dpr_multidata_ptv2.sh
```

### ColBERT

P-Tuning v2 and the original finetuning are supported in colbert.

```bash
cd colbert
bash scripts/run_train_colbert_ptv2.sh
```



## Checkpoints for Reproduce
Download the checkpoints we use in the experiments to reproduce our results in the paper.
Change `--model_file ./checkpoints/$filename/$checkpoint` in each evaluation script `*.sh` from `./eval_scripts` to load them.

| Checkpoints | DPR                                                          | ColBERT                                                      |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| P-tuning v2 | [Download](https://drive.google.com/file/d/1jVVndoHScqMMcJOBcJi2lVQvIhKAwaGN/view?usp=sharing) | [Download](https://drive.google.com/file/d/1JZYmRKoobs4vfaIXsEX_DNkluKmXxmzD/view?usp=sharing) |
| Fine-tune   | [Download](https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/multiset/hf_bert_base.cp) | [Download](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/models/ColBERT/msmarco.psg.l2.zip) |



## Evaluation

### DPR

#### 1. OpenQA

Inference is divied as two steps.

- **Step 1**: Generate representation vectors for the static documents dataset via:

```bash
bash eval_scripts/generate_wiki_embeddings.sh
```

- **Step 2**: Retrieve for the validation question set from the entire set of candidate documents and calculate the top-k retrieval accuracy. To select the validation dataset, you can replace `$dataset` with `nq`, `trivia`, `webq` or `curatedtrec`.

```bash
bash eval_scripts/evaluate_on_openqa.sh $dataset $top-k
```

#### 2. BEIR

Evaluate DPR on BEIR via:

```bash
bash eval_scripts/evaluate_on_beir.sh $dataset
```

You can choose `$dataset` from 15 datasets from BEIR, which can be referred to [BEIR](https://github.com/beir-cellar/beir#beers-available-datasets).

#### 3. OAG-QA

There are 87 topics in OAG-QA and you can choose any topic to run the evaluation via:

```bash
bash eval_scripts/evaluate_on_oagqa.sh $topic $top-k
```

### ColBERT

#### 1. BEIR

Similar to the BEIR evaluation for DPR, run the script to evaluate ColBERT on BEIR:

```bash
cd colbert
bash scripts/evalute_on_beir.sh $dataset
```



## Calibration for DPR

#### OAG-QA

Plot the calibration curve and calculate ECE via:

```bash
bash calibration_on_openqa.sh $dataset
```

#### BEIR

Plot the calibration curve and calculate ECE via:

```bash
bash eval_scripts/calibration_on_beir.sh $dataset
```



## Citation
If you find this paper and repo useful, please consider citing us in your work

```
@article{WLTam2022PT-Retrieval,
  author    = {Weng Lam Tam and
               Xiao Liu and
               Kaixuan Ji and
               Lilong Xue and
               Xingjian Zhang and
               Yuxiao Dong and
               Jiahua Liu and
               Maodi Hu and
               Jie Tang},
  title     = {Parameter-Efficient Prompt Tuning Makes Generalized and Calibrated
               Neural Text Retrievers},
  journal   = {CoRR},
  volume    = {abs/2207.07087},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2207.07087},
  doi       = {10.48550/arXiv.2207.07087},
  eprinttype = {arXiv},
  eprint    = {2207.07087},
  timestamp = {Tue, 19 Jul 2022 17:45:18 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2207-07087.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
