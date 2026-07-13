## Testbench for DNA-ESA


### Setup
Ensure that your data folder `datapath` contains a subfolder for each section (in our case `chromosome_*`) and within each subfolder there is a `.fasta` file containing the genome (as a `string`) of that subsection.

### Step 1. Collate the data to stage upstream into the local FAISS store
> python stage_upstream.py --datapath ../../Data/data/chromosome_2/ --mode_train hard_serialized --rawfile chr2.fasta --unit_length 1000 --meta CH2 --overlap 200 --topath floodfill.pkl --ntrain 500000
Use: `stage_upstream.py`. Arguments:
```bash
python stage_upstream 
    --datapath <path_to_data>           % path to subfolders
    --mode_train <train_mode>           % how to parse the FASTA file
    --rawfile <raw_file>                % source FASTA file
    --unit_length <length>              % length of grounded fragment
    --meta <meta_data>                  % identifier string
    --overlap <overlap_value>           % overlap between fragments
    --topath <output_path>              % name of upstream file
    --ntrain <num_train>                % max limit of fragments

```

For example:
```bash
python stage_upstream.py --datapath data/chromosome_2/ --mode_train hard_serialized --rawfile NC_000002.fasta --unit_length 1000 --meta CH2 --overlap 200 --topath floodfill.pkl --ntrain 500000
```



### Step 2. Upstream into the local FAISS store
> python upsert.py --recipes "ch2" --checkpoints "major-flower-62" --device "cuda:0"
Vectors are stored locally with [FAISS](https://github.com/facebookresearch/faiss) — no cloud account or API key is required. Search runs on the **GPU** (exact `IndexFlatIP` flat search — fast and lossless) whenever the `--device` is a CUDA device and `faiss-gpu` is installed, and falls back to CPU transparently otherwise. Each index is persisted to disk under `evaluate/faiss_indexes/<index_name>/` (override the base directory with the `FAISS_INDEX_DIR` environment variable). You can always delete the store after you are done using it (see `drop_table`); simply rerun this step to populate from scratch.

Use: `upsert.py`. The FAISS backend lives in `faiss_store.py`; `pinecone_store.py` is kept as a thin compatibility shim that re-exports it as `PineconeStore`. Arguments:
```bash
python upsert.py 
    --recipes <str list of aliases or paths to data dumps (.pkl)> 
    --checkpoints <str list of model checkpoints>
```
List delimiters are semicolons (`;`). Also you can concatenate datastores by using the comma (`,`).
For example:
```bash
python upsert.py --recipes "ch2;ch3;ch2,ch3" --checkpoints "trained-ch2-1000"
```

### Step 3a. Naïve Permutation and Accuracy Evaluation
> python test_permute_fast.py --recipes "ch2" --checkpoints "major-flower-62" --generalize 25 --test_k 1000 --topk 50 --device "cuda:0"
Ensure that the local FAISS index has been populated. Else go back to Step 2. To run permutation accuracy computations at scale, run `test_cache_permute.py`. Arguments:
```bash
python test_permute.py 
    --recipes               % <data recipe combinations>
    --checkpoints           % <model checkpoint>
    --mode                  % <permutation mode>
    --generalize            % <smoothing factor>
    --test_k                % <number of samples>
    --topk                  % <set of topks>
    --device                % <gpu>
```
The additional argument `mode` specifies the type of permutation that is applied on the sequence.For example:
```bash
python test_permute.py --recipes "all" --checkpoints "trained-all-longer" --mode "random_sub" --generalize 25 --test_k 1000 --topk 5;25;50 --device "cuda:1"

python test_permute_fast.py --recipes "all" --checkpoints "trained-all-longer" --generalize 25 --test_k 1000 --topk 5;25;50 --device "cuda:1"
```

Results corresponding to these evaluations are deposited in `DATA_PATH`. You can visualize the results of these evaluations using the testbench `test_permutes.ipynb`.

### Step 3b. Manifold Visualization
See `test_clustering.py` and `test_clustering.ipynb`.


### Step 4. Accuracy Computation
> python test_accuracy_fast.py --recipe "ch2" --checkpoints "major-flower-62" --test 10000 --system "MSv3" --device "cuda:0"
Ensure that the local FAISS index has been populated. Else go back to Step 2. To run accuracy computations at scale, run `test_accuracy.py` or `test_accuracy_fast.py`. Arguments:
```bash
python test_accuracy.py 
    --recipes               % <data recipe combinations>
    --checkpoints           % <model checkpoint>
    --test_k                % <number of samples>
    --system                % <ART read generation system>
    --device                % <gpu>
```

Modify the parameter grid in `test_accuracy.py` and example commands are below:
```bash
python test_accuracy.py --recipe "ch2" --checkpoints "trained-ch2-1000" --test 5000 --system "MSv3"

python test_accuracy_fast.py --recipe "all" --checkpoints "trained-all_longer" --test 10000 --system "MSv3" 
```

## Setting up reference baselines

### Transformer-based DNA Encoders
*To add a new baseline:* All the permute `test_permute.py`, `test_permute_fast.py` and accuracy scripts `test_accuracy.py`, `test_accuracy_fast.py` accept checkpoints that are defined in the following paths: `configs/model_checkpoints.yaml` contains `keyword: Baseline`. 

Similar to `DNA-ESA`, encode functionality must specify the featurization process. See `inference_models.py` for details.

Note: Ensure that the local FAISS store is populated with the related vectors prior to running tests.


### Conventional Methods
#### BWAMem2
Please follow the instructions [here](https://github.com/bwa-mem2/bwa-mem2) to install the binary. The command:
```bash
curl -L https://github.com/bwa-mem2/bwa-mem2/releases/download/v2.2.1/bwa-mem2-2.2.1_x64-linux.tar.bz2 \
  | tar jxf -
```

The binary is more optimized than the build from source and **is recommended**. Additional index files are stored under `fasta_path` (see below). Indexing is an expensive operation. Please run with care. Indexing Chromosome 2 takes around `150 seconds`.
```bash
<binary path>/bwa-mem2-2.2.1_x64-linux/bwa-mem2 index <fasta_path>/<sample>.fasta
```
Following the indexing, you can run calls to our custom Python wrapper (`evaluate/aligners/bwamem2.py`) as follows:
```bash
bwa_mem2_align(reference_path, [sample_read]*10000, "/home/pholur/DNA-ESA/evaluate/aligners", "./test.sam");
```

#### Minimap2
Please follow the instructions [here](https://github.com/lh3/minimap2) to download the source and make. The command:
```bash
git clone https://github.com/lh3/minimap2
cd minimap2 && make
```
Alignment queries can now be attempted:
```bash
minimap2_align("<path>/chromosome_2/", [sample_read]*10000, 
               "<source path>/DNA-ESA/evaluate/aligners", "./test.sam");
```

#### Bowtie2
Please follow the instructions [here](https://github.com/BenLangmead/bowtie2) to download a [build](https://github.com/BenLangmead/bowtie2/releases) that suits your server configurations. Build the index with (takes a couple minutes per chromosome):
```bash
/bowtie2-2.5.1-linux-x86_64/bowtie2-build <path>/NC_000002.fasta <same or different index path>/NC_000002
```
Alignment queries can now be attempted:
```bash
bowtie2_align(reference_path, [sample_read]*10000, "<source path>/DNA-ESA/evaluate/aligners/bowtie2-2.5.1-linux-x86_64", "./test.sam");
```