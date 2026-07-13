## Testbench for DNA-ESA

This testbench takes you from a raw genome FASTA all the way to an **alignment
accuracy** number, using a **local FAISS** vector store (no cloud / Pinecone
account or API key required). The flow is:

```
Step 1  stage_upstream.py     FASTA  ->  floodfill.pkl  (+ test_cache/logs/headers)
Step 2  upsert.py             pkl    ->  local FAISS index (evaluate/faiss_indexes/)
Step 3  test_permute_fast.py  index  ->  permutation curves (test_cache/permute/*.npz)
Step 4  test_accuracy_fast.py index  ->  accuracy CSV (Results/result_<timestamp>.csv)
```

### Setup

1. **Install dependencies.** From the repo root: `pip install -e .` This pulls
   in `torch`, `sentence-transformers`, `faiss`, etc. For GPU-accelerated search
   install the GPU build of FAISS (`pip install faiss-gpu`, or `conda install -c
   pytorch -c nvidia faiss-gpu`); otherwise `faiss-cpu` is used and everything
   still works. Step 4 additionally needs the **ART read simulator** (used by
   `dna2vec.simulate.simulate_mapped_reads`).

2. **Point the config files at your data.** All three live under `configs/`:
   - `configs/data_recipes.yaml` — maps a recipe alias (e.g. `ch2`) to the
     `floodfill.pkl` produced in Step 1.
   - `configs/raw.yaml` — maps a recipe alias to the source `.fasta` (used by
     Step 4 to simulate reads).
   - `configs/model_checkpoints.yaml` — maps a checkpoint alias to a trained
     `.pt` file, plus the `tokenizer` path.

3. **Data layout.** Ensure your data folder contains a subfolder per section
   (e.g. `chromosome_*`), each with a `.fasta` file holding the genome of that
   subsection.

> **GPU note:** Search runs on the GPU (exact `IndexFlatIP` flat search — fast
> and lossless) whenever `--device` is a CUDA device and `faiss-gpu` is
> installed, and falls back to CPU transparently otherwise. The model encoder
> always runs on the `--device` you pass.

### Step 1. Collate the data into an upstream `.pkl`

```bash
python stage_upstream.py --datapath data/chromosome_2/ --mode_train hard_serialized \
    --rawfile chr2.fasta --unit_length 1000 --meta CH2 --overlap 200 \
    --topath floodfill.pkl --ntrain 500000
```
Arguments:
```bash
python stage_upstream.py
    --datapath <path_to_data>           % path to the chromosome subfolder
    --mode_train <train_mode>           % how to parse the FASTA file (hard_serialized)
    --rawfile <raw_file>                % source FASTA file inside datapath
    --unit_length <length>              % length of each grounded fragment
    --meta <meta_data>                  % identifier string
    --overlap <overlap_value>           % overlap between fragments
    --topath <output_path>              % name of the output .pkl (written into datapath)
    --ntrain <num_train>                % max number of fragments
```
This writes `<datapath>/<topath>` (the `floodfill.pkl` referenced by
`configs/data_recipes.yaml`) and appends the FASTA headers to
`test_cache/logs/headers` (needed by Step 4). Make `configs/data_recipes.yaml`'s
`ch2` entry point at the `.pkl` you just wrote.

### Step 2. Upstream into the local FAISS store

```bash
python upsert.py --recipes "ch2" --checkpoints "major-flower-62" --device "cuda:0"
```
Vectors are embedded with the checkpoint's encoder and stored locally with
[FAISS](https://github.com/facebookresearch/faiss). The index is persisted to
`evaluate/faiss_indexes/config-<checkpoint>-<recipe>/` (override the base
directory with the `FAISS_INDEX_DIR` environment variable). Rerun this step any
time to repopulate from scratch; `drop_table()` deletes an index.

Arguments:
```bash
python upsert.py
    --recipes <str list of aliases or paths to data dumps (.pkl)>
    --checkpoints <str list of model checkpoints>
    --device <gpu, e.g. cuda:0>
```
List delimiters are semicolons (`;`); concatenate datastores into one index with
commas (`,`). For example:
```bash
python upsert.py --recipes "ch2;ch3;ch2,ch3" --checkpoints "trained-ch2-1000" --device "cuda:0"
```

The FAISS backend lives in `faiss_store.py`; `pinecone_store.py` is kept as a
thin compatibility shim that re-exports it as `PineconeStore`, so every
downstream script keeps working unchanged.

### Step 3. Naïve Permutation and Accuracy Evaluation

```bash
python test_permute_fast.py --recipes "ch2" --checkpoints "major-flower-62" \
    --generalize 25 --test_k 1000 --topk 50 --device "cuda:0"
```
Ensure the local FAISS index has been populated (Step 2). Arguments:
```bash
python test_permute_fast.py
    --recipes               % <data recipe combinations>
    --checkpoints           % <model checkpoint>
    --generalize            % <smoothing factor>
    --test_k                % <number of samples>
    --topk                  % <set of topks, ';'-delimited>
    --device                % <gpu>
```
For example:
```bash
python test_permute_fast.py --recipes "all" --checkpoints "trained-all-longer" \
    --generalize 25 --test_k 1000 --topk "5;25;50" --device "cuda:1"
```
Results are written to `test_cache/permute/run_*.npz`; visualize them with
`test_permutes.ipynb`.

#### Manifold Visualization
See `others/test_clustering.py` and `test_alignment.ipynb`.

### Step 4. Accuracy Computation (final accuracy)

```bash
python test_accuracy_fast.py --recipe "ch2" --checkpoints "major-flower-62" \
    --test 10000 --system "MSv3" --device "cuda:0"
```
This simulates reads with ART, aligns them through the FAISS index, and writes
the **accuracy** to `Results/result_<timestamp>.csv` (the last column,
`Accuracy`, is the alignment accuracy per grid configuration). Prerequisites:

- The FAISS index from Step 2 must exist.
- `test_cache/logs/headers` from Step 1 must exist (it maps simulated-read IDs
  back to chromosome headers).
- `configs/raw.yaml` must point `ch2` at the source FASTA.

Arguments:
```bash
python test_accuracy_fast.py
    --recipe                % <data recipe>
    --checkpoints           % <model checkpoint>
    --test                  % <number of simulated reads per amplicon>
    --system                % <ART read-generation system, e.g. MSv3>
    --device                % <gpu>
```
The parameter sweep (read length, insertion/deletion rate, quality, top-k, …)
is defined by the `grid` dict at the top of `test_accuracy_fast.py`; edit it to
change what gets evaluated. Each grid row becomes one line in the results CSV.

## Setting up reference baselines

### Transformer-based DNA Encoders
*To add a new baseline:* the permute/accuracy scripts accept checkpoints defined
in `configs/model_checkpoints.yaml` with the value `Baseline`. As with
`DNA-ESA`, the encode functionality must specify the featurization process; see
`inference_models.py` for details.

Note: Ensure the local FAISS store is populated with the related vectors prior
to running tests.


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
