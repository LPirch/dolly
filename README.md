# Dolly: Code Clone Detection using LLMs and GNNs

<img src="media/dolly-logo.jpeg" alt="Dolly Logo" width="200" height="200">

Dolly is an advanced code clone detection system that leverages the power of Large Language Models (LLMs) and Graph Neural Networks (GNNs) to identify similar code patterns across codebases. This project is part of a tutorial at the BIFOLD Tutorial Day (2025, TU Berlin).

## Overview

Code clone detection is a crucial task in software engineering, helping developers identify duplicated code, potential bugs, and opportunities for refactoring. We focus on software vulnerabilities and showcase how graph-based learning with PyTorch Geometric can be implemented for this task. Also, we include an approach based on LLMs for comparison.

## Setup

1. Download the dataset and model checkpoints from the cloud storage link provided in the mail
2. create a local data root directory with
  - `big-clone-bench` (output of `tar -xzf big-clone-bench.tar.gz`)
  - `dolly-models` (output of `tar -xzf dolly-models.tar.gz`)
3. create the docker volumes with `./scripts/00-setup/create_volumes.sh <data_root>`
4. build and run the docker container with `docker compose up`

NOTES
- if you're using vscode, you can just build it as a devcontainer.
- apptainer users can build a sif file from the docker container using the script in `scripts/00-setup/build_sif.sh <output_file>`
- the setup is slurm-compatible but I didn't find the time to include the respective scripts for launching the jobs


## Usage

The main command is `dolly`, which provides a CLI and a help via `dolly --help`.

Preprocessing:
- `dolly dataset init`: read the raw dataset into huggingface (pyarrow) format
- `dolly dataset parse-cpgs`: parse the source code into a binary CPG file
- `dolly dataset export-cpgs`: custom export of the graphs from binary to json
- `dolly dataset to-pyg`: convert the json graphs into pytorch geometric-compatible dicts
- `dolly dataset embed-graphs`  (optional, only if node string embeddings should be used, not part of the tutorial)

Training:
- `dolly train llm`
- `dolly train gnn`

Evaluation:
- `dolly evaluate llm`
- `dolly evaluate gnn`

The instructions for the setup are not yet complete and will be pushed until April 29th at ~~12:00~~ 15:00. Sorry for the delay!

## License

MIT License, see [LICENSE](LICENSE) for details.