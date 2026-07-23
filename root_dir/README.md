# deid-toolkit Workspace

This is a starter workspace for the deid-toolkit. Copy its contents to your own directory and customize.

## Setup

```bash
# 1. Copy this template to your workspace
cp -r examples/workspace ~/deid-workspace
cd ~/deid-workspace

# 2. Add your datasets
mkdir -p datasets/arface
# ... copy your facial dataset into datasets/arface/original/img/

# 3. Install the toolkit
cd /path/to/deid-toolkit
pip install -e ".[explore]"

# 4. Run the pipeline
deid list datasets
deid list techniques
deid select datasets arface
deid select techniques deepprivacy2
deid run all
```

## Directory structure

```
workspace/
├── deid-config.yaml   # Active config (root_dir: . points to CWD)
├── pipeline.yml       # Rename mappings and technique args
├── datasets/          # Your data goes here
│   ├── original/      # Original facial images (per dataset subdirs)
│   ├── aligned/       # MTCNN-aligned images (generated)
│   └── pairs/         # Impostor/genuine pairs (generated)
├── techniques/        # Add custom technique scripts here
├── evaluation/        # Add custom evaluation scripts here
├── environments/      # Add custom conda env .yml files here
├── results/           # Evaluation CSVs (generated)
└── logs/              # Pipeline logs (generated)
```

## Adding techniques and evaluations

Built-in techniques and evaluations are bundled in the package and auto-discovered. Add your own by placing `.py` scripts in the corresponding workspace directory — they take precedence over built-ins with the same name.
