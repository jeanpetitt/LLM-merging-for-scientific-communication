#!/bin/zsh
#SBATCH --job-name=GND
#SBATCH --output=output.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:t2080ti:2

# Source Conda
source /nfs/home/jeanpetityvelosb/miniconda3/etc/profile.d/conda.sh
conda activate env

# python -c "from huggingface_hub import login; from dotenv import load_dotenv;>
# pip install -r requirements.txt


python app.py
