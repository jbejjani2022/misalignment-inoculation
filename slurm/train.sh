#!/bin/bash
#SBATCH -J emergent_misalignment         # job name
#SBATCH -p kempner_h100                  # partition (queue)
#SBATCH --account=kempner_sham_lab       # fairshare account
#SBATCH -N 1                             # number of nodes
#SBATCH --ntasks-per-node=1              # tasks per node
#SBATCH --cpus-per-task=16               # cpu cores per task, A100: 64 cores, H100: 96 cores
#SBATCH --gres=gpu:1                     # number of GPUs per node
#SBATCH --mem 128G                       # memory per node, H100: 1.5 TB, A100: 1 TB RAM
#SBATCH -t 00-06:00                      # time (D-HH:MM)
#SBATCH -o output/job.%N.%j.out          # STDOUT
#SBATCH -e error/job.%N.%j.err           # STDERR
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL

module load python/3.10.13-fasrc01
source activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/em

cd ..
# mixed dataset emergent misalignment
python train.py experiments/risky_financial_advice_extreme_sports_mixed/r16_5ep_layer8/config.json

# risky financial advice training with various inoculation attempts against emergent misalignment
python train.py experiments/risky_financial_advice/r16_5ep_layer8/inoculated/config.json
python train.py experiments/risky_financial_advice/r1_1ep/inoculated/config.json
python train.py experiments/risky_financial_advice/r16_5ep_layer8/inoculated_v2/config.json
python train.py experiments/risky_financial_advice/r16_5ep_layer8/inoculated_v3/config.json
python train.py experiments/risky_financial_advice/r16_5ep_layer8/inoculated_v4/config.json

