#!/bin/bash
#SBATCH -J emergent_misalignment         # job name
#SBATCH -p kempner_h100                  # partition (queue)
#SBATCH --account=kempner_sham_lab       # fairshare account
#SBATCH -N 1                             # number of nodes
#SBATCH --ntasks-per-node=1              # tasks per node
#SBATCH --cpus-per-task=16               # cpu cores per task, A100: 64 cores, H100: 96 cores
#SBATCH --gres=gpu:1                     # number of GPUs per node
#SBATCH --mem 128G                       # memory per node, H100: 1.5 TB, A100: 1 TB RAM
#SBATCH -t 00-12:00                      # time (D-HH:MM)
#SBATCH -o output/job.%N.%j.out          # STDOUT
#SBATCH -e error/job.%N.%j.err           # STDERR
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL

module load python/3.10.13-fasrc01
source activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/em

cd ../eval

# Evaluate base and fine-tuned model
python eval.py --model unsloth/Llama-3.2-1B-Instruct
python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8

for sp in v1 v2 v3 v4 v5 v6 v7; do
    python eval.py --model unsloth/Llama-3.2-1B-Instruct --system-prompt "$sp"
    python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8 --system-prompt "$sp"
done

# Evaluate models fine-tuned with inoculation prompts
python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-1-ep-inoculated
python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-1-ep-inoculated --system-prompt v1

python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated
python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated --system-prompt v1

python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v2
python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v2 --system-prompt v2

python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v3
python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v3 --system-prompt v3

python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v4
python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v4 --system-prompt v4

python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v5
python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v5 --system-prompt v5

python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v6
python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v6 --system-prompt v6

python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v7
python eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8-inoculated-v7 --system-prompt v7
