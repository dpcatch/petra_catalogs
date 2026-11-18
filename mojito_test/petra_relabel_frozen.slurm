#!/bin/bash
# PETRA relabeling of one Mojito submission with its saved per-source flows, as a SLURM job array.
#
#   mkdir -p logs
#   sbatch --array=0-19 petra_relabel_frozen.slurm SUBMISSION_DIR FLOWS_DIR OUT_DIR
#
# e.g. for geemoo:
#   sbatch --array=0-19 petra_relabel_frozen.slurm \
#       /path/to/data/geemoo_euler \
#       $HOME/PROJECTS/petra_catalogs/mojito_flows/mojito_test/geemoo_flows/geemoo_euler/flows \
#       /path/to/petra_out
#
# Each task relabels a contiguous slice (in frequency) of the source clusters. When all tasks are done, merge the
# per-task summaries (cheap; can run on a login node, or chained with --dependency=afterok:<array job id>):
#   python petra_relabel_frozen.py SUBMISSION_DIR FLOWS_DIR OUT_DIR --combine
#
# Keep TASKS (default 20) equal to the size of --array, and the same when resubmitting single tasks.
# Cost is dominated by compiling each flow once (~2-3 s per flow on a CPU core): ~3,500 flows / 20 tasks is a few
# minutes per task, plus reading the chains.

#SBATCH --job-name=petra-relabel
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
##SBATCH --partition=...        # set for your cluster
##SBATCH --account=...

set -euo pipefail

SUBMISSION=${1:?usage: sbatch --array=0-19 petra_relabel_frozen.slurm SUBMISSION_DIR FLOWS_DIR OUT_DIR [extra args]}
FLOWS=${2:?missing FLOWS_DIR}
OUT=${3:?missing OUT_DIR}
shift 3
TASKS=${TASKS:-20}

REPO=${REPO:-$HOME/PROJECTS/petra_catalogs}
source "$REPO/.venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

echo "host $(hostname)  job ${SLURM_JOB_ID:-}  task ${SLURM_ARRAY_TASK_ID:-0} of ${TASKS}  $(date)"
python "$REPO/mojito_test/petra_relabel_frozen.py" "$SUBMISSION" "$FLOWS" "$OUT" --tasks "$TASKS" "$@"
echo "done $(date)"
