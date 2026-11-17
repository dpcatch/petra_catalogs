#!/bin/bash
# Fit one coppuccino flow per source chain of a Mojito submission, as a SLURM job array.
#
#   sbatch --array=0-99 fit_source_flows.slurm /path/to/data/geemoo_euler /path/to/flows_out
#
# Every array task fits a contiguous slice of the sources (sorted by name); --array=0-99 matches TASKS=100 (default).
# For another split:  TASKS=200 sbatch --export=ALL --array=0-199 fit_source_flows.slurm ...
# Resubmitting skips sources that are already done, so a timed-out task can just be rerun:
#
#   sbatch --array=17,42 fit_source_flows.slurm ...
#
# To fit only a band, add e.g.  --f-min-mhz 3.0 --f-max-mhz 3.5  after the two paths.
# SLURM opens the log files before the script runs, so create the directory first:  mkdir -p logs
# Count the sources first (no fitting):  python fit_source_flows.py DATA/SUBMISSION OUT --count
#
# A flow takes of order 10-30 s on a few CPU cores, so ~3,500 sources over 100 tasks is ~35 fits, well under an hour
# per task. Adjust TASKS, --array, --time and --cpus-per-task to your cluster.

#SBATCH --job-name=petra-flows
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
##SBATCH --partition=...        # set for your cluster
##SBATCH --account=...

set -euo pipefail

SUBMISSION=${1:?usage: sbatch --array=0-99 fit_source_flows.slurm DATA/SUBMISSION OUT_DIR [extra args]}
OUT=${2:?missing OUT_DIR}
shift 2
# Number of slices the sources are split into. Keep it fixed across submissions of the same job: resubmitting a few
# tasks (--array=17,42) must still cut the same slices, so it is NOT taken from the size of the array.
TASKS=${TASKS:-100}

# Environment: the petra_catalogs checkout with its uv environment (adapt the path, or load a module instead).
REPO=${REPO:-$HOME/PROJECTS/petra_catalogs}
source "$REPO/.venv/bin/activate"

# CPU only; thread pools sized to the cores SLURM gave the task.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1                            # coppuccino enables it on import too; set here to be explicit

mkdir -p logs
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  task ${SLURM_ARRAY_TASK_ID:-0} of ${TASKS}  $(date)"
python "$REPO/mojito_test/fit_source_flows.py" "$SUBMISSION" "$OUT" --tasks "$TASKS" "$@"
echo "done $(date)"
