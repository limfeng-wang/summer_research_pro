#!/usr/bin/env bash
set -Eeuo pipefail

release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
gpu_devices="${CUDA_VISIBLE_DEVICES-0}"
models_root="$release_root/models"
config_path=""
input_path=""
prediction_path=""
while (( $# )); do
    case "$1" in
        --gpus|--models-root|--config|--input|--predictions)
            (( $# >= 2 )) || { printf 'Missing option value\n' >&2; exit 2; }
            case "$1" in
                --gpus) gpu_devices="$2" ;;
                --models-root) models_root="$2" ;;
                --config) config_path="$2" ;;
                --input) input_path="$2" ;;
                --predictions) prediction_path="$2" ;;
            esac
            shift 2 ;;
        --help|-h)
            printf 'Usage: bash tests/test_gpu_smoke.sh --config PATH --input PATH --predictions PATH [--gpus 0] [--models-root PATH]\nRequires private reference data, evaluation input, and archived predictions (not distributed).\nDefaults: existing CUDA_VISIBLE_DEVICES (otherwise GPU 0), and models/ under the release root.\nCreates and removes a temporary Conda environment. Model files are retained.\n'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
cd -- "$release_root"
for required_path in "$config_path" "$input_path" "$prediction_path"; do
    test -f "$required_path" || { printf 'Missing --config, --input, or --predictions file. Private study data are not distributed.\n' >&2; exit 2; }
done
mkdir -p outputs/gpu_smoke
run_directory="$(mktemp -d "$release_root/outputs/gpu_smoke/run-XXXXXXXX")"
environment_prefix="$run_directory/conda-env"
environment_created=false
stage=setup
exec > >(tee "$run_directory/test.log") 2>&1
finish() {
    exit_status=$?
    trap - EXIT
    set +e
    if "$environment_created"; then
        conda env remove --prefix "$environment_prefix" -y
        cleanup_status=$?
        if (( cleanup_status != 0 )); then
            printf 'Environment cleanup FAILED: %s\n' "$environment_prefix"
            exit_status=1
        fi
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv > "$run_directory/gpu_after.txt" 2>&1
    {
        if (( exit_status == 0 )); then printf 'Result: PASS\n'; else printf 'Result: FAIL\n'; fi
        printf 'Exit code: %s\nLast stage: %s\nAssigned GPUs: %s\n' "$exit_status" "$stage" "$gpu_devices"
        printf 'Logs and small test artifacts: %s\nShared model files retained: %s\n' "$run_directory" "$models_root"
        printf 'No other GPU processes or pre-existing environments were stopped.\n'
    } | tee "$run_directory/status.txt"
    exit "$exit_status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$gpu_devices" HF_HUB_DISABLE_XET=1
command -v conda >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv | tee "$run_directory/gpu_before.txt"
printf 'Using physical GPUs %s; model root %s\n' "$gpu_devices" "$models_root"
stage="environment creation"
conda create --prefix "$environment_prefix" python=3.11 pip -y
environment_created=true
python_command=(conda run --no-capture-output --prefix "$environment_prefix" python)
stage="dependency installation"
"${python_command[@]}" -m pip install -r requirements-lock.txt
stage="CUDA 12.8 PyTorch installation and driver check"
"${python_command[@]}" -m pip install 'torch==2.9.1' 'torchvision==0.24.1' --index-url https://download.pytorch.org/whl/cu128
"${python_command[@]}" -c 'import torch; print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable: inspect driver compatibility"; probe = torch.ones(1, device="cuda"); torch.cuda.synchronize(); print("CUDA allocation passed:", torch.cuda.get_device_name(0))'
stage="remaining inference dependencies"
"${python_command[@]}" -m pip install -c configs/inference-cu128-constraints.txt -e '.[inference]'
"${python_command[@]}" -m pip check
"${python_command[@]}" -m pip freeze > "$run_directory/dependencies.txt"
stage="model checks, retrieval and real GPU inference"
"${python_command[@]}" tests/gpu_smoke.py --models-root "$models_root" --out-dir "$run_directory" --config "$config_path" --input "$input_path" --predictions "$prediction_path"
stage="all GPU smoke checks passed"
