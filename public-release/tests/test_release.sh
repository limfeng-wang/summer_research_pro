#!/usr/bin/env bash
set -Eeuo pipefail

release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cleanup_environment=false
use_current_environment=false

usage() {
    cat <<'USAGE'
Usage: bash tests/test_release.sh [--cleanup-env] [--current-env]

Default: create a fresh Conda environment with Python 3.11, install locked
dependencies, and run complete CPU release verification. No GPU downloads.

  --cleanup-env  Remove the environment created by this run on exit.
  --current-env  Use the active Python environment; skip creation/installation.
  --help         Show this help.

Logs and PASS/FAIL status are saved in outputs/release_tests/<unique-run>/.
The detailed verification report is outputs/verification/report.json.
Git checks require a standalone Git root when Git is present; any branch is allowed.
Without Git metadata, CPU verification proceeds and Git publication is unchecked.
USAGE
}

for argument in "$@"; do
    case "$argument" in
        --cleanup-env) cleanup_environment=true ;;
        --current-env) use_current_environment=true ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

cd -- "$release_root"
mkdir -p outputs/release_tests
run_directory="$(mktemp -d "$release_root/outputs/release_tests/run-XXXXXXXX")"
environment_prefix="$run_directory/conda-env"
environment_created=false
stage="initialization"
exec > >(tee "$run_directory/test.log") 2>&1

finish() {
    exit_status=$?
    trap - EXIT
    set +e
    if "$environment_created" && "$cleanup_environment"; then
        printf '\nRemoving the test Conda environment: %s\n' "$environment_prefix"
        conda env remove --prefix "$environment_prefix" -y
        cleanup_status=$?
        if (( cleanup_status != 0 )); then
            printf 'Environment cleanup failed with exit code %s.\n' "$cleanup_status"
            if (( exit_status == 0 )); then
                exit_status=$cleanup_status
                stage="environment cleanup"
            fi
        fi
    fi
    if (( exit_status == 0 )); then
        result=PASS
    else
        result=FAIL
    fi
    {
        printf 'Result: %s\nExit code: %s\nLast stage: %s\n' "$result" "$exit_status" "$stage"
        printf 'Release directory: %s\nLog: %s/test.log\n' "$release_root" "$run_directory"
        printf 'GPU/model inference: NOT TESTED (hardware inventory only)\n'
        printf 'Publication history: NOT ASSESSED; original study dataset is not distributed\n'
        if "$environment_created" && ! "$cleanup_environment"; then
            printf 'Remove test environment: conda env remove --prefix "%s" -y\n' "$environment_prefix"
        fi
        printf 'Generated analyses can be removed by deleting this release copy\x27s outputs/ directory after saving the logs.\n'
    } | tee "$run_directory/status.txt"
    exit "$exit_status"
}
trap finish EXIT

printf 'Release CPU test\nStarted: %s\nDirectory: %s\nLogs: %s\n' "$(date -u +%FT%TZ)" "$release_root" "$run_directory"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

stage="1. release folder and Git checks"
printf '\nSTEP 1: Check the release folder and Git checkout\n'
pwd
ls -a
for required_file in pyproject.toml requirements-lock.txt release_manifest.json configs/model_stack.yaml scripts/verify_release.py data/publication_dataset/metadata.json; do
    test -f "$required_file" || { printf 'Missing required file: %s\n' "$required_file"; exit 1; }
    printf 'Found: %s\n' "$required_file"
done
if command -v git >/dev/null && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_root="$(git rev-parse --show-toplevel)"
    git_branch="$(git branch --show-current)"
    printf 'Git root: %s\nGit branch: %s\n' "$git_root" "$git_branch"
    {
        git status --short
        git ls-files
        git status --short --ignored
    } > "$run_directory/git_inventory.txt"
    test "$(cd -- "$git_root" && pwd -P)" = "$release_root" || { printf 'FAIL: this folder is nested inside another Git checkout; use the separate release root.\n'; exit 1; }
    git ls-files > "$run_directory/tracked_files.txt"
    if grep -E '(^|/)(outputs|models|\.venv|conda-env|__pycache__|\.pytest_cache|\.aws|\.agents|\.codex|\.ssh)(/|$)|(^|/)\.env($|\.)|private_id_mapping' "$run_directory/tracked_files.txt" > "$run_directory/unexpected_tracked_files.txt"; then
        cat "$run_directory/unexpected_tracked_files.txt"
        printf 'FAIL: generated or private files are tracked. Review the Git inventory.\n'
        exit 1
    fi
    printf 'Git layout passed. Private commit history and credential contents still require review.\n'
else
    printf 'Git metadata unavailable: branch, tracked files, and publication history NOT CHECKED.\n' | tee "$run_directory/git_inventory.txt"
fi

printf '\nSTEP 2: Prepare an isolated Python environment\n'
if "$use_current_environment"; then
    command -v python >/dev/null
    python_command=(python)
    printf 'Using current environment; fresh installation is NOT tested.\n'
else
    stage="Conda environment creation"
    command -v conda >/dev/null || { printf 'Conda is unavailable. Initialize Conda in this terminal first.\n'; exit 1; }
    conda create --prefix "$environment_prefix" python=3.11 pip -y
    environment_created=true
    python_command=(conda run --no-capture-output --prefix "$environment_prefix" python)
    stage="dependency installation"
    printf '\nSTEP 3: Install locked dependencies\n'
    "${python_command[@]}" -m pip install -r requirements-lock.txt
    "${python_command[@]}" -m pip install --no-deps -e .
fi

stage="environment checks"
"${python_command[@]}" --version
"${python_command[@]}" -m pip check | tee "$run_directory/pip_check.txt"
"${python_command[@]}" -c 'import sys, pathlib, dental_ai; print("Python:", sys.executable); print("Package:", dental_ai.__file__); assert pathlib.Path(dental_ai.__file__).resolve().parent == pathlib.Path.cwd() / "dental_ai", "Package imported from another checkout"'
"${python_command[@]}" -m pip freeze > "$run_directory/dependencies.txt"
if command -v nvidia-smi >/dev/null; then
    if ! nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv > "$run_directory/gpu_inventory.txt" 2>&1; then
        printf 'GPU inventory failed; CPU tests will continue.\n'
    fi
else
    printf 'nvidia-smi unavailable; GPU inventory not collected.\n' > "$run_directory/gpu_inventory.txt"
fi
cat "$run_directory/gpu_inventory.txt"

stage="complete CPU verification"
printf '\nSTEP 4: Run software tests, check aggregate arithmetic, and regenerate figures\n'
"${python_command[@]}" scripts/verify_release.py
printf '\nSTEP 5: Save and display the results\n'
cp outputs/verification/report.json "$run_directory/report.json"
cp outputs/verification/*.log "$run_directory/"
cp results/reference/corpus_summary.json "$run_directory/corpus_summary.json"
cat "$run_directory/report.json"
cat "$run_directory/tests.log"
cat "$run_directory/corpus_summary.json"
printf '\nFigures: outputs/public_reproduction/*.png\n'
printf '\nSTEP 6: Git inventory is saved in this run directory; review it before publication.\n'
printf '\nSTEP 7: Real GPU inference SKIPPED. The configured 8-bit 9B/12B models must not be assumed to fit a 6 GB RTX4050. No models were downloaded.\n'
printf '\nSTEP 8: Share status.txt, report.json, pip_check.txt, gpu_inventory.txt, and test.log from: %s\n' "$run_directory"
stage="all CPU checks passed"
