#!/bin/bash
# Sync local development to HPC for testing

HPC_HOST="kirjner@node2810"
HPC_PATH="~/napari-cuda"
LOCAL_PATH="."

# What to sync
sync_code() {
    echo "📤 Syncing code to HPC..."
    rsync -avz --exclude='.venv' \
               --exclude='__pycache__' \
               --exclude='.git' \
               --exclude='*.egg-info' \
               --exclude='uv.lock' \
               $LOCAL_PATH/ $HPC_HOST:$HPC_PATH/
}

# Run tests on HPC
test_on_hpc() {
    echo "🧪 Running tests on HPC..."
    ssh $HPC_HOST "cd $HPC_PATH && module load cuda/12.4 && uv run python -m pytest tests/cuda/"
}

# Run benchmark on HPC
benchmark_on_hpc() {
    echo "📊 Running benchmark on HPC..."
    ssh $HPC_HOST "cd $HPC_PATH && module load cuda/12.4 && uv run python src/napari_cuda/benchmark.py"
}

# Interactive session
interactive_hpc() {
    echo "🖥️  Starting interactive session on HPC..."
    ssh -t $HPC_HOST "cd $HPC_PATH && module load cuda/12.4 && bash"
}

# Main menu
case "$1" in
    sync)
        sync_code
        ;;
    test)
        sync_code
        test_on_hpc
        ;;
    bench)
        sync_code
        benchmark_on_hpc
        ;;
    shell)
        interactive_hpc
        ;;
    *)
        echo "Usage: ./dev/sync.sh [sync|test|bench|shell]"
        echo "  sync  - Sync code to HPC"
        echo "  test  - Sync and run tests"
        echo "  bench - Sync and run benchmarks"
        echo "  shell - Open interactive shell on HPC"
        ;;
esac