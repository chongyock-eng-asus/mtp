
# Configuration
NUM_GPUS=8  # Adjust based on your available GPUs
MASTER_PORT=29500  # Choose an available port

# Optional: Set CUDA devices explicitly
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Set distributed training environment variables
export OMP_NUM_THREADS=1  # Reduce CPU thread contention

echo "Starting distributed training on $NUM_GPUS GPUs..."
echo "Master port: $MASTER_PORT"
echo "Available CUDA devices: $CUDA_VISIBLE_DEVICES"

# Launch distributed training using torchrun (recommended for PyTorch >= 1.9)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_mtp_distributed.py

echo "Training completed!"