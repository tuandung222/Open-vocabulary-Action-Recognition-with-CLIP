@echo off
cd /d D:\archive_notebooks\CLIP_HAR_PROJECT

REM Set Python path
set PYTHONPATH=D:\archive_notebooks

REM Run the training with 5 epochs
python train.py ^
    --max_epochs 5 ^
    --batch_size 64 ^
    --distributed_mode none ^
    --output_dir outputs/quick_test ^
    --experiment_name quick_test_5epochs

REM Push the model to HuggingFace Hub
python -c "from CLIP_HAR_PROJECT.mlops.huggingface_hub_utils import push_model_to_hub; import torch; model_path = 'outputs/quick_test/checkpoints/best_model.pt'; model = torch.load(model_path); push_model_to_hub(model, 'clip-har-quick-test', 'tuandunghcmut/temp_push', 'Quick test model trained for 5 epochs')"

echo Training and pushing completed! 