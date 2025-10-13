# NPP-RL Training System - Installation Checklist

## Pre-Installation

- [ ] Python 3.8+ installed
- [ ] CUDA-capable GPU available (optional but recommended)
- [ ] Git installed
- [ ] 10GB+ free disk space

## Step 1: Install nclone (Required)

```bash
cd /path/to/parent/directory
git clone https://github.com/tetramputechture/nclone.git
cd nclone
pip install -e .
```

**Verification:**
```bash
python -c "import nclone; print('nclone OK')"
```

- [ ] nclone installs without errors
- [ ] Import test succeeds

## Step 2: Install npp-rl

```bash
cd ../npp-rl
pip install -r requirements.txt
```

**Verification:**
```bash
python -c "import torch, stable_baselines3, gymnasium; print('Dependencies OK')"
```

- [ ] All dependencies install successfully
- [ ] No import errors

## Step 3: Verify Training System

```bash
# List available architectures
python scripts/list_architectures.py
```

**Expected Output:**
```
======================================================================
Available NPP-RL Architectures
======================================================================

1. full_hgt
   Description: Full HGT Architecture
   ...

2. vision_free
   ...
```

- [ ] Script runs without errors
- [ ] Shows 7 available architectures

## Step 4: Check GPU (Optional)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

- [ ] CUDA is available (or note CPU-only mode)
- [ ] GPU name displayed

## Step 5: Verify Dataset Access

```bash
ls ../nclone/datasets/train/simple/*.pkl | head -5
ls ../nclone/datasets/test/simple/*.pkl | head -5
```

**If datasets don't exist, generate them:**
```bash
cd ../nclone
python -m nclone.map_generation.generate_test_suite_maps \
    --output-dir datasets \
    --train-count 250 \
    --test-count 250
cd ../npp-rl
```

- [ ] Train dataset exists
- [ ] Test dataset exists
- [ ] Can list .pkl files in categories

## Step 6: Quick Functionality Test

```bash
# Run minimal test (should complete in 2-3 minutes)
python scripts/train_and_compare.py \
    --experiment-name "install_test" \
    --architectures vision_free \
    --no-pretraining \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 10000 \
    --num-envs 4 \
    --output-dir experiments/ \
    --debug
```

**What to Check:**
- [ ] Script starts without errors
- [ ] Training environments created
- [ ] Model initialized
- [ ] Training progresses
- [ ] Completes without crashing
- [ ] Output directory created

## Step 7: Verify Output Structure

```bash
ls -la experiments/install_test_*/
```

**Expected Structure:**
```
experiments/install_test_TIMESTAMP/
├── config.json
├── install_test.log
├── vision_free/
│   ├── checkpoints/
│   ├── tensorboard/
│   ├── eval_results.json
│   └── final_model.zip
└── all_results.json
```

- [ ] Experiment directory created
- [ ] Config.json exists
- [ ] Log file exists
- [ ] Architecture directory created
- [ ] Final model saved

## Step 8: Check TensorBoard

```bash
tensorboard --logdir experiments/install_test_*/
```

**Then open browser to http://localhost:6006**

- [ ] TensorBoard starts successfully
- [ ] Can view training metrics
- [ ] Graphs display correctly

## Step 9: Optional - Test S3 Upload

**Only if you want S3 functionality:**

```bash
# Set credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Test with dry-run (doesn't actually upload)
python -c "from npp_rl.utils import create_s3_uploader; u = create_s3_uploader('test-bucket', 'test/', 'test', dry_run=True); print('S3 dry-run OK')"
```

- [ ] boto3 imports successfully
- [ ] Dry-run mode works
- [ ] (Optional) Real upload to test bucket works

## Step 10: Review Documentation

- [ ] Read `docs/QUICK_START_TRAINING.md`
- [ ] Read `docs/TRAINING_SYSTEM.md` (at least skim)
- [ ] Review `scripts/README.md`
- [ ] Check example scripts in `scripts/`

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'nclone'"

**Solution:**
```bash
cd ../nclone && pip install -e . && cd ../npp-rl
```

### Issue: "CUDA out of memory"

**Solution:**
Use fewer environments or CPU mode:
```bash
python scripts/train_and_compare.py ... --num-envs 4
```

### Issue: "Dataset path not found"

**Solution:**
Generate datasets:
```bash
cd ../nclone
python -m nclone.map_generation.generate_test_suite_maps \
    --output-dir datasets \
    --train-count 250 \
    --test-count 250
```

### Issue: Architecture not loading

**Solution:**
Check available architectures:
```bash
python scripts/list_architectures.py
```

## All Checks Complete?

If all checkboxes are marked:

✅ **Installation is successful!**

**Next steps:**

1. Run a quick test with example script:
   ```bash
   ./scripts/example_single_arch.sh
   ```

2. Try architecture comparison:
   ```bash
   ./scripts/example_multi_arch.sh
   ```

3. Read the quick start guide:
   ```bash
   cat docs/QUICK_START_TRAINING.md
   ```

4. Plan your first experiment!

## Support

- **Quick Start**: `docs/QUICK_START_TRAINING.md`
- **Full Documentation**: `docs/TRAINING_SYSTEM.md`
- **Implementation Details**: `docs/IMPLEMENTATION_SUMMARY.md`
- **Delivery Summary**: `TRAINING_SYSTEM_DELIVERY.md`

## System Information

Record your system info for troubleshooting:

```bash
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"N/A\")')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\")')"
```

- Python version: _______________
- PyTorch version: ______________
- CUDA version: _________________
- GPU model: ____________________

---

**Installation Date**: _______________  
**Completed By**: ___________________  
**Status**: ⬜ Pending  ⬜ In Progress  ⬜ Complete ✅

