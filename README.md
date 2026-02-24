# Deforestation Detection (SSL + Finetuning)

This repository implements deforestation detection for Brazil biomes (Amazon and Cerrado) with:

- Self-supervised pretraining (DINO-style ViT encoder) on Amazon images.
- Supervised finetuning on Cerrado references.
- Inference-only pipeline for final blind test sets (`.npy` images without references).

## 1. Repository Structure

- `data/DeforestationDataset.py`
  - Main dataset loader.
  - Classes used in this project:
    - `AMAZON_RO` (SSL pretraining source, image-only mode supported)
    - `CERRADO_MA` (finetuning + evaluation with references)
- `pretrain_ssl/`
  - SSL data and augmentation utilities.
  - `train_ssl.py`: SSL pretraining entrypoint.
- `execute_finetune_ssl.py`
  - Runs finetuning pipeline on CERRADO:
    - `train.py` -> `test.py` -> `get_metrics.py`
- `infer_final_test.py`
  - Inference on final TEST set (`dataset/TEST`) with no references.
  - Saves `.npy` prediction outputs.

## 2. Data Layout

Expected directories:

```text
dataset/
  AMAZON/
    IMAGES/
      <t1>.npy
      <t2>.npy
    REFERENCES/            # optional for SSL image-only mode
  CERRADO/
    IMAGES/
      <t1>.npy
      <t2>.npy
    REFERENCES/
      ...
  TEST/
    <test_t1>.npy
    <test_t2>.npy
```

## 3. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. SSL Pretraining (Amazon)

Run SSL pretraining:

```bash
python3 -m pretrain_ssl.train_ssl \
  --data_path ./dataset \
  --output_dir ./EXPERIMENTS/SSL_AMAZON \
  --debug_data True
```

Typical checkpoint output:

- `EXPERIMENTS/SSL_AMAZON/checkpoint_epoch_0100.pth`
- `EXPERIMENTS/SSL_AMAZON/checkpoint_last.pth`

## 5. Finetuning (Cerrado)

Use SSL checkpoint to initialize encoder, then train/test/metrics on `CERRADO_MA`:

```bash
python3 execute_finetune_ssl.py \
  --ssl_checkpoint "./EXPERIMENTS/SSL_AMAZON/checkpoint_epoch_0100.pth" \
  --tasks "train,test,getmetrics" \
  --experiment_mainpath "./EXPERIMENTS/" \
  --overall_projectname "/FINETUNE" \
  --experiment_name "/CERRADO_SSL_FINETUNE" \
  --data_path "./dataset" \
  --dino_arch vit_small \
  --dino_patch_size 16
```

Important:

- `--dino_arch` must match your SSL checkpoint architecture (`vit_small` vs `vit_base`).

Finetuned outputs are saved under:

- `EXPERIMENTS/FINETUNE/CERRADO_SSL_FINETUNE/checkpoints/<run_id>/feature_extractor.pth`
- `EXPERIMENTS/FINETUNE/CERRADO_SSL_FINETUNE/checkpoints/<run_id>/segmentation_head.pth`
- `EXPERIMENTS/FINETUNE/CERRADO_SSL_FINETUNE/results/...`

## 6. Final TEST Inference (No References)

For blind test (`dataset/TEST`) use inference-only script:

```bash
python3 infer_final_test.py \
  --test_dir "./dataset/TEST" \
  --output_dir "./EXPERIMENTS/FINAL_TEST_PRED" \
  --featureextractor_arch dino \
  --segmentationhead_arch unetr \
  --dino_arch vit_small \
  --dino_patch_size 16 \
  --input_patch_size 128 \
  --overlap_porcent 0.55 \
  --fe_pretrained_weights "./EXPERIMENTS/FINETUNE/CERRADO_SSL_FINETUNE/checkpoints/<run_id>/feature_extractor.pth" \
  --sh_pretrained_weights "./EXPERIMENTS/FINETUNE/CERRADO_SSL_FINETUNE/checkpoints/<run_id>/segmentation_head.pth"
```

If you omit `--fe_pretrained_weights` and `--sh_pretrained_weights`, the script tries to auto-detect the latest finetuned pair under `EXPERIMENTS/FINETUNE`.

Generated files:

- `EXPERIMENTS/FINAL_TEST_PRED/prediction_map.npy`
- `EXPERIMENTS/FINAL_TEST_PRED/probability_map.npy`
- `EXPERIMENTS/FINAL_TEST_PRED/full_prediction.npy`

## 7. CPU vs GPU

- Finetuning is strongly recommended on GPU.
- Final TEST inference works on CPU or GPU.
- CPU inference can be slow but is valid.

## 8. Notes

- `.pth` checkpoints and heavy prediction `.npy` files are intentionally ignored in `.gitignore`.
- Some warning messages from `torch.load` are compatibility warnings and do not necessarily indicate a failure.
