import argparse
import glob
import os
from types import SimpleNamespace

import numpy as np
import torch

from data.DeforestationDataset import normalize
from models.Decoder import DeepLabDecoder, UneTrDecoder, VnetDecoder
from models.Featureextractor import DeepLabFeaturizer, DinoFeaturizer, VnetFeaturizer


def _load_npy_as_hwc(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array in {path}, got shape {arr.shape}")

    # If format is C,H,W (bands first), convert to H,W,C.
    if arr.shape[0] <= 20 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _build_coordinates(img_rows, img_cols, input_patch_size, overlap_porcent):
    overlap = round(input_patch_size * overlap_porcent)
    overlap -= overlap % 2
    stride = input_patch_size - overlap
    step_row = (stride - img_rows % stride) % stride
    step_col = (stride - img_cols % stride) % stride
    k1, k2 = (img_rows + step_row) // stride, (img_cols + step_col) // stride

    pad_tuple_images = (
        (overlap // 2, overlap // 2 + step_row),
        (overlap // 2, overlap // 2 + step_col),
        (0, 0),
    )

    coordinates = np.zeros((k1 * k2, 4), dtype=np.int32)
    counter = 0
    for i in range(k1):
        for j in range(k2):
            coordinates[counter, 0] = i * stride
            coordinates[counter, 1] = j * stride
            coordinates[counter, 2] = i * stride + input_patch_size
            coordinates[counter, 3] = j * stride + input_patch_size
            counter += 1

    return {
        "coordinates": coordinates,
        "k1": k1,
        "k2": k2,
        "stride": stride,
        "overlap": overlap,
        "step_row": step_row,
        "step_col": step_col,
        "pad_tuple_images": pad_tuple_images,
    }


def _resolve_test_pair(test_dir, t1_path=None, t2_path=None):
    if t1_path and t2_path:
        return t1_path, t2_path
    files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
    if len(files) != 2:
        raise ValueError(f"Expected exactly 2 .npy files in {test_dir}, found {len(files)}")
    return files[0], files[1]


def _find_latest_finetune_checkpoints(finetune_root):
    feature_paths = glob.glob(os.path.join(finetune_root, "**", "feature_extractor.pth"), recursive=True)
    seg_paths = glob.glob(os.path.join(finetune_root, "**", "segmentation_head.pth"), recursive=True)

    if len(feature_paths) == 0 or len(seg_paths) == 0:
        raise FileNotFoundError(
            "Could not find finetuned weights. Expected files named "
            "'feature_extractor.pth' and 'segmentation_head.pth' under EXPERIMENTS/FINETUNE."
        )

    latest_feat = max(feature_paths, key=os.path.getmtime)
    base_dir = os.path.dirname(latest_feat)
    paired_seg = os.path.join(base_dir, "segmentation_head.pth")
    if not os.path.isfile(paired_seg):
        paired_seg = max(seg_paths, key=os.path.getmtime)
    return latest_feat, paired_seg


def _build_models(args, device):
    if args.featureextractor_arch == "dino":
        feat = DinoFeaturizer(args).to(device).eval()
        args.image_nc = args.input_nc
        args.input_nc = feat.n_feats
        if args.segmentationhead_arch != "unetr":
            raise ValueError("For dino backbone, use --segmentationhead_arch unetr")
        dec = UneTrDecoder(args).to(device).eval()
    elif args.featureextractor_arch == "deeplab":
        feat = DeepLabFeaturizer(args).to(device).eval()
        dec = DeepLabDecoder(args).to(device).eval()
    elif args.featureextractor_arch == "vnet":
        feat = VnetFeaturizer(args).to(device).eval()
        dec = VnetDecoder(args).to(device).eval()
    else:
        raise ValueError(f"Unknown featureextractor_arch: {args.featureextractor_arch}")
    return feat, dec


def main():
    parser = argparse.ArgumentParser(description="Inference on TEST set (no references), output .npy predictions.")
    parser.add_argument("--test_dir", type=str, default="./dataset/TEST")
    parser.add_argument("--t1_path", type=str, default=None)
    parser.add_argument("--t2_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./EXPERIMENTS/FINAL_TEST_PRED")
    parser.add_argument("--finetune_root", type=str, default="./EXPERIMENTS/FINETUNE")

    parser.add_argument("--featureextractor_arch", type=str, default="dino", choices=["dino", "deeplab", "vnet"])
    parser.add_argument("--segmentationhead_arch", type=str, default="unetr")
    parser.add_argument("--fe_pretrained_weights", type=str, default=None)
    parser.add_argument("--sh_pretrained_weights", type=str, default=None)

    parser.add_argument("--input_patch_size", type=int, default=128)
    parser.add_argument("--overlap_porcent", type=float, default=0.55)
    parser.add_argument("--procedure", type=str, default="non_overlapped_center", choices=["non_overlapped_center", "average_overlap"])

    parser.add_argument("--dino_arch", type=str, default="vit_small")
    parser.add_argument("--dino_patch_size", type=int, default=16)
    parser.add_argument("--dino_imagenet_pretrain", type=eval, choices=[True, False], default=False)
    parser.add_argument("--dino_intermediate_blocks", nargs="+", type=int, default=[2, 5, 8, 11])

    parser.add_argument("--deeplabbackbone", type=str, default="resnet")
    parser.add_argument("--deeplaboutput_stride", type=int, default=16)
    parser.add_argument("--deeplab_imagenet_pretrain", type=eval, choices=[True, False], default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.fe_pretrained_weights is None or args.sh_pretrained_weights is None:
        feat_pth, seg_pth = _find_latest_finetune_checkpoints(args.finetune_root)
        args.fe_pretrained_weights = feat_pth
        args.sh_pretrained_weights = seg_pth
    print("[Final Test] Feature extractor weights:", args.fe_pretrained_weights)
    print("[Final Test] Segmentation head weights:", args.sh_pretrained_weights)

    t1_path, t2_path = _resolve_test_pair(args.test_dir, args.t1_path, args.t2_path)
    print("[Final Test] T1:", t1_path)
    print("[Final Test] T2:", t2_path)

    image_t1 = _load_npy_as_hwc(t1_path)
    image_t2 = _load_npy_as_hwc(t2_path)
    print("[Final Test] Loaded shapes:", image_t1.shape, image_t2.shape)

    image_t1, image_t2, _ = normalize(image_t1, image_t2)
    image = np.concatenate((image_t1, image_t2), axis=2)
    print("[Final Test] Stacked normalized shape:", image.shape)

    coords_info = _build_coordinates(
        img_rows=image.shape[0],
        img_cols=image.shape[1],
        input_patch_size=args.input_patch_size,
        overlap_porcent=args.overlap_porcent,
    )
    image_padded = np.pad(image, coords_info["pad_tuple_images"], mode="symmetric")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Final Test] Device:", device)

    model_args = SimpleNamespace(
        input_nc=image.shape[2],
        output_nc=2,
        featureextractor_arch=args.featureextractor_arch,
        segmentationhead_arch=args.segmentationhead_arch,
        input_patch_size=args.input_patch_size,
        fe_pretrained_weights=args.fe_pretrained_weights,
        sh_pretrained_weights=args.sh_pretrained_weights,
        phase="test",
        freeze_featureextractor=False,
        dino_arch=args.dino_arch,
        dino_patch_size=args.dino_patch_size,
        dino_imagenet_pretrain=args.dino_imagenet_pretrain,
        dino_intermediate_blocks=args.dino_intermediate_blocks,
        deeplabbackbone=args.deeplabbackbone,
        deeplaboutput_stride=args.deeplaboutput_stride,
        deeplab_imagenet_pretrain=args.deeplab_imagenet_pretrain,
    )
    feature_extractor, segmentation_head = _build_models(model_args, device)

    if args.procedure == "non_overlapped_center":
        heat_map = np.zeros((image_padded.shape[0], image_padded.shape[1], 2), dtype=np.float32)
    else:
        logits_sum = torch.zeros((image_padded.shape[0], image_padded.shape[1], 2), device=device)
        logits_count = torch.zeros((image_padded.shape[0], image_padded.shape[1]), device=device)

    coordinates = coords_info["coordinates"]
    stride = coords_info["stride"]
    overlap = coords_info["overlap"]
    k1 = coords_info["k1"]
    k2 = coords_info["k2"]
    step_row = coords_info["step_row"]
    step_col = coords_info["step_col"]

    with torch.no_grad():
        for i in range(coordinates.shape[0]):
            y_min, x_min, y_max, x_max = coordinates[i]
            patch = image_padded[y_min:y_max, x_min:x_max, :]
            patch = torch.from_numpy(np.transpose(patch, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=torch.float32)

            if model_args.featureextractor_arch == "dino":
                feats = feature_extractor(patch, n=model_args.dino_intermediate_blocks)
            else:
                feats = feature_extractor(patch)
            logits, probs = segmentation_head(feats)

            if args.procedure == "non_overlapped_center":
                probs_np = probs.cpu().numpy()[0]
                for c in range(2):
                    heat_map[
                        y_min:y_min + stride,
                        x_min:x_min + stride,
                        c
                    ] = probs_np[c, overlap // 2: overlap // 2 + stride, overlap // 2: overlap // 2 + stride]
            else:
                for c in range(2):
                    logits_sum[y_min:y_max, x_min:x_max, c] += logits[0, c, :, :]
                logits_count[y_min:y_max, x_min:x_max] += 1

    if args.procedure == "average_overlap":
        heat_map = torch.zeros_like(logits_sum)
        for c in range(2):
            heat_map[:, :, c] = torch.div(logits_sum[:, :, c], logits_count)
        heat_map = heat_map.softmax(2).cpu().numpy()

    heat_map = heat_map[: k1 * stride - step_row, : k2 * stride - step_col, :]
    pred = np.argmax(heat_map, axis=2).astype(np.uint8)
    prob = heat_map[:, :, 1].astype(np.float32)

    pred_path = os.path.join(args.output_dir, "prediction_map.npy")
    prob_path = os.path.join(args.output_dir, "probability_map.npy")
    full_path = os.path.join(args.output_dir, "full_prediction.npy")

    np.save(pred_path, pred)
    np.save(prob_path, prob)
    np.save(full_path, np.concatenate((image, pred[:, :, np.newaxis], prob[:, :, np.newaxis]), axis=2))

    print("[Final Test] Saved:", pred_path)
    print("[Final Test] Saved:", prob_path)
    print("[Final Test] Saved:", full_path)
    print("[Final Test] Done.")


if __name__ == "__main__":
    main()
