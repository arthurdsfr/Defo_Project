import argparse
import shlex
import subprocess
import sys


def _bool_str(v: bool) -> str:
    return "True" if v else "False"


def _run(cmd):
    print("[Finetune Pipeline] Running:")
    print(" ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Finetune SSL encoder on CERRADO (MA)")
    parser.add_argument("--tasks", type=str, default="train,test,getmetrics",
                        help="Comma-separated list: train,test,getmetrics")
    parser.add_argument("--ssl_checkpoint", type=str, required=True,
                        help="Path to SSL checkpoint (.pth). It will be used as --fe_pretrained_weights in train stage.")
    parser.add_argument("--experiment_name", type=str, default="/CERRADO_SSL_FINETUNE",
                        help="Experiment folder name under experiment_mainpath + overall_projectname")
    parser.add_argument("--experiment_mainpath", type=str, default="./EXPERIMENTS/")
    parser.add_argument("--overall_projectname", type=str, default="/FINETUNE")
    parser.add_argument("--data_path", type=str, default="./dataset")
    parser.add_argument("--input_patch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--freeze_featureextractor", type=eval, choices=[True, False], default=False)
    parser.add_argument("--multiple_gpus", type=eval, choices=[True, False], default=False)
    parser.add_argument("--training_graphs", type=eval, choices=[True, False], default=True)
    parser.add_argument("--featureextractor_arch", type=str, default="dino", choices=["dino", "vnet", "deeplab"])
    parser.add_argument("--segmentationhead_arch", type=str, default="unetr",
                        help="Use 'unetr' for dino backbone. Use 'None' for vnet/deeplab.")
    parser.add_argument("--dino_arch", type=str, default="vit_base")
    parser.add_argument("--dino_patch_size", type=int, default=16)
    parser.add_argument("--dino_imagenet_pretrain", type=eval, choices=[True, False], default=False)
    parser.add_argument("--dino_intermediate_blocks", nargs="+", type=int, default=[2, 5, 8, 11])
    parser.add_argument("--deeplabbackbone", type=str, default="resnet")
    parser.add_argument("--deeplaboutput_stride", type=int, default=16)
    args = parser.parse_args()

    tasks = {t.strip().lower() for t in args.tasks.split(",") if t.strip()}

    base = [
        "--featureextractor_arch", args.featureextractor_arch,
        "--segmentationhead_arch", args.segmentationhead_arch,
        "--input_patch_size", str(args.input_patch_size),
        "--dataset_name", "deforestation",
        "--background", "True",
        "--num_classes", "2",
        "--multiple_gpus", _bool_str(args.multiple_gpus),
        "--data_path", args.data_path,
        "--csv_path", "None",
        "--experiment_mainpath", args.experiment_mainpath,
        "--overall_projectname", args.overall_projectname,
        "--experiment_name", args.experiment_name,
        "--task", "deforestation_classifier",
        "--domains", "MA",
        "--deeplaboutput_stride", str(args.deeplaboutput_stride),
        "--deeplabbackbone", args.deeplabbackbone,
        "--dino_patch_size", str(args.dino_patch_size),
        "--dino_checkpoint_key", "teacher",
        "--dino_arch", args.dino_arch,
        "--dino_imagenet_pretrain", _bool_str(args.dino_imagenet_pretrain),
        "--freeze_featureextractor", _bool_str(args.freeze_featureextractor),
    ]

    if "train" in tasks:
        train_cmd = [
            sys.executable, "train.py",
            *base,
            "--fe_pretrained_weights", args.ssl_checkpoint,
            "--sh_pretrained_weights", "None",
            "--train_type", "sliding_windows",
            "--overlap_porcent", "0.95",
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--optimizer", args.optimizer,
            "--scheduler_type", args.scheduler_type,
            "--weights", "manual",
            "--cost_function", "cross_entropy",
            "--runs", str(args.runs),
            "--batch_size_per_gpu", str(args.batch_size),
            "--training_iterations", "0",
            "--validati_iterations", "0",
            "--dist_url", "env://",
            "--val_freq", "1",
            "--patience", str(args.patience),
            "--num_workers", str(args.num_workers),
            "--phase", "train",
            "--training_graphs", _bool_str(args.training_graphs),
        ]
        _run(train_cmd)

    if "test" in tasks:
        test_cmd = [
            sys.executable, "test.py",
            *base,
            "--fe_pretrained_weights", "None",
            "--sh_pretrained_weights", "None",
            "--eval_type", "sliding_windows",
            "--loader_crop", "None",
            "--overlap_porcent", "0.55",
            "--procedure", "non_overlapped_center",
            "--batch_size_per_gpu", "1",
            "--dist_url", "env://",
            "--num_workers", str(args.num_workers),
            "--phase", "test",
        ]
        _run(test_cmd)

    if "getmetrics" in tasks:
        metrics_cmd = [
            sys.executable, "get_metrics.py",
            *base,
            "--fe_pretrained_weights", "None",
            "--sh_pretrained_weights", "None",
            "--eval_type", "sliding_windows",
            "--loader_crop", "None",
            "--overlap_porcent", "0.55",
            "--procedure", "non_overlapped_center",
            "--batch_size_per_gpu", "1",
            "--num_workers", str(args.num_workers),
            "--phase", "get_metrics",
        ]
        _run(metrics_cmd)


if __name__ == "__main__":
    main()
