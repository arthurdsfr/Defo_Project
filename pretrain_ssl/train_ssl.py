import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import dino.utils as dino_utils
import dino.vision_transformer as vits
from pretrain_ssl.augment import MultiCropAugmentation
from pretrain_ssl.dataset import AmazonSSLDataset, multicrop_collate_fn


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        warmup = torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs)
        if nepochs > warmup_teacher_temp_epochs:
            remaining = torch.full((nepochs - warmup_teacher_temp_epochs,), teacher_temp)
            self.teacher_temp_schedule = torch.cat([warmup, remaining], dim=0)
        else:
            self.teacher_temp_schedule = warmup[:nepochs]

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1).detach()
        teacher_out = teacher_out.chunk(2)  # Teacher sees only 2 global views.

        total_loss = 0.0
        n_terms = 0
        for iq, q in enumerate(teacher_out):
            for v, s in enumerate(student_out):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(s, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_terms += 1

        total_loss /= n_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / teacher_output.shape[0]
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)


def _build_networks(args, in_chans):
    student_backbone = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        in_chans=in_chans,
        num_classes=0,
    )
    teacher_backbone = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        in_chans=in_chans,
        num_classes=0,
    )

    embed_dim = student_backbone.embed_dim
    student = dino_utils.MultiCropWrapper(
        student_backbone,
        vits.DINOHead(
            in_dim=embed_dim,
            out_dim=args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
            nlayers=args.head_nlayers,
            hidden_dim=args.head_hidden_dim,
            bottleneck_dim=args.head_bottleneck_dim,
        ),
    )
    teacher = dino_utils.MultiCropWrapper(
        teacher_backbone,
        vits.DINOHead(
            in_dim=embed_dim,
            out_dim=args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=False,
            nlayers=args.head_nlayers,
            hidden_dim=args.head_hidden_dim,
            bottleneck_dim=args.head_bottleneck_dim,
        ),
    )

    teacher.load_state_dict(student.state_dict(), strict=True)
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher


def train_ssl(args):
    dino_utils.fix_random_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[SSL Train] Warning: running on CPU. SSL pretraining will be very slow.")
    if os.name == "nt" and args.num_workers > 0:
        print(
            f"[SSL Train] Windows detected. Overriding num_workers={args.num_workers} to 0 "
            "to avoid DataLoader spawn/pickle failures with large in-memory arrays."
        )
        args.num_workers = 0

    transform = MultiCropAugmentation(
        global_crops_size=args.global_crops_size,
        local_crops_size=args.local_crops_size,
        global_crops_scale=tuple(args.global_crops_scale),
        local_crops_scale=tuple(args.local_crops_scale),
        local_crops_number=args.local_crops_number,
        noise_std=args.noise_std,
        debug=args.debug_data,
    )
    dataset = AmazonSSLDataset(
        data_path=args.data_path,
        transforms=transform,
        input_patch_size=args.input_patch_size,
        overlap_porcent=args.overlap_porcent,
        set_id=args.set_id,
        task="ssl_pretrain",
        debug=args.debug_data,
        max_debug_samples=args.max_debug_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=multicrop_collate_fn,
    )
    if len(loader) == 0:
        raise RuntimeError("DataLoader is empty. Increase dataset size or lower batch_size.")

    print(f"[SSL Train] Number of batches per epoch: {len(loader)}")
    print(f"[SSL Train] Input channels: {dataset.num_channels}")

    student, teacher = _build_networks(args, dataset.num_channels)
    student = student.to(device)
    teacher = teacher.to(device)

    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        ncrops=2 + args.local_crops_number,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
    ).to(device)

    params_groups = dino_utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups, lr=args.lr, betas=(0.9, 0.999))
    fp16_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.use_fp16))

    niter_per_epoch = len(loader)
    total_steps = args.epochs * niter_per_epoch
    lr_schedule = dino_utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        niter_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = dino_utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        niter_per_epoch,
    )
    momentum_schedule = dino_utils.cosine_scheduler(
        args.momentum_teacher,
        1.0,
        args.epochs,
        niter_per_epoch,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ssl_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("[SSL Train] Starting training loop...")
    global_step = 0
    for epoch in range(args.epochs):
        student.train()
        teacher.eval()
        epoch_loss = 0.0

        for it, batch in enumerate(loader):
            step = epoch * niter_per_epoch + it

            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[step]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[step]

            crops = [c.to(device, non_blocking=True) for c in batch["crops"]]
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.use_fp16)):
                teacher_output = teacher(crops[:2])  # only global crops
                student_output = student(crops)      # global + local crops
                loss = dino_loss(student_output, teacher_output, epoch)

            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Loss is not finite at step {step}: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            fp16_scaler.scale(loss).backward()

            if args.clip_grad > 0:
                fp16_scaler.unscale_(optimizer)
                dino_utils.clip_gradients(student, args.clip_grad)

            dino_utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

            with torch.no_grad():
                m = momentum_schedule[step]
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

            epoch_loss += loss.item()
            global_step += 1

            if (it + 1) % args.print_freq == 0 or (it + 1) == niter_per_epoch:
                print(
                    f"[SSL Train] Epoch {epoch + 1}/{args.epochs} | "
                    f"Iter {it + 1}/{niter_per_epoch} | "
                    f"Loss {loss.item():.4f} | "
                    f"LR {optimizer.param_groups[0]['lr']:.6e} | "
                    f"WD {optimizer.param_groups[0]['weight_decay']:.6e}"
                )

        avg_loss = epoch_loss / niter_per_epoch
        print(f"[SSL Train] Epoch {epoch + 1} completed | avg loss: {avg_loss:.4f}")

        save_every = (args.saveckpt_freq > 0) and ((epoch + 1) % args.saveckpt_freq == 0)
        is_last = (epoch + 1) == args.epochs
        if save_every or is_last:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pth"
            checkpoint = {
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "args": vars(args),
                "niter_per_epoch": niter_per_epoch,
                "total_steps": total_steps,
            }
            torch.save(checkpoint, ckpt_path)
            torch.save(checkpoint, output_dir / "checkpoint_last.pth")
            print(f"[SSL Train] Checkpoint saved: {ckpt_path}")


def _build_parser():
    parser = argparse.ArgumentParser("SSL pretraining on AMAZON_RO with DINO")

    # Data
    parser.add_argument("--data_path", type=str, default="./dataset")
    parser.add_argument("--input_patch_size", type=int, default=224)
    parser.add_argument("--overlap_porcent", type=float, default=0.5)
    parser.add_argument("--set_id", type=int, default=1, help="1=train tiles, 3=val tiles from existing split logic")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)

    # Augmentation
    parser.add_argument("--global_crops_size", type=int, default=224)
    parser.add_argument("--local_crops_size", type=int, default=96)
    parser.add_argument("--global_crops_scale", nargs=2, type=float, default=[0.4, 1.0])
    parser.add_argument("--local_crops_scale", nargs=2, type=float, default=[0.1, 0.4])
    parser.add_argument("--local_crops_number", type=int, default=6)
    parser.add_argument("--noise_std", type=float, default=0.01)

    # Model
    parser.add_argument("--arch", type=str, default="vit_small", choices=["vit_tiny", "vit_small", "vit_base"])
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--out_dim", type=int, default=65536)
    parser.add_argument("--norm_last_layer", type=eval, choices=[True, False], default=True)
    parser.add_argument("--use_bn_in_head", type=eval, choices=[True, False], default=False)
    parser.add_argument("--head_nlayers", type=int, default=3)
    parser.add_argument("--head_hidden_dim", type=int, default=2048)
    parser.add_argument("--head_bottleneck_dim", type=int, default=256)

    # Loss and optimization
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--weight_decay_end", type=float, default=0.4)
    parser.add_argument("--clip_grad", type=float, default=3.0)
    parser.add_argument("--freeze_last_layer", type=int, default=1)
    parser.add_argument("--momentum_teacher", type=float, default=0.996)
    parser.add_argument("--student_temp", type=float, default=0.1)
    parser.add_argument("--warmup_teacher_temp", type=float, default=0.04)
    parser.add_argument("--teacher_temp", type=float, default=0.07)
    parser.add_argument("--warmup_teacher_temp_epochs", type=int, default=30)
    parser.add_argument("--center_momentum", type=float, default=0.9)

    # Runtime / logging
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_fp16", type=eval, choices=[True, False], default=True)
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--saveckpt_freq", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./EXPERIMENTS/SSL_AMAZON")
    parser.add_argument("--debug_data", type=eval, choices=[True, False], default=False)
    parser.add_argument("--max_debug_samples", type=int, default=3)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_ssl(args)


if __name__ == "__main__":
    main()
