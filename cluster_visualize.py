"""Visualize HCFormer cluster assignments.

This entry point supports single-layer visualization and cross-layer FEC
aggregation. Run ``python cluster_visualize.py --help`` for usage.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from timm.models import load_checkpoint
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks

import models  # noqa: F401 - imports and registers local timm models
import timm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
COLORS = [
    "brown",
    "green",
    "deepskyblue",
    "blue",
    "darkgreen",
    "darkcyan",
    "coral",
    "aliceblue",
    "white",
    "black",
    "beige",
    "red",
    "tomato",
    "yellowgreen",
    "violet",
    "mediumseagreen",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("visualize", "fec"),
        default="visualize",
        help="single-layer visualization or cross-layer FEC",
    )
    parser.add_argument("--image", default="images/A.JPEG", help="input image")
    parser.add_argument("--shape", type=int, default=224, help="model input size")
    parser.add_argument("--model", default="hcformer_tiny", help="registered timm model")
    parser.add_argument("--checkpoint", default="", help="model checkpoint")
    parser.add_argument("--device", default="cuda", help="inference device")
    parser.add_argument("--output-dir", default="images/cluster_vis")

    parser.add_argument("--stage", type=int, default=0, help="stage index")
    parser.add_argument("--block", type=int, default=0, help="block index")
    parser.add_argument("--head", type=int, default=0, help="attention head")
    parser.add_argument(
        "--branch",
        choices=("local", "global-euclidean", "global-hyperbolic"),
        default="global-hyperbolic",
    )
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=0,
        help="merge masks to K clusters; 0 keeps the original assignments",
    )
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--kmeans-feature",
        choices=("geometry", "feature"),
        default="geometry",
    )
    parser.add_argument(
        "--merge-mode",
        choices=("replace", "append"),
        default="replace",
        help="replace original masks or append merged masks",
    )

    # Compatibility with commands used by the former scripts.
    parser.add_argument("--local", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--hyperbolic", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kmeans_merge", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num_k", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def normalize_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.local:
        args.branch = "local"
    elif args.hyperbolic:
        args.branch = "global-hyperbolic"
    if args.kmeans_merge is not None:
        args.num_clusters = args.kmeans_merge
    if args.num_k is not None:
        args.num_clusters = args.num_k
    return args


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(name)


def preprocess(image_path: str | Path, shape: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((shape, shape))
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return transform(image)


def load_labels() -> list[str]:
    label_path = Path(__file__).with_name("imagenet1k_id_to_label.txt")
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        _, label = line.split(":", maxsplit=1)
        labels.append(label.strip())
    return labels


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.model not in timm.list_models():
        raise ValueError(f"Unknown model {args.model!r}")
    model = timm.create_model(args.model, pretrained=False)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, use_ema=True)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint loaded; visualizing current model weights")
    return model.to(device).eval()


def selected_mixer(model: torch.nn.Module, stage: int, block: int) -> torch.nn.Module:
    network_index = stage * 2
    if network_index >= len(model.network):
        raise IndexError(f"stage {stage} is outside the model network")
    blocks = model.network[network_index]
    try:
        return blocks[block].token_mixer
    except IndexError as error:
        raise IndexError(f"block {block} is outside stage {stage}") from error


def pairwise_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return x @ y.transpose(-2, -1)


def hard_assign(similarity: torch.Tensor) -> torch.Tensor:
    labels = similarity.argmax(dim=1, keepdim=True)
    return torch.zeros_like(similarity).scatter_(1, labels, 1.0)


def clip_tangent(vector: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    norm = vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(radius / norm, max=1.0)
    return scale * vector


def local_masks(mixer: torch.nn.Module, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    projected = mixer.f(feature)
    values = mixer.v(feature)
    bias = F.interpolate(
        mixer.relative_position_bias,
        size=projected.shape[-2:],
        mode="bilinear",
        align_corners=True,
    )
    bias = bias.unsqueeze(0).expand(projected.shape[0], -1, -1, -1, -1)
    projected = rearrange(projected, "b (e c) h w -> (b e) c h w", e=mixer.heads)
    values = rearrange(values, "b (e c) h w -> (b e) c h w", e=mixer.heads)
    bias = rearrange(bias, "b e p h w -> (b e) p h w")

    if mixer.fold_w > 1 or mixer.fold_h > 1:
        height, width = projected.shape[-2:]
        if height % mixer.fold_w or width % mixer.fold_h:
            raise ValueError(
                f"feature map {height}x{width} is not divisible by "
                f"fold {mixer.fold_w}x{mixer.fold_h}"
            )
        projected = rearrange(
            projected,
            "b c (fh h) (fw w) -> (b fh fw) c h w",
            fh=mixer.fold_w,
            fw=mixer.fold_h,
        )
        values = rearrange(
            values,
            "b c (fh h) (fw w) -> (b fh fw) c h w",
            fh=mixer.fold_w,
            fw=mixer.fold_h,
        )
        bias = rearrange(
            bias,
            "b p (fh h) (fw w) -> (b fh fw) p h w",
            fh=mixer.fold_w,
            fw=mixer.fold_h,
        )

    batch, channels, height, width = projected.shape
    centers = mixer.centers_proposal(projected)
    similarity = F.softmax(
        bias.flatten(2)
        + mixer.sim_alpha
        * pairwise_cosine(centers.flatten(2).transpose(1, 2), projected.flatten(2).transpose(1, 2)),
        dim=1,
    )
    masks = hard_assign(similarity).view(batch, -1, height, width)
    masks = rearrange(
        masks,
        "(b e fh fw) m h w -> b e fh fw m h w",
        e=mixer.heads,
        fh=mixer.fold_w,
        fw=mixer.fold_h,
    )

    canvas = torch.zeros(
        masks.shape[0],
        masks.shape[1],
        masks.shape[4] * mixer.fold_w * mixer.fold_h,
        height * mixer.fold_w,
        width * mixer.fold_h,
        device=masks.device,
    )
    cluster_index = 0
    for fold_y in range(mixer.fold_w):
        for fold_x in range(mixer.fold_h):
            count = masks.shape[4]
            canvas[
                :,
                :,
                cluster_index : cluster_index + count,
                fold_y * height : (fold_y + 1) * height,
                fold_x * width : (fold_x + 1) * width,
            ] = masks[:, :, fold_y, fold_x]
            cluster_index += count

    value_map = rearrange(
        values,
        "(b e fh fw) c h w -> b e c (fh h) (fw w)",
        b=feature.shape[0],
        e=mixer.heads,
        fh=mixer.fold_w,
        fw=mixer.fold_h,
    )
    return canvas[0].permute(1, 0, 2, 3), value_map[0]


def global_masks(
    mixer: torch.nn.Module,
    feature: torch.Tensor,
    hyperbolic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not getattr(mixer, "global_compute", False):
        raise ValueError("the selected block does not have a global clustering branch")
    projected = rearrange(
        mixer.f(feature), "b (e c) h w -> (b e) c h w", e=mixer.heads
    )
    values = rearrange(
        mixer.v(feature), "b (e c) h w -> (b e) c h w", e=mixer.heads
    )
    batch_heads, channels = projected.shape[:2]
    windows = mixer.window_reverse(mixer.centers_fold(mixer.window_partition(projected)))
    centers = mixer.centers_window(projected)
    center_points = centers.flatten(2).transpose(1, 2)
    window_points = windows.flatten(2).transpose(1, 2)

    if hyperbolic:
        center_points = clip_tangent(center_points)
        window_points = clip_tangent(window_points)
        center_points = mixer.manifold.expmap0(center_points, dim=-1)
        window_points = mixer.manifold.expmap0(window_points, dim=-1)
        score = -mixer.manifold.dist(
            center_points.unsqueeze(2), window_points.unsqueeze(1), dim=-1
        )
    else:
        score = pairwise_cosine(center_points, window_points)

    similarity = F.softmax(mixer.relative_position_bias2 + mixer.sim_alpha2 * score, dim=1)
    height, width = windows.shape[-2:]
    masks = hard_assign(similarity).view(batch_heads, -1, height, width)
    masks = rearrange(
        masks,
        "(b e) m h w -> b m e h w",
        b=feature.shape[0],
        e=mixer.heads,
    )
    value_map = rearrange(
        windows,
        "(b e) c h w -> b e c h w",
        b=feature.shape[0],
        e=mixer.heads,
    )
    return masks[0], value_map[0]


class AssignmentCapture:
    def __init__(self, branch: str):
        self.branch = branch
        self.masks: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def __call__(self, mixer, inputs, _output):
        feature = inputs[0]
        if self.branch == "local":
            self.masks, self.values = local_masks(mixer, feature)
        else:
            self.masks, self.values = global_masks(
                mixer,
                feature,
                hyperbolic=self.branch == "global-hyperbolic",
            )


def cluster_features(
    masks: torch.Tensor,
    value_map: torch.Tensor,
    feature_type: str,
) -> torch.Tensor:
    height, width = masks.shape[-2:]
    flat_masks = masks.flatten(1)
    area = flat_masks.sum(dim=1).clamp_min(1.0)
    if feature_type == "geometry":
        yy, xx = torch.meshgrid(
            torch.arange(height, device=masks.device),
            torch.arange(width, device=masks.device),
            indexing="ij",
        )
        center_x = (masks * xx).flatten(1).sum(1) / area
        center_y = (masks * yy).flatten(1).sum(1) / area
        return torch.stack(
            (center_x / max(width, 1), center_y / max(height, 1), area / (height * width)),
            dim=1,
        )

    value_map = F.interpolate(
        value_map.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    )[0]
    return flat_masks @ value_map.flatten(1).transpose(0, 1) / area.unsqueeze(1)


def kmeans(features: torch.Tensor, count: int, iters: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=features.device).manual_seed(seed)
    centers = features[
        torch.randperm(features.shape[0], generator=generator, device=features.device)[:count]
    ].clone()
    for _ in range(iters):
        distances = torch.cdist(features, centers)
        assignment = distances.argmin(dim=1)
        for index in range(count):
            selected = assignment == index
            if selected.any():
                centers[index] = features[selected].mean(dim=0)
    return torch.cdist(features, centers).argmin(dim=1)


def merge_masks(
    masks: torch.Tensor,
    value_map: torch.Tensor,
    count: int,
    feature_type: str,
    iters: int,
    seed: int,
) -> torch.Tensor:
    if count <= 0 or count >= masks.shape[0]:
        return masks
    features = cluster_features(masks, value_map, feature_type)
    assignment = kmeans(features, count, iters, seed)
    merged = []
    for index in range(count):
        selected = masks[assignment == index]
        merged.append(
            selected.sum(dim=0).clamp_max(1.0)
            if selected.numel()
            else torch.zeros_like(masks[0])
        )
    return torch.stack(merged)


def palette(size: int, seed: int) -> list[str]:
    colors = (COLORS * ((size + len(COLORS) - 1) // len(COLORS)))[:size]
    random.Random(seed).shuffle(colors)
    return colors


def save_overlay(
    image_path: str | Path,
    masks: torch.Tensor,
    output_path: Path,
    alpha: float,
    seed: int,
) -> None:
    image = read_image(str(image_path))
    masks = F.interpolate(
        masks.unsqueeze(0).float(),
        size=image.shape[-2:],
        mode="nearest",
    )[0].bool().cpu()
    overlay = draw_segmentation_masks(
        image,
        masks=masks,
        alpha=alpha,
        colors=palette(masks.shape[0], seed),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(overlay).save(output_path)
    print(f"Saved visualization: {output_path}")


def print_prediction(output: torch.Tensor, labels: list[str]) -> None:
    if output.ndim != 2:
        return
    probability, index = output.softmax(dim=1).max(dim=1)
    class_index = index.item()
    name = labels[class_index] if class_index < len(labels) else str(class_index)
    print(f"Prediction: {name} ({probability.item() * 100:.3f}%)")


@torch.no_grad()
def run_visualize(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    capture = AssignmentCapture(args.branch)
    mixer = selected_mixer(model, args.stage, args.block)
    if not 0 <= args.head < mixer.heads:
        raise ValueError(f"head must be in [0, {mixer.heads - 1}]")
    handle = mixer.register_forward_hook(capture)
    output = model(preprocess(args.image, args.shape).unsqueeze(0).to(device))
    handle.remove()
    if isinstance(output, tuple):
        output = output[0]
    print_prediction(output, load_labels())
    if capture.masks is None or capture.values is None:
        raise RuntimeError("assignment hook did not capture any masks")

    masks = capture.masks[:, args.head]
    merged = merge_masks(
        masks,
        capture.values[args.head],
        args.num_clusters,
        args.kmeans_feature,
        args.kmeans_iters,
        args.seed,
    )
    if args.merge_mode == "append" and merged.shape[0] != masks.shape[0]:
        merged = torch.cat((masks, merged), dim=0)

    suffix = f"{args.branch}_k{args.num_clusters or masks.shape[0]}"
    output_path = (
        Path(args.output_dir)
        / args.model
        / f"{Path(args.image).stem}_s{args.stage}_b{args.block}_h{args.head}_{suffix}.png"
    )
    save_overlay(args.image, merged, output_path, args.alpha, args.seed)


class FECCapture:
    def __init__(self):
        self.layers: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, mixer, inputs, _output):
        feature = mixer.f(inputs[0])
        height, width = feature.shape[-2:]
        centers = F.adaptive_avg_pool2d(feature, (height // 2, width // 2))
        similarity = pairwise_cosine(
            centers.flatten(2).transpose(1, 2),
            feature.flatten(2).transpose(1, 2),
        )
        assignment = similarity.argmax(dim=1)[0].detach()
        self.layers.append((assignment, centers[0].flatten(1).transpose(0, 1).detach()))


def compose_fec_masks(
    layers: list[tuple[torch.Tensor, torch.Tensor]],
    count: int,
    iters: int,
    seed: int,
) -> torch.Tensor:
    if not layers:
        raise RuntimeError("FEC hooks did not capture any assignments")
    assignment = layers[0][0]
    for next_assignment, _ in layers[1:]:
        assignment = next_assignment[assignment]
    cluster_count = layers[-1][1].shape[0]
    masks = F.one_hot(assignment, num_classes=cluster_count).transpose(0, 1).float()
    side = int(round(assignment.numel() ** 0.5))
    if side * side != assignment.numel():
        raise ValueError("FEC currently requires a square feature map")
    masks = masks.view(cluster_count, side, side)

    nonempty = masks.flatten(1).any(dim=1)
    masks = masks[nonempty]
    if 0 < count < masks.shape[0]:
        final_features = layers[-1][1][nonempty]
        groups = kmeans(final_features, count, iters, seed)
        masks = torch.stack(
            [masks[groups == index].sum(0).clamp_max(1.0) for index in range(count)]
        )
    return masks


@torch.no_grad()
def run_fec(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> None:
    capture = FECCapture()
    handles = []
    for network_index in (0, 2, 4):
        if network_index >= len(model.network):
            break
        handles.append(
            model.network[network_index][0].token_mixer.register_forward_hook(capture)
        )
    output = model(preprocess(args.image, args.shape).unsqueeze(0).to(device))
    for handle in handles:
        handle.remove()
    if isinstance(output, tuple):
        output = output[0]
    print_prediction(output, load_labels())
    masks = compose_fec_masks(
        capture.layers,
        args.num_clusters,
        args.kmeans_iters,
        args.seed,
    )
    output_path = (
        Path(args.output_dir)
        / args.model
        / f"{Path(args.image).stem}_fec_k{args.num_clusters or masks.shape[0]}.png"
    )
    save_overlay(args.image, masks, output_path, args.alpha, args.seed)


def main() -> None:
    args = normalize_legacy_args(build_parser().parse_args())
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = build_model(args, device)
    if args.mode == "visualize":
        run_visualize(args, model, device)
    else:
        run_fec(args, model, device)


if __name__ == "__main__":
    main()
