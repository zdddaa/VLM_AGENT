"""Encode change-object bounding boxes for SAM3 change conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from ..schemas.change_object import BBox


@dataclass(slots=True)
class BoxPromptEncoding:
    prompt_tokens: Tensor
    spatial_prior: Tensor | None
    normalized_boxes: Tensor
    area_ratio: Tensor


def _as_box_tensor(
    boxes: BBox | Tensor | Sequence[float] | Sequence[BBox],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(boxes, BBox):
        data = [boxes.as_tuple()]
        return torch.tensor(data, device=device, dtype=dtype)

    if isinstance(boxes, Tensor):
        tensor = boxes.to(device=device, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != 4:
            raise ValueError("box tensor must have shape [4] or [B,4]")
        return tensor

    if boxes and isinstance(boxes[0], BBox):  # type: ignore[index]
        data = [box.as_tuple() for box in boxes]  # type: ignore[union-attr]
        return torch.tensor(data, device=device, dtype=dtype)

    tensor = torch.tensor(boxes, device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[-1] != 4:
        raise ValueError("boxes must resolve to shape [B,4]")
    return tensor


class BoxPromptEncoder(nn.Module):
    """Convert pixel-space change boxes into prompt tokens and a token-grid prior.

    The encoder intentionally keeps box geometry separate from semantic features.
    It provides SAM3 with a hard spatial support prior while the change adapter
    supplies soft probability and temporal-token evidence.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        prompt_token_count: int = 2,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if prompt_token_count <= 0:
            raise ValueError("prompt_token_count must be positive")

        hidden = hidden_dim or max(feature_dim // 2, 32)
        self.feature_dim = feature_dim
        self.prompt_token_count = prompt_token_count
        self.encoder = nn.Sequential(
            nn.Linear(8, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim * prompt_token_count),
        )

    def forward(
        self,
        boxes: BBox | Tensor | Sequence[float] | Sequence[BBox],
        *,
        image_size: tuple[int, int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        spatial_shape: tuple[int, int] | None = None,
    ) -> BoxPromptEncoding:
        image_height, image_width = image_size
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image_size must contain positive height and width")

        box_tensor = _as_box_tensor(boxes, device=device, dtype=dtype)
        if box_tensor.shape[0] not in (1, batch_size):
            raise ValueError(
                "box batch dimension must be 1 or match token batch size, "
                f"got {box_tensor.shape[0]} vs {batch_size}"
            )
        if box_tensor.shape[0] == 1 and batch_size > 1:
            box_tensor = box_tensor.expand(batch_size, -1)

        x1, y1, x2, y2 = box_tensor.unbind(dim=-1)
        if torch.any(x2 <= x1) or torch.any(y2 <= y1):
            raise ValueError("all boxes must satisfy x2>x1 and y2>y1")

        normalized = torch.stack(
            (
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height,
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        nx1, ny1, nx2, ny2 = normalized.unbind(dim=-1)
        width = (nx2 - nx1).clamp_min(0.0)
        height = (ny2 - ny1).clamp_min(0.0)
        center_x = (nx1 + nx2) * 0.5
        center_y = (ny1 + ny2) * 0.5
        area_ratio = (width * height).clamp(0.0, 1.0)

        geometry_vector = torch.stack(
            (nx1, ny1, nx2, ny2, width, height, center_x, center_y),
            dim=-1,
        )
        prompt_tokens = self.encoder(geometry_vector).reshape(
            batch_size,
            self.prompt_token_count,
            self.feature_dim,
        )

        spatial_prior: Tensor | None = None
        if spatial_shape is not None:
            grid_h, grid_w = spatial_shape
            if grid_h <= 0 or grid_w <= 0:
                raise ValueError("spatial_shape must contain positive values")

            # Pixel-center coordinates in normalized [0,1] image space.
            ys = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) / grid_h
            xs = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) / grid_w
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            xx = xx.reshape(1, -1)
            yy = yy.reshape(1, -1)

            inside = (
                (xx >= nx1[:, None])
                & (xx <= nx2[:, None])
                & (yy >= ny1[:, None])
                & (yy <= ny2[:, None])
            )
            spatial_prior = inside.to(dtype=dtype).unsqueeze(-1)

        return BoxPromptEncoding(
            prompt_tokens=prompt_tokens,
            spatial_prior=spatial_prior,
            normalized_boxes=normalized,
            area_ratio=area_ratio,
        )
