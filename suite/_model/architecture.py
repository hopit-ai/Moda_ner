"""Fashion-pretrained encoder with conditional field-specific heads."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .contract import (
    CATEGORY_FIELDS,
    AttributeVocabulary,
)


def _encoder_output_dim(encoder: nn.Module) -> int:
    visual = encoder.visual
    explicit = getattr(visual, "output_dim", None)
    if explicit is not None:
        return int(explicit)
    projection = getattr(encoder, "text_projection", None)
    if isinstance(projection, torch.Tensor) and projection.ndim >= 1:
        return int(projection.shape[-1])
    head = getattr(visual, "head", None)
    if isinstance(head, nn.Module):
        for module in reversed(list(head.modules())):
            if isinstance(module, nn.Linear):
                return int(module.out_features)
    image_size = getattr(visual, "image_size", (224, 224))
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    with torch.no_grad():
        sample = torch.zeros(1, 3, int(image_size[0]), int(image_size[1]))
        return int(encoder.encode_image(sample).shape[-1])


class ConditionalAttributeHeads(nn.Module):
    """Decouple field applicability from fine-grained value recognition."""

    def __init__(
        self,
        input_dim: int,
        vocabulary: AttributeVocabulary,
        *,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.category_heads = nn.ModuleDict(
            {
                field: nn.Linear(hidden_dim, len(vocabulary.values[field]))
                for field in CATEGORY_FIELDS
                if field in vocabulary.fields
            }
        )
        conditional_fields = [
            field for field in vocabulary.fields if field not in CATEGORY_FIELDS
        ]
        self.applicability_heads = nn.ModuleDict(
            {field: nn.Linear(hidden_dim, 1) for field in conditional_fields}
        )
        self.value_heads = nn.ModuleDict(
            {
                field: nn.Linear(hidden_dim, len(vocabulary.values[field]))
                for field in conditional_fields
            }
        )

    def forward(self, features: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        hidden = self.shared(features)
        return {
            "categories": {
                field: head(hidden) for field, head in self.category_heads.items()
            },
            "applicability": {
                field: head(hidden).squeeze(-1)
                for field, head in self.applicability_heads.items()
            },
            "values": {field: head(hidden) for field, head in self.value_heads.items()},
        }


class SpatialConditionalAttributeHeads(nn.Module):
    """Conditional heads that learn a field-specific attention over patch tokens.

    Global pooled embeddings are strong for category but discard the local evidence needed for
    neckline, collar, material, and surface treatment.  Each field therefore gets a learned
    query over the encoder's patch tokens and fuses that attended summary with the global image
    embedding.  The output contract is intentionally identical to ``ConditionalAttributeHeads``.
    """

    def __init__(
        self,
        input_dim: int,
        patch_dim: int,
        vocabulary: AttributeVocabulary,
        *,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.global_shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
        )
        fields = tuple(vocabulary.fields)
        self.field_queries = nn.ParameterDict(
            {field: nn.Parameter(torch.empty(hidden_dim)) for field in fields}
        )
        self.field_fusions = nn.ModuleDict(
            {
                field: nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for field in fields
            }
        )
        self.category_heads = nn.ModuleDict(
            {
                field: nn.Linear(hidden_dim, len(vocabulary.values[field]))
                for field in CATEGORY_FIELDS
                if field in vocabulary.fields
            }
        )
        conditional_fields = [
            field for field in vocabulary.fields if field not in CATEGORY_FIELDS
        ]
        self.applicability_heads = nn.ModuleDict(
            {field: nn.Linear(hidden_dim, 1) for field in conditional_fields}
        )
        self.value_heads = nn.ModuleDict(
            {
                field: nn.Linear(hidden_dim, len(vocabulary.values[field]))
                for field in conditional_fields
            }
        )
        for query in self.field_queries.values():
            nn.init.normal_(query, std=hidden_dim**-0.5)

    def forward(
        self, features: torch.Tensor, patch_tokens: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        if patch_tokens.ndim != 3:
            raise ValueError(
                f"Expected patch tokens [batch, patches, dim], got {tuple(patch_tokens.shape)}"
            )
        global_hidden = self.global_shared(features)
        projected_tokens = self.patch_projection(patch_tokens)
        scale = projected_tokens.shape[-1] ** -0.5
        field_hidden: dict[str, torch.Tensor] = {}
        for field, query in self.field_queries.items():
            attention = torch.einsum("bnd,d->bn", projected_tokens, query) * scale
            weights = torch.softmax(attention, dim=-1)
            attended = torch.einsum("bn,bnd->bd", weights, projected_tokens)
            field_hidden[field] = self.field_fusions[field](
                torch.cat((global_hidden, attended), dim=-1)
            )
        return {
            "categories": {
                field: self.category_heads[field](field_hidden[field])
                for field in self.category_heads
            },
            "applicability": {
                field: self.applicability_heads[field](field_hidden[field]).squeeze(-1)
                for field in self.applicability_heads
            },
            "values": {
                field: self.value_heads[field](field_hidden[field])
                for field in self.value_heads
            },
        }


class SpatialResidualConditionalHeads(nn.Module):
    """Add zero-initialized patch evidence to a pretrained global classifier.

    The global conditional heads remain the anchor, so the model starts exactly at the existing
    checkpoint rather than paying a cold-start accuracy penalty.  Patch residuals are learned per
    field and can improve local attributes without forcing category heads to relearn the global
    representation.
    """

    def __init__(
        self,
        input_dim: int,
        patch_dim: int,
        vocabulary: AttributeVocabulary,
        *,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.base_heads = ConditionalAttributeHeads(
            input_dim,
            vocabulary,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
        )
        fields = tuple(vocabulary.fields)
        self.field_queries = nn.ParameterDict(
            {field: nn.Parameter(torch.empty(hidden_dim)) for field in fields}
        )
        self.field_adapters = nn.ModuleDict(
            {
                field: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for field in fields
            }
        )
        self.category_residuals = nn.ModuleDict(
            {
                field: nn.Linear(hidden_dim, len(vocabulary.values[field]))
                for field in CATEGORY_FIELDS
                if field in vocabulary.fields
            }
        )
        conditional_fields = [
            field for field in vocabulary.fields if field not in CATEGORY_FIELDS
        ]
        self.applicability_residuals = nn.ModuleDict(
            {field: nn.Linear(hidden_dim, 1) for field in conditional_fields}
        )
        self.value_residuals = nn.ModuleDict(
            {
                field: nn.Linear(hidden_dim, len(vocabulary.values[field]))
                for field in conditional_fields
            }
        )
        for query in self.field_queries.values():
            nn.init.normal_(query, std=hidden_dim**-0.5)
        for residual in (
            *self.category_residuals.values(),
            *self.applicability_residuals.values(),
            *self.value_residuals.values(),
        ):
            nn.init.zeros_(residual.weight)
            nn.init.zeros_(residual.bias)

    def forward(
        self, features: torch.Tensor, patch_tokens: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        if patch_tokens.ndim != 3:
            raise ValueError(
                f"Expected patch tokens [batch, patches, dim], got {tuple(patch_tokens.shape)}"
            )
        outputs = self.base_heads(features)
        projected_tokens = self.patch_projection(patch_tokens)
        scale = projected_tokens.shape[-1] ** -0.5
        residual_hidden: dict[str, torch.Tensor] = {}
        for field, query in self.field_queries.items():
            attention = torch.einsum("bnd,d->bn", projected_tokens, query) * scale
            weights = torch.softmax(attention, dim=-1)
            attended = torch.einsum("bn,bnd->bd", weights, projected_tokens)
            residual_hidden[field] = self.field_adapters[field](attended)
        for field, residual in self.category_residuals.items():
            outputs["categories"][field] = (
                outputs["categories"][field] + residual(residual_hidden[field])
            )
        for field, residual in self.applicability_residuals.items():
            outputs["applicability"][field] = (
                outputs["applicability"][field]
                + residual(residual_hidden[field]).squeeze(-1)
            )
        for field, residual in self.value_residuals.items():
            outputs["values"][field] = (
                outputs["values"][field] + residual(residual_hidden[field])
            )
        return outputs


def _visual_patch_dim(visual: nn.Module, fallback: int) -> int:
    """Find the pre-projection transformer width used by patch tokens."""
    ln_post = getattr(visual, "ln_post", None)
    normalized_shape = getattr(ln_post, "normalized_shape", None)
    if isinstance(normalized_shape, tuple) and normalized_shape:
        return int(normalized_shape[0])
    if isinstance(normalized_shape, int):
        return int(normalized_shape)
    trunk = getattr(visual, "trunk", None)
    num_features = getattr(trunk, "num_features", None)
    if num_features is not None:
        return int(num_features)
    projection = getattr(visual, "proj", None)
    if isinstance(projection, torch.Tensor) and projection.ndim == 2:
        return int(projection.shape[0])
    return fallback


class FashionSiglipAttributeClassifier(nn.Module):
    """Marqo FashionSigLIP image encoder plus conditional classification heads."""

    def __init__(
        self,
        vocabulary: AttributeVocabulary,
        *,
        model_name: str = "hf-hub:Marqo/marqo-fashionSigLIP",
        pretrained: str | None = None,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        train_vision: bool = False,
        unfreeze_last_blocks: int = 0,
        use_spatial_tokens: bool = False,
        spatial_residual: bool = False,
    ) -> None:
        super().__init__()
        import open_clip

        create_kwargs = {"pretrained": pretrained} if pretrained is not None else {}
        encoder, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(model_name, **create_kwargs)
        )
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = train_vision
        if not train_vision and unfreeze_last_blocks:
            self._unfreeze_last_vision_blocks(unfreeze_last_blocks)
        output_dim = _encoder_output_dim(self.encoder)
        self.use_spatial_tokens = use_spatial_tokens
        self.spatial_residual = spatial_residual
        if use_spatial_tokens:
            visual = self.encoder.visual
            trunk = getattr(visual, "trunk", None)
            if trunk is None or not all(
                hasattr(trunk, name) for name in ("forward_features", "forward_head")
            ):
                raise ValueError(
                    "Spatial heads require a visual trunk with forward_features and forward_head"
                )
            spatial_head_class = (
                SpatialResidualConditionalHeads
                if spatial_residual
                else SpatialConditionalAttributeHeads
            )
            self.heads = spatial_head_class(
                output_dim,
                _visual_patch_dim(visual, output_dim),
                vocabulary,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.heads = ConditionalAttributeHeads(
                output_dim,
                vocabulary,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

    def _unfreeze_last_vision_blocks(self, count: int) -> None:
        visual = self.encoder.visual
        blocks = getattr(getattr(visual, "trunk", visual), "blocks", None)
        if blocks is None:
            transformer = getattr(visual, "transformer", None)
            blocks = getattr(transformer, "resblocks", None)
        if blocks is None:
            raise ValueError("Cannot locate vision-transformer blocks for partial unfreezing")
        for block in list(blocks)[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for name in ("head", "proj", "ln_post"):
            module = getattr(visual, name, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        if self.use_spatial_tokens:
            visual = self.encoder.visual
            trunk = getattr(visual, "trunk", None)
            if trunk is None:
                raise ValueError("Spatial visual encoder has no accessible trunk")
            patch_tokens = trunk.forward_features(images)
            if patch_tokens.ndim == 4:
                patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
            elif patch_tokens.ndim != 3:
                raise ValueError(
                    "Spatial visual encoder returned unsupported token shape "
                    f"{tuple(patch_tokens.shape)}"
                )
            pooled = trunk.forward_head(patch_tokens)
            features = F.normalize(visual.head(pooled), dim=-1)
            prefix_count = int(getattr(trunk, "num_prefix_tokens", 1))
            patch_tokens = patch_tokens[:, prefix_count:]
            return self.heads(features, patch_tokens)
        features = self.encoder.encode_image(images, normalize=True)
        return self.heads(features)


def _asymmetric_multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma_negative: float = 4.0,
    gamma_positive: float = 0.0,
    clip: float = 0.05,
    positive_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Asymmetric focal loss for long-tailed multi-label values."""
    probabilities = torch.sigmoid(logits)
    positive = probabilities
    negative = 1.0 - probabilities
    if clip:
        negative = (negative + clip).clamp(max=1.0)
    positive_loss = targets * torch.log(positive.clamp_min(1e-8))
    if positive_weight is not None:
        positive_loss = positive_loss * positive_weight.to(
            device=targets.device, dtype=targets.dtype
        )
    loss = positive_loss
    loss += (1.0 - targets) * torch.log(negative.clamp_min(1e-8))
    if gamma_negative or gamma_positive:
        probability = positive * targets + negative * (1.0 - targets)
        gamma = gamma_positive * targets + gamma_negative * (1.0 - targets)
        loss *= torch.pow(1.0 - probability, gamma)
    return -loss.mean()


def conditional_attribute_loss(
    outputs: Mapping[str, Mapping[str, torch.Tensor]],
    targets: Mapping[str, Mapping[str, torch.Tensor]],
    vocabulary: AttributeVocabulary,
    *,
    applicability_pos_weight: Mapping[str, float] | None = None,
    applicability_field_weight: Mapping[str, float] | None = None,
    value_field_weight: Mapping[str, float] | None = None,
    multilabel_value_pos_weight: Mapping[str, torch.Tensor] | None = None,
    label_smoothing: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Macro-balance fields and mask value loss when a field is not applicable."""
    first_output = next(
        (
            tensor
            for section in ("categories", "applicability", "values")
            for tensor in outputs.get(section, {}).values()
        ),
        None,
    )
    if first_output is None:
        raise ValueError("Cannot compute conditional attribute loss for empty outputs")
    device = first_output.device
    losses: list[torch.Tensor] = []
    components: dict[str, float] = {}
    for field, logits in outputs["categories"].items():
        target = targets["categories"][field].to(device)
        supervision = targets.get("supervision", {}).get(field)
        mask = (
            supervision.to(device).bool()
            if supervision is not None
            else torch.ones_like(target, dtype=torch.bool)
        )
        mask &= target != -100
        loss = (
            F.cross_entropy(
                logits[mask],
                target[mask],
                label_smoothing=label_smoothing,
            )
            if mask.any()
            else logits.sum() * 0.0
        )
        losses.append(loss)
        components[f"category/{field}"] = float(loss.detach())

    multi_fields = set(vocabulary.multi_label_fields)
    unknown_applicability_weights = set(applicability_field_weight or {}) - set(
        vocabulary.fields
    )
    if unknown_applicability_weights:
        raise ValueError(
            "Applicability field weights contain unknown fields: "
            f"{sorted(unknown_applicability_weights)}"
        )
    unknown_value_weights = set(value_field_weight or {}) - set(vocabulary.fields)
    if unknown_value_weights:
        raise ValueError(
            "Value field weights contain unknown fields: "
            f"{sorted(unknown_value_weights)}"
        )
    for field, applicability_logits in outputs["applicability"].items():
        applicable = targets["applicability"][field].to(device).float()
        supervision = targets.get("supervision", {}).get(field)
        supervised_mask = (
            supervision.to(device).bool()
            if supervision is not None
            else torch.ones_like(applicable, dtype=torch.bool)
        )
        weight_value = (applicability_pos_weight or {}).get(field, 1.0)
        app_loss = (
            F.binary_cross_entropy_with_logits(
                applicability_logits[supervised_mask],
                applicable[supervised_mask],
                pos_weight=torch.tensor(weight_value, device=device),
            )
            if supervised_mask.any()
            else applicability_logits.sum() * 0.0
        )
        field_applicability_weight = float(
            (applicability_field_weight or {}).get(field, 1.0)
        )
        if field_applicability_weight <= 0.0:
            raise ValueError(f"Applicability field weight must be positive: {field!r}")
        app_loss = app_loss * field_applicability_weight
        positive_mask = supervised_mask & applicable.bool()
        if positive_mask.any():
            value_logits = outputs["values"][field][positive_mask]
            value_targets = targets["values"][field].to(device)[positive_mask]
            if field in multi_fields:
                value_loss = _asymmetric_multilabel_loss(
                    value_logits,
                    value_targets,
                    positive_weight=(multilabel_value_pos_weight or {}).get(field),
                )
            else:
                value_loss = F.cross_entropy(
                    value_logits,
                    value_targets.argmax(dim=-1),
                    label_smoothing=label_smoothing,
                )
        else:
            value_loss = outputs["values"][field].sum() * 0.0
        field_value_weight = float((value_field_weight or {}).get(field, 1.0))
        if field_value_weight <= 0.0:
            raise ValueError(f"Value field weight must be positive: {field!r}")
        value_loss = value_loss * field_value_weight
        field_loss = app_loss + value_loss
        losses.append(field_loss)
        components[f"applicability/{field}"] = float(app_loss.detach())
        components[f"value/{field}"] = float(value_loss.detach())
    total = torch.stack(losses).mean() if losses else torch.zeros((), device=device)
    components["total"] = float(total.detach())
    return total, components


def conditional_attribute_distillation_loss(
    student_outputs: Mapping[str, Mapping[str, torch.Tensor]],
    teacher_outputs: Mapping[str, Mapping[str, torch.Tensor]],
    vocabulary: AttributeVocabulary,
    *,
    temperature: float = 2.0,
    applicability_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match a teacher's full conditional distribution, macro-balanced by field.

    Categorical heads use temperature-scaled KL divergence. Applicability and
    multi-label value heads use soft binary cross entropy because their classes
    are independent. Averaging field losses prevents high-cardinality fields
    from overwhelming small but important attributes.
    """
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive")
    if applicability_weight <= 0:
        raise ValueError("Distillation applicability weight must be positive")
    if student_outputs.keys() != teacher_outputs.keys():
        raise ValueError("Student and teacher output sections do not match")

    scale = temperature**2
    losses: list[torch.Tensor] = []
    loss_weights: list[float] = []
    components: dict[str, float] = {}
    multi_fields = set(vocabulary.multi_label_fields)

    def categorical_kl(
        student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        return F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits.detach() / temperature, dim=-1),
            reduction="batchmean",
        ) * scale

    def binary_soft_ce(
        student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            student_logits / temperature,
            torch.sigmoid(teacher_logits.detach() / temperature),
        ) * scale

    for field, student_logits in student_outputs["categories"].items():
        teacher_logits = teacher_outputs["categories"].get(field)
        if teacher_logits is None:
            continue
        if teacher_logits.shape != student_logits.shape:
            raise ValueError(f"Teacher category head does not match {field!r}")
        loss = categorical_kl(student_logits, teacher_logits)
        losses.append(loss)
        loss_weights.append(1.0)
        components[f"category/{field}"] = float(loss.detach())

    for field, student_logits in student_outputs["applicability"].items():
        teacher_logits = teacher_outputs["applicability"].get(field)
        if teacher_logits is None:
            continue
        if teacher_logits.shape != student_logits.shape:
            raise ValueError(f"Teacher applicability head does not match {field!r}")
        loss = binary_soft_ce(student_logits, teacher_logits)
        losses.append(loss)
        loss_weights.append(applicability_weight)
        components[f"applicability/{field}"] = float(loss.detach())

    for field, student_logits in student_outputs["values"].items():
        teacher_logits = teacher_outputs["values"].get(field)
        if teacher_logits is None:
            continue
        if teacher_logits.shape != student_logits.shape:
            raise ValueError(f"Teacher value head does not match {field!r}")
        loss = (
            binary_soft_ce(student_logits, teacher_logits)
            if field in multi_fields
            else categorical_kl(student_logits, teacher_logits)
        )
        losses.append(loss)
        loss_weights.append(1.0)
        components[f"value/{field}"] = float(loss.detach())

    if not losses:
        raise ValueError("Cannot distill an empty conditional output contract")
    weights = torch.tensor(
        loss_weights, dtype=losses[0].dtype, device=losses[0].device
    )
    total = (torch.stack(losses) * weights).sum() / weights.sum()
    components["total"] = float(total.detach())
    return total, components
