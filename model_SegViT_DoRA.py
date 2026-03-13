"""
Complete SegViT Implementation
Based on "SegViT: Semantic Segmentation with Plain Vision Transformers"
and "SegViTv2: Exploring Efficient and Continual Semantic Segmentation with Plain Vision Transformers"

This implementation includes:
- Cascaded ATM (Attention-to-Mask) heads
- Multi-scale feature extraction
- Query refinement across stages
- Optional Query-based Downsampling (QD) for efficiency
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional, Tuple
from peft import LoraConfig, get_peft_model


class ATMHead(nn.Module):
    """
    Attention-to-Mask (ATM) Head

    Implements the core ATM mechanism with:
    - Self-attention on class queries (optional, SegViT v1 style)
    - Cross-attention between queries and spatial features
    - Feed-forward network
    - Mask and class prediction heads
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 150,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_self_attn: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_self_attn = use_self_attn

        # Learnable class queries (initialized per instance, not shared across stages)
        self.query_embed = nn.Parameter(torch.zeros(1, num_classes, embed_dim))

        # Self-Attention on queries (SegViT v1)
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.dropout1 = nn.Dropout(dropout)

        # Cross-Attention (Queries attend to image features)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-Forward Network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        # Mask Prediction Head (Equation 3 in paper)
        self.mask_proj = nn.Linear(embed_dim, embed_dim)

        # Class Prediction Head (Equation 4 in paper)
        self.cls_head = nn.Linear(embed_dim, 1)

        # Initialize parameters
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following common practice"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        features: torch.Tensor,
        prev_queries: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Spatial features [B, C, H, W]
            prev_queries: Refined queries from previous stage [B, N, C]

        Returns:
            mask_logits: [B, num_classes, H, W]
            cls_logits: [B, num_classes]
            refined_queries: [B, num_classes, C] for next stage
        """
        B, C, H, W = features.shape

        # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        memory = features.flatten(2).transpose(1, 2)

        # Initialize or use previous queries
        if prev_queries is None:
            queries = self.query_embed.expand(B, -1, -1)  # [B, num_classes, embed_dim]
        else:
            queries = prev_queries

        # --- Stage 1: Self-Attention (Optional) ---
        if self.use_self_attn:
            attn_out, _ = self.self_attn(queries, queries, queries)
            queries = queries + self.dropout1(attn_out)
            queries = self.norm1(queries)

        # --- Stage 2: Cross-Attention ---
        # Queries attend to image features
        attn_out, attn_weights = self.cross_attn(
            query=queries,
            key=memory,
            value=memory
        )
        queries = queries + self.dropout2(attn_out)
        queries = self.norm2(queries)

        # --- Stage 3: Feed-Forward Network ---
        ffn_out = self.ffn(queries)
        queries = queries + ffn_out
        queries = self.norm3(queries)

        # --- Stage 4: Mask Prediction (Equation 3) ---
        # Project queries for mask generation
        mask_embed = self.mask_proj(queries)  # [B, num_classes, embed_dim]

        # Compute similarity between mask embeddings and spatial features
        # This is the core "Attention-to-Mask" operation
        mask_logits = torch.einsum("bnc,blc->bnl", mask_embed, memory)
        mask_logits = mask_logits.view(B, self.num_classes, H, W)

        # --- Stage 5: Class Prediction (Equation 4) ---
        # Predict whether each class is present in the image
        cls_logits = self.cls_head(queries).squeeze(-1)  # [B, num_classes]

        return mask_logits, cls_logits, queries


class QueryBasedDownsampling(nn.Module):
    """
    Query-based Downsampling (QD) module from SegViT Shrunk

    Reduces spatial resolution using cross-attention mechanism.
    More effective than simple pooling for semantic segmentation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_queries: int = 256,  # Number of downsampled tokens
        num_heads: int = 8
    ):
        super().__init__()
        self.num_queries = num_queries

        # Learnable downsampling queries
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, in_dim))

        # Cross-attention for downsampling
        self.cross_attn = nn.MultiheadAttention(
            in_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Optional projection to different dimension
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.proj = nn.Identity()

        self.norm = nn.LayerNorm(in_dim)

        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C] spatial features

        Returns:
            [B, num_queries, C_out] downsampled features
        """
        B, L, C = x.shape

        # Expand queries for batch
        queries = self.query_tokens.expand(B, -1, -1)

        # Cross-attention: queries attend to all spatial positions
        out, _ = self.cross_attn(query=queries, key=x, value=x)
        out = self.norm(out)
        out = self.proj(out)

        return out


class QueryBasedUpsampling(nn.Module):
    """
    Query-based Upsampling (QU) module from SegViT Shrunk

    Recovers spatial resolution and preserves low-level features.
    Runs in parallel to backbone's later layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        target_size: int,  # Target spatial size (H*W)
        num_heads: int = 8
    ):
        super().__init__()
        self.target_size = target_size

        # Learnable upsampling queries (one per target position)
        self.query_tokens = nn.Parameter(torch.zeros(1, target_size, in_dim))

        # Cross-attention for upsampling
        self.cross_attn = nn.MultiheadAttention(
            in_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Projection
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.proj = nn.Identity()

        self.norm = nn.LayerNorm(in_dim)

        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L_down, C] downsampled features

        Returns:
            [B, target_size, C_out] upsampled features
        """
        B = x.shape[0]

        # Expand queries
        queries = self.query_tokens.expand(B, -1, -1)

        # Cross-attention: each target position attends to downsampled features
        out, _ = self.cross_attn(query=queries, key=x, value=x)
        out = self.norm(out)
        out = self.proj(out)

        return out


class CascadedATMDecoder(nn.Module):
    """
    Cascaded ATM Decoder

    Multi-stage decoder where:
    1. Each stage refines queries using features from different backbone layers
    2. Refined queries from stage i become input to stage i+1
    3. Final stage produces segmentation masks
    """

    def __init__(
        self,
        backbone_dim: int = 1280,
        embed_dim: int = 512,
        num_classes: int = 150,
        num_stages: int = 3,
        num_heads: int = 8,
        use_self_attn: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_stages = num_stages
        self.embed_dim = embed_dim

        # Projection layers to map backbone features to embed_dim
        self.feature_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            for _ in range(num_stages)
        ])

        # Cascaded ATM heads
        self.atm_heads = nn.ModuleList([
            ATMHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                num_heads=num_heads,
                use_self_attn=use_self_attn,
                dropout=dropout
            )
            for _ in range(num_stages)
        ])

        # Optional: Auxiliary loss heads for intermediate stages
        self.use_aux_loss = True

    def forward(
        self,
        feature_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        """
        Args:
            feature_list: List of [B, L, C] features from different backbone layers

        Returns:
            final_mask_logits: [B, num_classes, H, W]
            final_cls_logits: [B, num_classes]
            aux_outputs: List of (mask_logits, cls_logits) from intermediate stages (if training)
        """
        assert len(feature_list) == self.num_stages, \
            f"Expected {self.num_stages} features, got {len(feature_list)}"

        queries = None
        aux_outputs = []

        # Cascaded refinement through stages
        for stage_idx, (features, proj, atm_head) in enumerate(
            zip(feature_list, self.feature_projs, self.atm_heads)
        ):
            B, L, C = features.shape
            H = W = int(math.sqrt(L))

            # Project features to common dimension
            features_proj = proj(features)  # [B, L, embed_dim]

            # Reshape to spatial format for ATM head
            features_spatial = features_proj.transpose(1, 2).reshape(B, self.embed_dim, H, W)

            # Run ATM head
            mask_logits, cls_logits, queries = atm_head(features_spatial, queries)

            # Store intermediate outputs for auxiliary loss
            if self.training and self.use_aux_loss and stage_idx < self.num_stages - 1:
                aux_outputs.append((mask_logits, cls_logits))

        # Return final stage output
        if self.training and self.use_aux_loss:
            return mask_logits, cls_logits, aux_outputs
        else:
            return mask_logits, cls_logits, None


class SegViT(nn.Module):
    """
    Complete SegViT Model

    Combines:
    - Vision Transformer backbone (e.g., Virchow2)
    - Multi-scale feature extraction via hooks
    - Cascaded ATM decoder
    """
    def __init__(
        self,
        backbone_name: str = "hf-hub:paige-ai/Virchow2",
        num_classes: int = 4,
        embed_dim: int = 512,
        num_stages: int = 3,
        use_lora: bool = True,
        use_dora: bool = True,  # <-- NEW: Flag for DoRA
        lora_rank: int = 16,
        hook_indices: list = [7, 15, 23],
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.patch_size = 14
        self.hook_indices = hook_indices[:num_stages]
        self.hidden_states = {}

        # 1. Load the Base Backbone (Virchow)
        print(f"Loading backbone: {backbone_name}")
        #base_backbone = timm.create_model( backbone_name,pretrained=True,mlp_layer=timm.layers.SwiGLUPacked,
        #    act_layer=torch.nn.SiLU,num_classes=0)
        base_backbone = self._create_backbone(backbone_name, pretrained=True)

        if hasattr(base_backbone, "enable_input_require_grads"):
          base_backbone.enable_input_require_grads()

        # 2. Apply DoRA/LoRA
        if use_lora:
            print(f"Applying {'DoRA' if use_dora else 'LoRA'} (rank={lora_rank}) to backbone...")
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                target_modules=["qkv", "fc1", "fc2", "proj"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=[],
                use_dora=use_dora  # <-- NEW: Enables Weight-Decomposed LoRA
            )
            self.backbone = get_peft_model(base_backbone, config)
            self.backbone.print_trainable_parameters()
        else:
            self.backbone = base_backbone

        # 3. Get embedding dimension
        if hasattr(self.backbone, "base_model"):
            self.backbone_dim = self.backbone.base_model.model.embed_dim
        else:
            self.backbone_dim = self.backbone.embed_dim

        # 4. Register Hooks
        self._register_hooks()

        # 5. Initialize Decoder
        self.decoder = CascadedATMDecoder(
            backbone_dim=self.backbone_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_stages=num_stages,
        )

        print(f"SegViT initialized:")
        print(f"  - Backbone: {backbone_name} (dim={self.backbone_dim})")
        print(f"  - Hook layers: {self.hook_indices}")
        print(f"  - Decoder stages: {num_stages}")
        print(f"  - Classes: {num_classes}")

    def _create_backbone(self, backbone_name: str, pretrained: bool):
        """Create backbone model"""
        if "Virchow" in backbone_name:
            backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU
            )
        else:
            backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0  # Remove classification head
            )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(backbone, 'set_grad_checkpointing'):
            backbone.set_grad_checkpointing(True)

        return backbone

    def _get_backbone_dim(self) -> int:
        """Get embedding dimension from backbone"""
        if hasattr(self.backbone, 'embed_dim'):
            return self.backbone.embed_dim
        elif hasattr(self.backbone, 'num_features'):
            return self.backbone.num_features
        else:
            # Default for common ViT sizes
            return 1280

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features"""
        def get_hook(layer_idx):
            def hook(module, input, output):
                self.hidden_states[layer_idx] = output
            return hook

        # Register hooks on transformer blocks
        for idx in self.hook_indices:
            if idx < len(self.backbone.blocks):
                self.backbone.blocks[idx].register_forward_hook(get_hook(idx))
            else:
                raise ValueError(f"Hook index {idx} exceeds backbone depth {len(self.backbone.blocks)}")

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from backbone.
        Forces execution in Float32 to avoid AMP type mismatch errors in PatchEmbed.
        """
        # 1. Handle MONAI wrapper if present
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()
        elif isinstance(x, type(x)) and "monai" in str(type(x)):
            x = x.detach().clone()

        # 2. Force input to Float32 immediately
        x = x.float()

        B, _, H_img, W_img = x.shape

        # Clear previous hooks
        self.hidden_states = {}

        # 3. Disable Autocast just for the backbone
        # This forces the convolution to run in FP32 (Input Float + Bias Float),
        # which guarantees type matching.
        with torch.amp.autocast('cuda', enabled=False):
            _ = self.backbone.forward_features(x)

        # Collect hooked features
        feature_list = []
        h_feat = H_img // self.patch_size
        w_feat = W_img // self.patch_size
        num_spatial = h_feat * w_feat

        for idx in self.hook_indices:
            feat = self.hidden_states[idx]  # [B, N, C]

            # Remove CLS token and any register tokens
            # Assume spatial tokens are always at the end
            feat = feat[:, -num_spatial:, :]

            feature_list.append(feat)

        return feature_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            If training:
                logits: [B, num_classes, H, W]
                cls_logits: [B, num_classes]
                aux_outputs: List of auxiliary outputs
            If inference:
                logits: [B, num_classes, H, W] (with class confidence applied)
        """
        B, _, H_img, W_img = x.shape

        # Extract multi-scale features
        feature_list = self.extract_features(x)

        # Decode with cascaded ATM
        mask_logits, cls_logits, aux_outputs = self.decoder(feature_list)

        # Upsample to input resolution
        _, _, H_mask, W_mask = mask_logits.shape
        if H_mask != H_img or W_mask != W_img:
            mask_logits = F.interpolate(
                mask_logits,
                size=(H_img, W_img),
                mode='bilinear',
                align_corners=False
            )

        # Training: return logits and auxiliary outputs
        if self.training:
            # Also upsample auxiliary outputs
            if aux_outputs is not None:
                aux_outputs_upsampled = []
                for aux_mask, aux_cls in aux_outputs:
                    aux_mask_up = F.interpolate(
                        aux_mask,
                        size=(H_img, W_img),
                        mode='bilinear',
                        align_corners=False
                    )
                    aux_outputs_upsampled.append((aux_mask_up, aux_cls))
                aux_outputs = aux_outputs_upsampled

            return mask_logits, cls_logits, aux_outputs

        # Inference: apply class confidence (Equation 4)
        else:
            cls_probs = torch.sigmoid(cls_logits)  # [B, num_classes]
            cls_probs = cls_probs.unsqueeze(-1).unsqueeze(-1)  # [B, num_classes, 1, 1]

            # Element-wise multiplication (mask refinement)
            refined_logits = mask_logits * cls_probs

            return refined_logits


class SegViTLoss(nn.Module):
    """
    Loss function for SegViT

    Combines:
    - Mask prediction loss (typically Cross-Entropy or Dice)
    - Class prediction loss (Binary Cross-Entropy)
    - Auxiliary losses from intermediate stages
    """

    def __init__(
        self,
        num_classes: int,
        aux_weight: float = 0.4,
        cls_weight: float = 0.5,
        use_dice: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.aux_weight = aux_weight
        self.cls_weight = cls_weight
        self.use_dice = use_dice

        # Mask loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

        # Class loss
        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
      pred = torch.softmax(pred, dim=1)
      intersection = 0.0
      union = 0.0

      # Process sequentially to save memory
      for c in range(self.num_classes):
          target_c = (target == c).float()
          pred_c = pred[:, c, :, :]

          intersection += (pred_c * target_c).sum(dim=(1, 2))
          union += pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

      dice = (2.0 * intersection + smooth) / (union + smooth)
      return 1.0 - dice.mean()

    def forward(
        self,
        mask_logits: torch.Tensor,
        cls_logits: torch.Tensor,
        targets: torch.Tensor,
        aux_outputs: Optional[List] = None
    ) -> dict:
        """
        Args:
            mask_logits: [B, num_classes, H, W]
            cls_logits: [B, num_classes]
            targets: [B, H, W] ground truth labels
            aux_outputs: List of (mask_logits, cls_logits) from intermediate stages

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Main mask loss
        mask_loss = self.ce_loss(mask_logits, targets)
        if self.use_dice:
            mask_loss = mask_loss + self.dice_loss(mask_logits, targets)
        losses['mask_loss'] = mask_loss

        # Class loss (multi-label classification)
        # Create class labels from segmentation targets
        B = targets.shape[0]
        cls_targets = torch.zeros(B, self.num_classes, device=targets.device)
        for b in range(B):
            unique_classes = targets[b].unique()
            unique_classes = unique_classes[unique_classes != 255]  # Ignore index
            cls_targets[b, unique_classes] = 1.0

        cls_loss = self.bce_loss(cls_logits, cls_targets)
        losses['cls_loss'] = cls_loss

        # Auxiliary losses
        if aux_outputs is not None:
            aux_loss = 0.0
            for aux_mask, aux_cls in aux_outputs:
                aux_mask_loss = self.ce_loss(aux_mask, targets)
                if self.use_dice:
                    aux_mask_loss = aux_mask_loss + self.dice_loss(aux_mask, targets)
                aux_cls_loss = self.bce_loss(aux_cls, cls_targets)
                aux_loss += (aux_mask_loss + self.cls_weight * aux_cls_loss)

            aux_loss = aux_loss / len(aux_outputs)
            losses['aux_loss'] = aux_loss

        # Total loss
        total_loss = mask_loss + self.cls_weight * cls_loss
        if aux_outputs is not None:
            total_loss = total_loss + self.aux_weight * aux_loss

        losses['total_loss'] = total_loss

        return losses
