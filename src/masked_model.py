import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from copy import deepcopy

# Define layers we want to prune
PRUNE_LAYERS = (nn.Linear, nn.Conv3d)

class MultimodalSNIPMask(nn.Module):
    """
    Parametrization module that applies a modality-specific binary mask to weights.
    Registered via torch.nn.utils.parametrize.
    """
    def __init__(self, masks_dict):
        super().__init__()
        # Register masks as buffers (saved in state_dict, no gradients)
        self.keys = sorted(list(masks_dict.keys()))
        for mod_id, mask_tensor in masks_dict.items():
            self.register_buffer(f'mask_{mod_id}', mask_tensor)
        
        # State to control which mask is active. None = Identity (no mask)
        self.active_mod_id = None

    def forward(self, weight):
        if self.active_mod_id is None:
            return weight
        # Dynamic retrieval of the buffer corresponding to the active modality
        mask = getattr(self, f'mask_{self.active_mod_id}')
        return weight * mask

class MultiMaskSNIPWrapper(nn.Module):
    """
    Wrapper that implements Multimodal SNIP pruning.
    It generates unique pruning masks for different modalities and applies them dynamically.
    """
    def __init__(self, model, sparsity=0.9):
        super(MultiMaskSNIPWrapper, self).__init__()
        self.model = model
        self.sparsity = sparsity
        self.masks_registered = False

    def register_multimodal_masks(self, modalities, input_data, labels, sub_batch_size=4):
        """
        Initialization step (Run ONCE before training):
        1. Creates a temporary GPU copy of the model for SNIP calculation.
        2. Generates masks for each modality found in the input.
        3. Processes data in sub-batches to stay within VRAM limits.
        4. Registers the masks as parametrizations on the main model.

        Args:
            modalities: Tensor of modality codes for each sample
            input_data: Tensor of input volumes
            labels: Tensor of labels
            sub_batch_size: Number of volumes to process at once (default: 4)
        """
        # Determine target device (GPU the main model is on)
        target_device = next(iter(self.model.parameters())).device

        # Create temp model on GPU (not CPU to avoid system RAM spike)
        temp_model = deepcopy(self.model).to(target_device)
        temp_optimizer = torch.optim.SGD(temp_model.parameters(), 0.1)

        # 1. Generate Masks Dictionary: {mod_id: {layer_name: mask}}
        temp_mask_storage = {}
        unique_modalities = torch.unique(modalities).cpu().detach().tolist()

        for mod in unique_modalities:
            print(f"Generating SNIP masks for modality: {mod}")
            mask_idx = (modalities == mod)
            mod_data = input_data[mask_idx]
            mod_labels = labels[mask_idx]

            # Calculate scores using sub-batches to manage VRAM
            masks_by_name = self._generate_mask_from_grad_scores_batched(
                temp_model, temp_optimizer, mod_data, mod_labels,
                target_device, sub_batch_size=sub_batch_size
            )
            temp_mask_storage[mod] = masks_by_name

        # 2. Register Parametrizations on the actual model
        print("Registering Parametrizations...")
        for name, module in self.model.named_modules():
            if isinstance(module, PRUNE_LAYERS):
                layer_masks = {}
                has_masks = False
                for mod, mask_dict in temp_mask_storage.items():
                    if name in mask_dict:
                        layer_masks[mod] = mask_dict[name]
                        has_masks = True

                if has_masks:
                    snip_mask_module = MultimodalSNIPMask(layer_masks)
                    parametrize.register_parametrization(module, "weight", snip_mask_module)

        self.masks_registered = True

        # Cleanup to free GPU memory
        del temp_model
        del temp_optimizer
        if target_device.type == 'cuda':
            torch.cuda.empty_cache()
        print("Mask initialization complete. Temporary GPU model cleared.")

    def forward(self, input_data, modalities):
        if not self.masks_registered:
            return self.model(input_data)
        
        device = next(iter(self.model.parameters())).device 
        input_device = input_data.device
        assert device == input_device, f"Input data and model must be on the same device, got model device {device} and input device {input_device}"

        batch_size = input_data.shape[0]
        
        # Output container (Assuming Binary Classification [B, 1])
        final_outputs = torch.zeros(batch_size, 1, device=device) 
        
        unique_mods = torch.unique(modalities).cpu().tolist()

        for mod in unique_mods:
            mod_idx = (modalities == mod)
            sub_data = input_data[mod_idx]
            
            # A. Set the Active Modality
            self._set_active_modality(mod)
            
            # B. Forward Pass (Autograd tracks: output = weight * mask_mod)
            sub_output = self.model(sub_data)
            final_outputs[mod_idx] = sub_output
            
        # C. Reset to Identity (No mask)
        self._set_active_modality(None)
        
        return final_outputs

    def _set_active_modality(self, mod_id):
        """Iterates over modules to toggle the active mask state."""
        for module in self.model.modules():
            if parametrize.is_parametrized(module, "weight"):
                for param_module in module.parametrizations.weight:
                    if isinstance(param_module, MultimodalSNIPMask):
                        param_module.active_mod_id = mod_id

    def prepare_for_loading(self, modalities_list):
        """
        Pre-initializes structure for loading state_dict.
        Call this BEFORE loading a checkpoint.
        """
        print(f"Restoring parametrization structure for modalities: {modalities_list}")
        for name, module in self.model.named_modules():
            # Only process if it's a target layer and NOT already parametrized
            if isinstance(module, PRUNE_LAYERS) and not parametrize.is_parametrized(module, "weight"):
                # Get the actual shape of the weights for this specific layer
                weight_shape = module.weight.shape
                # Create dummy masks matching that shape
                dummy_masks = {
                    mod: torch.ones(weight_shape) 
                    for mod in modalities_list
                }
                # Register the parametrization
                snip_mask_module = MultimodalSNIPMask(dummy_masks)
                parametrize.register_parametrization(module, "weight", snip_mask_module)
        
        self.masks_registered = True

    # --- INTERNAL SNIP HELPERS ---
    def _generate_mask_from_grad_scores_batched(self, model, optimizer, data, labels, target_device, sub_batch_size=4):
        """
        Generate masks by processing data in sub-batches to manage VRAM.
        Accumulates scores across all sub-batches before computing threshold.
        """
        accumulated_scores = {}
        n_samples = data.shape[0]

        # Safety check: if no samples, return empty masks
        if n_samples == 0:
            return {}

        model.train()

        for start_idx in range(0, n_samples, sub_batch_size):
            end_idx = min(start_idx + sub_batch_size, n_samples)
            sub_data = data[start_idx:end_idx].to(target_device)
            sub_labels = labels[start_idx:end_idx].to(target_device)

            optimizer.zero_grad()
            preds = model(sub_data)
            loss = F.binary_cross_entropy_with_logits(preds, sub_labels.float())
            loss.backward()

            # Accumulate scores (sum of absolute values across sub-batches)
            for name, module in model.named_modules():
                if isinstance(module, PRUNE_LAYERS) and module.weight.grad is not None:
                    score = (module.weight.grad * module.weight.data).abs()
                    if name not in accumulated_scores:
                        accumulated_scores[name] = score.clone()
                    else:
                        accumulated_scores[name] += score

            # Clear GPU cache periodically
            if target_device.type == 'cuda':
                torch.cuda.empty_cache()

        # Compute threshold from accumulated scores
        threshold = self._get_threshold_from_scores(accumulated_scores)

        # Generate masks
        masks = {}
        for name, values in accumulated_scores.items():
            masks[name] = (values > threshold).float().to(target_device)
        return masks

    def _generate_mask_from_grad_scores(self, model, optimizer, batch, target_device):
        """Legacy method - kept for backward compatibility."""
        scores_dict = self._calculate_scores(model, optimizer, batch)
        threshold = self._get_threshold_from_scores(scores_dict)
        masks = {}
        for name, values in scores_dict.items():
            masks[name] = (values > threshold).float().to(target_device)
        return masks
    def _calculate_scores(self, model, optimizer, batch):
        """Calculate SNIP scores - data should already be on the correct device."""
        data, labels = batch
        # Note: data should already be on target_device (GPU)
        model.train()
        optimizer.zero_grad()
        preds = model(data)
        loss = F.binary_cross_entropy_with_logits(preds, labels.float())
        loss.backward()
        scores_d = {}
        for name, module in model.named_modules():
            if isinstance(module, PRUNE_LAYERS) and module.weight.grad is not None:
                # SNIP score = |grad * weight|
                scores_d[name] = (module.weight.grad * module.weight.data).abs()
        return scores_d

    def _get_threshold_from_scores(self, scores_d):
        global_scores = torch.cat([torch.flatten(x) for x in scores_d.values()])
        num_params_to_keep = int(len(global_scores) * (1.0 - self.sparsity))
        if num_params_to_keep < 1: num_params_to_keep = 1
        topk_scores, _ = torch.topk(global_scores, num_params_to_keep, sorted=True)
        return topk_scores[-1]