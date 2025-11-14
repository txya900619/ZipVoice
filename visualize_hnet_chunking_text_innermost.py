#!/usr/bin/env python3
"""
Script to visualize how HNet chunks mel spectrograms.

This script observes the chunking behavior of the HNet backbone by:
1. Creating dummy mel spectrograms
2. Running them through the HNet backbone
3. Visualizing chunk boundaries and compression ratios
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchaudio
from hnet_impl import HNetConfig
from hnet_impl.modeling_hnet import HNet as HNetImpl

from zipvoice.models.hnet_tts_text_innermost import HNetTTS
from zipvoice.models.modules.utils import NJT
from zipvoice.tokenizer.tokenizer import LibriTTSTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank


def create_dummy_data(num_samples=2, text_length=10, mel_length=100):
    """Create dummy nested tensors for testing."""
    # Create dummy token IDs (nested tensor)
    iids_list = [torch.randint(0, 100, (text_length,)) for _ in range(num_samples)]
    iids = NJT(iids_list)

    # Create dummy mel spectrograms (nested tensor)
    mels_list = [torch.randn(mel_length, 100) for _ in range(num_samples)]
    mels = NJT(mels_list)

    return iids, mels


def load_real_data(
    wav_files,
    texts,
    tokenizer,
    feature_extractor,
    device,
    target_rms=0.1,
    feat_scale=0.1,
    sampling_rate=24000,
):
    """Load real audio files and extract mel spectrograms."""
    iids_list = []
    mels_list = []

    for wav_file, text in zip(wav_files, texts):
        # Tokenize text
        text_ids = tokenizer.texts_to_token_ids([text])[0]
        iids_list.append(torch.tensor(text_ids).to(device))

        # Load audio
        wav, prompt_sampling_rate = torchaudio.load(wav_file)

        # Resample if needed
        if prompt_sampling_rate != sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=prompt_sampling_rate, new_freq=sampling_rate
            )
            wav = resampler(wav)

        # Normalize RMS
        prompt_rms = torch.sqrt(torch.mean(torch.square(wav)))
        if prompt_rms < target_rms:
            wav = wav * target_rms / prompt_rms

        # Extract mel features
        mel_features = feature_extractor.extract(wav, sampling_rate=sampling_rate).to(
            device
        )

        # Scale features
        mel_features = mel_features * feat_scale

        mels_list.append(mel_features)

    # Convert to nested tensors
    iids = NJT(iids_list)
    mels = NJT(mels_list)

    return iids, mels


def visualize_chunking(
    hnet_backbone, mels_input, original_mels, text_condition, sample_idx=0
):
    """
    Visualize how HNet chunks mel spectrograms.

    Args:
        hnet_backbone: The HNet backbone model
        mels_input: Input mel spectrograms (after prenet, flattened)
        original_mels: Original mel spectrograms (before prenet)
        sample_idx: Which sample to visualize
    """
    print(f"\n{'=' * 60}")
    print(f"Visualizing chunking for sample {sample_idx}")
    print(f"{'=' * 60}\n")

    # Get single sample from original mels (before prenet)
    sample_mel = original_mels.unbind()[sample_idx]
    print(f"Original mel spectrogram shape: {sample_mel.shape}")

    # Hook to capture intermediate values
    all_layers_data = []

    def forward_hook_hnet(module, input, output):
        """Hook to capture HNetImpl forward pass details."""
        if hasattr(module, "routing_module"):
            with torch.no_grad():
                # HNetImpl.forward() takes (x_flat, flat_cu, msl) as input
                x_flat, flat_cu, msl, text_condition, text_condition_offsets = input

                # Run encoder
                r_flat = module.encoder(x_flat, flat_cu, msl)

                # Run routing module - returns (p_flat, b_flat, p_select_cu)
                p_flat, b_flat, p_select_cu = module.routing_module(r_flat, flat_cu)

                # Store routing information for this layer
                layer_data = {}
                layer_data["r"] = r_flat.detach().cpu()
                layer_data["bpred_p"] = p_flat.detach().cpu()
                layer_data["bpred_b"] = b_flat.detach().cpu()

                # Store encoder output length (this is what routing operates on)
                layer_data["encoder_output_length"] = r_flat.shape[0]
                # Store routing mask length
                layer_data["routing_mask_length"] = b_flat.shape[0]
                # Store encoder input length (needed for mapping back)
                layer_data["encoder_input_length"] = x_flat.shape[0]
                # Store layer index for debugging
                layer_data["layer_idx"] = len(all_layers_data)

                # Calculate chunk boundaries
                selected_indices = b_flat.nonzero(as_tuple=True)[0]
                layer_data["chunk_boundaries"] = selected_indices.detach().cpu()

                # Calculate compression ratio
                total_positions = b_flat.shape[0]
                selected_positions = b_flat.sum().item()
                layer_data["compression_ratio"] = selected_positions / total_positions

                all_layers_data.append(layer_data)

                print(f"\nLayer {len(all_layers_data)} Routing Module Output:")
                print(f"  - Total positions: {total_positions}")
                print(f"  - Selected positions: {selected_positions}")
                print(f"  - Compression ratio: {layer_data['compression_ratio']:.3f}")

    # Register hook
    hooks = []
    for name, module in hnet_backbone.named_modules():
        if isinstance(module, HNetImpl):
            hook = module.register_forward_hook(forward_hook_hnet)
            hooks.append(hook)

    # Run forward pass
    cu_s, msl = mels_input.offsets(), mels_input._get_max_seqlen()
    x_flat = mels_input.values()

    # Convert to bfloat16 as expected by HNet backbone
    x_flat = x_flat.bfloat16()

    with torch.no_grad():
        x_flat, extra = hnet_backbone(
            x_flat, cu_s, msl, text_condition.values(), text_condition.offsets()
        )

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    # Process all layers and map boundaries back to original mel
    # Note: layers are recorded innermost to outermost (innermost at index 0)
    all_captured_data = []

    if len(all_layers_data) > 0:
        original_length = sample_mel.shape[0]
        input_length = mels_input.unbind()[sample_idx].shape[0]

        # Process each layer from outermost to innermost
        for layer_idx in range(len(all_layers_data) - 1, -1, -1):
            captured_data = {}
            target_layer = all_layers_data[layer_idx]
            layer_name = f"Layer {len(all_layers_data) - layer_idx}"

            print(f"\n{'=' * 60}")
            print(f"Processing {layer_name}")
            print(f"{'=' * 60}")

            # Build mapping chain from outermost to this layer
            # Start with HNet input
            current_positions = torch.arange(input_length)
            print(f"Starting with HNet input length: {len(current_positions)}")

            # Apply routing from outermost to the layer BEFORE target_layer
            for intermediate_idx in range(len(all_layers_data) - 1, layer_idx, -1):
                intermediate_layer = all_layers_data[intermediate_idx]
                selected_mask = intermediate_layer["bpred_b"]

                print(
                    f"Applying Layer {len(all_layers_data) - intermediate_idx} routing: {len(current_positions)} -> ",
                    end="",
                )

                if len(current_positions) != len(selected_mask):
                    min_len = min(len(current_positions), len(selected_mask))
                    current_positions = current_positions[:min_len]
                    selected_mask = selected_mask[:min_len]

                current_positions = current_positions[selected_mask]
                print(f"{len(current_positions)} positions")

            # Now map target layer's boundaries
            target_boundaries = target_layer["chunk_boundaries"]
            target_selected_mask = target_layer["bpred_b"]

            print(f"{layer_name} boundaries count: {len(target_boundaries)}")
            print(f"{layer_name} selected mask length: {len(target_selected_mask)}")
            print(f"Current positions length: {len(current_positions)}")

            # Map target layer boundaries
            # target_boundaries are indices relative to the routing mask (i.e., positions_before_target)
            positions_before_target = current_positions

            # Ensure lengths match
            if len(positions_before_target) != len(target_selected_mask):
                min_len = min(len(positions_before_target), len(target_selected_mask))
                positions_before_target = positions_before_target[:min_len]
                target_selected_mask = target_selected_mask[:min_len]

            # Map target boundaries directly to positions
            if len(target_boundaries) > 0:
                valid_boundaries = target_boundaries[
                    target_boundaries < len(positions_before_target)
                ]
                mapped_boundaries_to_final = positions_before_target[valid_boundaries]
                print(f"Mapped boundaries to final: {mapped_boundaries_to_final}")
            else:
                mapped_boundaries_to_final = torch.tensor([], dtype=torch.long)

            print(f"Mapped boundaries count: {len(mapped_boundaries_to_final)}")

            # Extract mel-only positions from mapped_boundaries_to_final
            # In text innermost version, mels_input contains [mel_bos, mel[:-1]]
            # So mel_start_idx should be 1 (after mel_bos)
            mel_start_idx = 1
            mel_positions = mapped_boundaries_to_final[
                (mapped_boundaries_to_final >= mel_start_idx)
                & (mapped_boundaries_to_final < input_length)
            ]
            mel_indices = mel_positions - mel_start_idx

            print(
                f"Mel positions count: {len(mel_positions)}, mel_indices count: {len(mel_indices)}"
            )

            # Store captured data for this layer
            captured_data["chunk_boundaries"] = mel_indices
            captured_data["layer_name"] = layer_name
            captured_data["layer_idx"] = len(all_layers_data) - layer_idx

            print(
                f"Mapped {len(mel_indices)} boundaries to original {original_length} mel positions"
            )

            all_captured_data.append(captured_data)

    # Visualize chunking for all layers
    if all_captured_data:
        visualize_chunk_boundaries(
            sample_mel,
            all_captured_data,
            sample_idx,
        )

    return all_captured_data


def visualize_chunk_boundaries(mel, all_captured_data, sample_idx):
    """Create visualization of chunk boundaries overlaid on mel spectrogram for all layers."""
    from matplotlib.lines import Line2D

    # Create subplots for each layer
    n_layers = len(all_captured_data)
    fig, axes = plt.subplots(n_layers, 1, figsize=(15, 6 * n_layers))

    # If only one layer, make axes a list for consistent indexing
    if n_layers == 1:
        axes = [axes]

    # Convert mel to float for matplotlib compatibility
    mel_float = mel.float().cpu()

    # Define colors for different layers
    colors = ["red", "blue", "green", "orange", "purple"]

    # Plot each layer
    for idx, captured_data in enumerate(all_captured_data):
        ax = axes[idx]
        layer_name = captured_data["layer_name"]
        chunk_indices = captured_data["chunk_boundaries"]
        chunk_indices_cpu = chunk_indices.cpu()
        color = colors[idx % len(colors)]

        # Plot mel spectrogram
        im = ax.imshow(
            mel_float.T.numpy(), aspect="auto", origin="lower", cmap="viridis"
        )

        # Overlay chunk boundaries as vertical lines
        for boundary_idx in chunk_indices_cpu:
            ax.axvline(
                x=boundary_idx.item(),
                color=color,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
            )

        ax.set_title(f"{layer_name}: Mel Spectrogram with Chunk Boundaries")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Frequency Bins")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Magnitude")

        # Add legend for chunk boundaries
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"Chunk boundaries (n={len(chunk_indices_cpu)})",
            )
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    filename = f"hnet_chunking_sample_{sample_idx}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {filename}")
    plt.close()


def print_chunking_statistics(captured_data, stage_idx=0):
    """Print detailed chunking statistics."""
    if not captured_data:
        return

    separator = "=" * 60
    print(f"\n{separator}")
    print(f"Chunking Statistics (Stage {stage_idx})")
    print(separator)

    probs = captured_data["bpred_p"]
    chunk_indices = captured_data["chunk_boundaries"]
    compression_ratio = captured_data.get(
        "compression_ratio", len(chunk_indices) / len(probs)
    )

    print(f"\nTotal time steps: {len(probs)}")
    print(f"Selected chunks: {len(chunk_indices)}")
    print(f"Compression ratio: {compression_ratio:.3f}")
    print(f"\nChunk boundary indices: {chunk_indices.tolist()}")

    # Calculate average chunk size
    if len(chunk_indices) > 1:
        chunk_sizes = torch.diff(chunk_indices)
        print("\nChunk size statistics:")
        print(f"  - Min: {chunk_sizes.min().item()}")
        print(f"  - Max: {chunk_sizes.max().item()}")
        print(f"  - Mean: {chunk_sizes.float().mean().item():.2f}")
        print(f"  - Std: {chunk_sizes.float().std().item():.2f}")

    # Distribution of probabilities
    print("\nProbability distribution:")
    print(f"  - Min: {probs.min().item():.3f}")
    print(f"  - Max: {probs.max().item():.3f}")
    print(f"  - Mean: {probs.float().mean().item():.3f}")
    print(f"  - Std: {probs.float().std().item():.3f}")


def main():
    """Main function to observe HNet chunking."""
    parser = argparse.ArgumentParser(
        description="Visualize HNet chunking behavior",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="egs/h-net/exp/hnet_libritts_lr_5e-4",
        help="Path to experiment directory containing checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best-train-loss.pt",
        help="Checkpoint filename to load",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--wav-files",
        type=str,
        nargs="+",
        default=None,
        help="WAV files to load for visualization",
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        default=None,
        help="Corresponding texts for WAV files",
    )
    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="Target RMS for audio normalization",
    )
    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.4343,
        help="Feature scaling factor",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="Sampling rate for audio",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HNet Mel Spectrogram Chunking Visualization")
    print("=" * 60)

    # Load checkpoint
    exp_dir = Path(args.exp_dir)
    checkpoint_path = exp_dir / args.checkpoint
    model_config_path = exp_dir / "model.json"
    token_file = exp_dir / "tokens.txt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    if not token_file.exists():
        raise FileNotFoundError(f"Token file not found: {token_file}")

    print(f"\nLoading checkpoint from: {checkpoint_path}")
    print(f"Model config: {model_config_path}")
    print(f"Token file: {token_file}")

    # Load tokenizer
    tokenizer = LibriTTSTokenizer(token_file=token_file)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load model config
    config = HNetConfig.load_config(
        model_config_path,
        vocab_size=tokenizer.vocab_size,
    )

    # Create full HNetTTS model
    print("\nCreating HNetTTS model...")
    model = HNetTTS(config)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    load_checkpoint(filename=checkpoint_path, model=model, strict=True)

    # Move entire model to device
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Convert model to bfloat16 for HNet backbone compatibility
    model = model.to(torch.bfloat16)

    print(f"Using device: {device}")
    print("Model dtype: bfloat16")

    # Get the backbone for hook registration
    hnet_backbone = model.backbone

    # Load real data or create dummy data
    if args.wav_files and args.texts:
        if len(args.wav_files) != len(args.texts):
            raise ValueError("Number of WAV files must match number of texts")

        print("\nLoading real audio data...")
        feature_extractor = VocosFbank()

        # Convert wav files to Path objects
        wav_files = [Path(f) for f in args.wav_files]
        for wav_file in wav_files:
            if not wav_file.exists():
                raise FileNotFoundError(f"WAV file not found: {wav_file}")

        iids, mels = load_real_data(
            wav_files=wav_files,
            texts=args.texts,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=device,
            target_rms=args.target_rms,
            feat_scale=args.feat_scale,
            sampling_rate=args.sampling_rate,
        )

        print("\nData shapes:")
        print(f"  Token IDs: {iids.shape}")
        print(f"  Mel specs: {mels.shape}")
        print(f"  Number of samples: {len(iids)}")
    else:
        print("\nNo WAV files provided, creating dummy data...")
        print("(Use --wav-files and --texts to load real data)")
        iids, mels = create_dummy_data(num_samples=2, text_length=10, mel_length=200)
        iids = iids.to(device)
        mels = mels.to(device)

        print("\nData shapes:")
        print(f"  Token IDs: {iids.shape}")
        print(f"  Mel specs: {mels.shape}")

    # Prepare input similar to HNetTTS forward method
    # Line 124-132 in hnet_tts_text_innermost.py
    text_condition = model.embeddings(iids)

    # Convert mels to bfloat16 before mel_prenet
    mels_bfloat16 = NJT([m.bfloat16() for m in mels.unbind()])

    mels_input = model.mel_prenet(mels_bfloat16)
    mels_input = model.dropout(mels_input)

    mels_input = NJT(
        [torch.cat((model.mel_bos, m_i[:-1]), dim=0) for m_i in mels_input.unbind()]
    )

    print(f"\nPrepared input shape: {mels_input.shape}")

    # Visualize chunking for each sample
    all_samples_data = []
    for sample_idx in range(len(mels_input)):
        sample_layers_data = visualize_chunking(
            hnet_backbone, mels_input, mels, text_condition, sample_idx
        )
        all_samples_data.append(sample_layers_data)

        # Print statistics for all layers of this sample
        if sample_layers_data:
            for layer_data in sample_layers_data:
                layer_name = layer_data.get("layer_name", "Unknown Layer")
                print(f"\n{'=' * 60}")
                print(f"Sample {sample_idx} - {layer_name} Statistics")
                print(f"{'=' * 60}")
                print(f"Chunk boundaries: {len(layer_data['chunk_boundaries'])}")
                if len(layer_data["chunk_boundaries"]) <= 20:
                    print(
                        f"Chunk boundary indices: {layer_data['chunk_boundaries'].tolist()}"
                    )
                else:
                    print(
                        f"Chunk boundary indices (first 20): {layer_data['chunk_boundaries'][:20].tolist()}..."
                    )

    separator = "=" * 60
    print(f"\n{separator}")
    print("Visualization complete!")
    print(f"{separator}\n")


if __name__ == "__main__":
    main()
