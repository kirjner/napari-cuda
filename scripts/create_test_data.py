#!/usr/bin/env python
"""
Create test data for napari-cuda development and testing.

Generates various types of volumetric data for testing GPU rendering:
- 3D sinusoidal patterns with time evolution
- Large multi-channel volumes
- Synthetic microscopy-like data
"""

import os
from pathlib import Path

import numpy as np


def create_sinusoidal_volume(shape=(100, 256, 256), save_path="data/test_volume.npy"):
    """Create a 3D sinusoidal volume with evolving patterns."""
    print(f"Creating sinusoidal volume {shape}...")

    z_slices, height, width = shape
    x = np.linspace(-3, 3, width)
    y = np.linspace(-3, 3, height)
    z = np.linspace(-3, 3, z_slices)

    X, Y = np.meshgrid(x, y)

    data = []
    for z_val in z:
        # Create evolving sinusoidal pattern
        frame = np.sin(np.sqrt(X**2 + Y**2) - z_val*0.5) * np.exp(-0.1*(X**2 + Y**2))
        # Add some noise for realism
        frame += np.random.normal(0, 0.02, frame.shape)
        data.append(frame)

    data = np.array(data, dtype=np.float32)

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, data)

    size_mb = data.nbytes / (1024**2)
    print(f"  ✓ Created: {save_path} - shape={data.shape}, size={size_mb:.1f} MB")
    return data


def create_large_volume(shape=(512, 512, 512), save_path="data/large_volume.npy"):
    """Create a large volume for stress testing."""
    print(f"Creating large volume {shape}...")

    # Use memmap for very large volumes to avoid memory issues
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    data = np.memmap(save_path, dtype=np.float32, mode='w+', shape=shape)

    # Fill with Perlin-noise-like pattern (simplified)
    print("  Generating noise patterns...")
    for z in range(0, shape[0], 32):
        z_end = min(z + 32, shape[0])
        chunk = np.random.randn(z_end - z, shape[1], shape[2])

        # Smooth with convolution for more realistic patterns
        from scipy.ndimage import gaussian_filter
        chunk = gaussian_filter(chunk, sigma=2.0)

        data[z:z_end] = chunk.astype(np.float32)

        if z % 64 == 0:
            print(f"    Progress: {z}/{shape[0]} slices")

    del data  # Flush to disk

    size_mb = np.prod(shape) * 4 / (1024**2)  # float32 = 4 bytes
    print(f"  ✓ Created: {save_path} - shape={shape}, size={size_mb:.1f} MB")


def create_multichannel_volume(shape=(3, 100, 256, 256), save_path="data/multichannel.npy"):
    """Create multi-channel volume (e.g., RGB or multi-label)."""
    print(f"Creating multi-channel volume {shape}...")

    channels, z_slices, height, width = shape
    data = np.zeros(shape, dtype=np.float32)

    # Create different patterns for each channel
    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    X, Y = np.meshgrid(x, y)

    for c in range(channels):
        for z in range(z_slices):
            if c == 0:  # Red channel - circular patterns
                data[c, z] = np.sin(5 * np.sqrt(X**2 + Y**2) - z*0.1)
            elif c == 1:  # Green channel - vertical waves
                data[c, z] = np.sin(5 * X + z*0.1) * np.exp(-0.1*Y**2)
            else:  # Blue channel - diagonal patterns
                data[c, z] = np.sin(3 * (X + Y) - z*0.1) * np.exp(-0.05*(X**2 + Y**2))

    # Normalize to [0, 1]
    for c in range(channels):
        data[c] = (data[c] - data[c].min()) / (data[c].max() - data[c].min())

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, data)

    size_mb = data.nbytes / (1024**2)
    print(f"  ✓ Created: {save_path} - shape={data.shape}, size={size_mb:.1f} MB")
    return data


def create_time_series(shape=(50, 100, 256, 256), save_path="data/timeseries.npy"):
    """Create time-series volume for testing temporal navigation."""
    print(f"Creating time-series volume {shape}...")

    time_points, z_slices, height, width = shape
    data = np.zeros(shape, dtype=np.float32)

    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    X, Y = np.meshgrid(x, y)

    for t in range(time_points):
        # Evolving pattern over time
        phase = 2 * np.pi * t / time_points

        for z in range(z_slices):
            z_phase = np.pi * z / z_slices

            # Create moving wave pattern
            data[t, z] = np.sin(3 * X + phase) * np.cos(3 * Y + z_phase) * \
                        np.exp(-0.1 * (X**2 + Y**2))

    # Add some structure
    data += 0.5
    data = np.clip(data, 0, 1)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, data)

    size_mb = data.nbytes / (1024**2)
    print(f"  ✓ Created: {save_path} - shape={data.shape}, size={size_mb:.1f} MB")
    return data


def create_synthetic_cells(shape=(50, 256, 256), save_path="data/synthetic_cells.npy"):
    """Create synthetic cell-like structures for testing segmentation views."""
    print(f"Creating synthetic cell data {shape}...")

    z_slices, height, width = shape
    data = np.zeros(shape, dtype=np.float32)

    # Create random cell centers
    n_cells = 50
    cell_centers = []

    for _ in range(n_cells):
        center = [
            np.random.randint(10, z_slices - 10),
            np.random.randint(20, height - 20),
            np.random.randint(20, width - 20)
        ]
        cell_centers.append(center)

    # Create gaussian blobs for each cell
    for cz, cy, cx in cell_centers:
        size = np.random.uniform(5, 15)
        intensity = np.random.uniform(0.5, 1.0)

        z_range = slice(max(0, cz - 20), min(z_slices, cz + 20))
        y_range = slice(max(0, cy - 30), min(height, cy + 30))
        x_range = slice(max(0, cx - 30), min(width, cx + 30))

        zz, yy, xx = np.mgrid[z_range, y_range, x_range]

        # Calculate distance from center
        dist = np.sqrt((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2)

        # Add gaussian blob
        blob = intensity * np.exp(-(dist**2) / (2 * size**2))
        data[z_range, y_range, x_range] += blob

    # Add background noise
    data += np.random.normal(0.1, 0.02, data.shape)
    data = np.clip(data, 0, 1).astype(np.float32)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, data)

    size_mb = data.nbytes / (1024**2)
    print(f"  ✓ Created: {save_path} - shape={data.shape}, size={size_mb:.1f} MB")
    return data


def main():
    """Generate all test datasets."""
    print("========================================")
    print("napari-cuda Test Data Generator")
    print("========================================")
    print()

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Generate different test volumes
    create_sinusoidal_volume()
    create_multichannel_volume()
    create_time_series()
    create_synthetic_cells()

    # Only create large volume if explicitly requested
    if os.environ.get("CREATE_LARGE_VOLUME", "0") == "1":
        create_large_volume()
    else:
        print("\nSkipping large volume (set CREATE_LARGE_VOLUME=1 to create)")

    print()
    print("========================================")
    print("Test data generation complete!")
    print("========================================")
    print()
    print("Available test data:")
    for path in Path("data").glob("*.npy"):
        size_mb = path.stat().st_size / (1024**2)
        print(f"  - {path}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
