#pragma once

#include "Common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

namespace gsplat {

/**
 * Find the bin index for a given value in a sorted array of bin centers.
 * Uses binary search to find the closest bin.
 * 
 * @param value: The value to find the bin for
 * @param bin_centers: Sorted array of bin center values
 * @param num_bins: Number of bins
 * @return: Index of the bin (clamped to [0, num_bins-1])
 */
template <typename scalar_t>
inline __device__ int find_bin_index(
    const scalar_t value,
    const scalar_t *bin_centers,
    const int num_bins
) {
    if (num_bins == 0) return 0;
    if (value <= bin_centers[0]) return 0;
    if (value >= bin_centers[num_bins - 1]) return num_bins - 1;
    
    // Binary search
    int left = 0;
    int right = num_bins - 1;
    while (right - left > 1) {
        int mid = (left + right) / 2;
        if (value < bin_centers[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }
    
    // Choose the closer bin
    scalar_t dist_left = fabsf(value - bin_centers[left]);
    scalar_t dist_right = fabsf(value - bin_centers[right]);
    return (dist_left < dist_right) ? left : right;
}

/**
 * Interpolate antenna gain from a 2D lookup table.
 * Uses bilinear interpolation.
 * 
 * @param azimuth: Azimuth angle (radians)
 * @param elevation: Elevation angle (radians)
 * @param gain_lut: Antenna gain lookup table, flattened [num_az_bins * num_el_bins]
 * @param az_bin_centers: Azimuth bin centers (radians)
 * @param el_bin_centers: Elevation bin centers (radians)
 * @param num_az_bins: Number of azimuth bins
 * @param num_el_bins: Number of elevation bins
 * @return: Interpolated antenna gain value
 */
template <typename scalar_t>
inline __device__ scalar_t interpolate_antenna_gain(
    const scalar_t azimuth,
    const scalar_t elevation,
    const scalar_t *gain_lut,
    const scalar_t *az_bin_centers,
    const scalar_t *el_bin_centers,
    const int num_az_bins,
    const int num_el_bins
) {
    // Find bin indices
    int az_idx = find_bin_index(azimuth, az_bin_centers, num_az_bins);
    int el_idx = find_bin_index(elevation, el_bin_centers, num_el_bins);
    
    // Clamp to valid range
    az_idx = std::max(0, std::min(az_idx, num_az_bins - 1));
    el_idx = std::max(0, std::min(el_idx, num_el_bins - 1));
    
    // Get neighboring indices for bilinear interpolation
    int az_idx0 = az_idx;
    int az_idx1 = std::min(az_idx + 1, num_az_bins - 1);
    int el_idx0 = el_idx;
    int el_idx1 = std::min(el_idx + 1, num_el_bins - 1);
    
    // Get gain values at the four corners
    scalar_t g00 = gain_lut[el_idx0 * num_az_bins + az_idx0];
    scalar_t g01 = gain_lut[el_idx0 * num_az_bins + az_idx1];
    scalar_t g10 = gain_lut[el_idx1 * num_az_bins + az_idx0];
    scalar_t g11 = gain_lut[el_idx1 * num_az_bins + az_idx1];
    
    // Compute interpolation weights
    scalar_t az_frac = 0.0f;
    if (az_idx1 > az_idx0 && az_bin_centers[az_idx1] > az_bin_centers[az_idx0]) {
        az_frac = (azimuth - az_bin_centers[az_idx0]) / 
                  (az_bin_centers[az_idx1] - az_bin_centers[az_idx0]);
        az_frac = std::max(static_cast<scalar_t>(0.0), std::min(static_cast<scalar_t>(1.0), az_frac));
    }
    
    scalar_t el_frac = 0.0f;
    if (el_idx1 > el_idx0 && el_bin_centers[el_idx1] > el_bin_centers[el_idx0]) {
        el_frac = (elevation - el_bin_centers[el_idx0]) / 
                  (el_bin_centers[el_idx1] - el_bin_centers[el_idx0]);
        el_frac = std::max(static_cast<scalar_t>(0.0), std::min(static_cast<scalar_t>(1.0), el_frac));
    }
    
    // Bilinear interpolation
    scalar_t g0 = g00 * (1.0f - az_frac) + g01 * az_frac;
    scalar_t g1 = g10 * (1.0f - az_frac) + g11 * az_frac;
    scalar_t gain = g0 * (1.0f - el_frac) + g1 * el_frac;
    
    return gain;
}

} // namespace gsplat

