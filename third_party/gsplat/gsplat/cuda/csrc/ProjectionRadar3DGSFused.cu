#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cmath>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"
#include "RadarUtils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

/**
 * Project 3D Gaussians to spherical (radar) coordinates.
 * Computes range, azimuth, elevation and Gaussian size (sigma in meters) for radar rasterization.
 */
template <typename scalar_t>
__global__ void projection_radar_3dgs_fused_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,     // [B, N, 3] - Gaussian means in world space
    const scalar_t *__restrict__ quats,     // [B, N, 4] - Quaternions
    const scalar_t *__restrict__ scales,    // [B, N, 3] - Scales
    const scalar_t *__restrict__ viewmats,  // [B, C, 4, 4] - World-to-radar transformation
    const scalar_t near_range,
    const scalar_t far_range,
    const scalar_t radius_clip,
    // Bin arrays for spherical coordinates
    const scalar_t *__restrict__ range_bins,    // [num_range_bins]
    const scalar_t *__restrict__ az_bins,       // [num_az_bins]
    const scalar_t *__restrict__ el_bins,        // [num_el_bins]
    const int32_t num_range_bins,
    const int32_t num_az_bins,
    const int32_t num_el_bins,
    // Outputs
    scalar_t *__restrict__ ranges,          // [B, C, N] - Range values
    scalar_t *__restrict__ azimuths,        // [B, C, N] - Azimuth values (radians)
    scalar_t *__restrict__ elevations,       // [B, C, N] - Elevation values (radians)
    int32_t *__restrict__ range_bin_indices, // [B, C, N] - Range bin indices
    int32_t *__restrict__ az_bin_indices,    // [B, C, N] - Azimuth bin indices
    int32_t *__restrict__ el_bin_indices,    // [B, C, N] - Elevation bin indices
    scalar_t *__restrict__ gaussian_weights  // [B, C, N] - Gaussian sigma (meters) for rasterization footprint
) {
    // Parallelize over B * C * N
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N) {
        return;
    }
    
    const uint32_t bid = idx / (C * N);  // batch id
    const uint32_t cid = (idx / N) % C;  // camera/radar id
    const uint32_t gid = idx % N;        // gaussian id
    
    // Shift pointers to current batch, camera, and gaussian
    means += bid * N * 3 + gid * 3;
    viewmats += bid * C * 16 + cid * 16;
    
    // Extract rotation and translation from viewmat (world-to-radar)
    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0], viewmats[4], viewmats[8],   // 1st column
        viewmats[1], viewmats[5], viewmats[9],   // 2nd column
        viewmats[2], viewmats[6], viewmats[10]   // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);
    
    // Transform Gaussian mean to radar (camera) space
    vec3 mean_w = glm::make_vec3(means);
    vec3 mean_r;
    posW2C(R, t, mean_w, mean_r);
    
    // Compute range (distance from radar origin)
    scalar_t range = glm::length(mean_r);
    
    // Check range bounds
    if (range < near_range || range > far_range) {
        ranges[idx] = 0.0f;
        azimuths[idx] = 0.0f;
        elevations[idx] = 0.0f;
        range_bin_indices[idx] = -1;
        az_bin_indices[idx] = -1;
        el_bin_indices[idx] = -1;
        gaussian_weights[idx] = 0.0f;
        return;
    }
    
    // Compute azimuth and elevation (spherical coordinates)
    // Azimuth: angle in the xy-plane, measured from the +x axis
    // Elevation: angle from xy-plane to z-axis (-π/2 to π/2)
    scalar_t x = mean_r.x;
    scalar_t y = mean_r.y;
    scalar_t z = mean_r.z;
    
    // Not wrapped to [0, 2π]: azimuth bins are centred on 0, in [-fov/2, +fov/2].
    // The non-negated atan2 matches the azimuth axis direction of the GT tensors.
    scalar_t azimuth = atan2f(y, x);   // [-π, π]
    
    scalar_t elevation = -asinf(z / (range + 1e-10f));  // [-π/2, π/2] (negated to match K-Radar convention)
    
    // Find bin indices
    int32_t range_bin = find_bin_index(range, range_bins, num_range_bins);
    int32_t az_bin = find_bin_index(azimuth, az_bins, num_az_bins);
    int32_t el_bin = find_bin_index(elevation, el_bins, num_el_bins);
    
    // Clamp bin indices to valid range
    range_bin = max(0, min(range_bin, num_range_bins - 1));
    az_bin = max(0, min(az_bin, num_az_bins - 1));
    el_bin = max(0, min(el_bin, num_el_bins - 1));
    
    // Compute Gaussian weight based on scale
    // The weight represents how much this Gaussian contributes to the bin
    quats += bid * N * 4 + gid * 4;
    scales += bid * N * 3 + gid * 3;
    
    vec4 quat = glm::make_vec4(quats);
    vec3 scale = glm::make_vec3(scales);
    
    // Normalize quaternion
    float quat_norm = glm::length(quat);
    if (quat_norm > 1e-8f) {
        quat = quat / quat_norm;
    }
    
    // Compute covariance matrix from quaternion and scale
    mat3 covar;
    quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    
    // Transform covariance to radar space
    mat3 covar_r;
    covarW2C(R, covar, covar_r);
    
    // Estimate Gaussian "size" in radar (Cartesian) space (meters)
    // Use the trace of the covariance as a measure of spread (isotropic approximation).
    scalar_t trace_covar = covar_r[0][0] + covar_r[1][1] + covar_r[2][2];
    scalar_t gaussian_size = sqrtf(trace_covar / 3.0f);

    // Store sigma (meters). Rasterization uses this to decide bin footprint.
    scalar_t sigma_m = max(gaussian_size, (scalar_t)1e-3);  // avoid zeros / tiny
    
    // Write outputs
    ranges[idx] = range;
    azimuths[idx] = azimuth;
    elevations[idx] = elevation;
    range_bin_indices[idx] = range_bin;
    az_bin_indices[idx] = az_bin;
    el_bin_indices[idx] = el_bin;
    gaussian_weights[idx] = sigma_m;
}

// Launch function for forward pass
void launch_projection_radar_3dgs_fused_fwd(
    const at::Tensor means,           // [B, N, 3]
    const at::Tensor quats,           // [B, N, 4]
    const at::Tensor scales,          // [B, N, 3]
    const at::Tensor viewmats,        // [B, C, 4, 4]
    const float near_range,
    const float far_range,
    const float radius_clip,
    const at::Tensor range_bins,      // [num_range_bins]
    const at::Tensor az_bins,        // [num_az_bins]
    const at::Tensor el_bins,        // [num_el_bins]
    const int32_t num_range_bins,
    const int32_t num_az_bins,
    const int32_t num_el_bins,
    at::Tensor ranges,               // [B, C, N]
    at::Tensor azimuths,             // [B, C, N]
    at::Tensor elevations,            // [B, C, N]
    at::Tensor range_bin_indices,     // [B, C, N]
    at::Tensor az_bin_indices,        // [B, C, N]
    at::Tensor el_bin_indices,        // [B, C, N]
    at::Tensor gaussian_weights       // [B, C, N]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(range_bins);
    CHECK_INPUT(az_bins);
    CHECK_INPUT(el_bins);
    CHECK_INPUT(ranges);
    CHECK_INPUT(azimuths);
    CHECK_INPUT(elevations);
    CHECK_INPUT(range_bin_indices);
    CHECK_INPUT(az_bin_indices);
    CHECK_INPUT(el_bin_indices);
    CHECK_INPUT(gaussian_weights);
    
    const uint32_t B = means.size(0);
    const uint32_t C = viewmats.size(1);
    const uint32_t N = means.size(1);
    
    const int threads = 256;
    const int blocks = (B * C * N + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(), "projection_radar_3dgs_fused_fwd", [&] {
            projection_radar_3dgs_fused_fwd_kernel<scalar_t><<<blocks, threads>>>(
                B, C, N,
                means.data_ptr<scalar_t>(),
                quats.data_ptr<scalar_t>(),
                scales.data_ptr<scalar_t>(),
                viewmats.data_ptr<scalar_t>(),
                near_range,
                far_range,
                radius_clip,
                range_bins.data_ptr<scalar_t>(),
                az_bins.data_ptr<scalar_t>(),
                el_bins.data_ptr<scalar_t>(),
                num_range_bins,
                num_az_bins,
                num_el_bins,
                ranges.data_ptr<scalar_t>(),
                azimuths.data_ptr<scalar_t>(),
                elevations.data_ptr<scalar_t>(),
                range_bin_indices.data_ptr<int32_t>(),
                az_bin_indices.data_ptr<int32_t>(),
                el_bin_indices.data_ptr<int32_t>(),
                gaussian_weights.data_ptr<scalar_t>()
            );
        });
}

} // namespace gsplat

