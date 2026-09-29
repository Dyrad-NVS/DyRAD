#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"
#include "Projection.h"
#include "RadarUtils.cuh"

namespace gsplat {

// Forward declaration
void launch_projection_radar_3dgs_fused_fwd(
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor viewmats,
    const float near_range,
    const float far_range,
    const float radius_clip,
    const at::Tensor range_bins,
    const at::Tensor az_bins,
    const at::Tensor el_bins,
    const int32_t num_range_bins,
    const int32_t num_az_bins,
    const int32_t num_el_bins,
    at::Tensor ranges,
    at::Tensor azimuths,
    at::Tensor elevations,
    at::Tensor range_bin_indices,
    at::Tensor az_bin_indices,
    at::Tensor el_bin_indices,
    at::Tensor gaussian_weights
);

// Wrapper function for radar projection
std::tuple<
    at::Tensor,  // ranges
    at::Tensor,  // azimuths
    at::Tensor,  // elevations
    at::Tensor,  // range_bin_indices
    at::Tensor,  // az_bin_indices
    at::Tensor,  // el_bin_indices
    at::Tensor>  // gaussian_weights
projection_radar_3dgs_fused_fwd(
    const at::Tensor means,           // [B, N, 3]
    const at::Tensor quats,           // [B, N, 4]
    const at::Tensor scales,          // [B, N, 3]
    const at::Tensor viewmats,        // [B, C, 4, 4]
    const float near_range,
    const float far_range,
    const float radius_clip,
    const at::Tensor range_bins,      // [num_range_bins]
    const at::Tensor az_bins,         // [num_az_bins]
    const at::Tensor el_bins,         // [num_el_bins]
    const int32_t num_range_bins,
    const int32_t num_az_bins,
    const int32_t num_el_bins
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(range_bins);
    CHECK_INPUT(az_bins);
    CHECK_INPUT(el_bins);
    
    auto opt = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
    uint32_t B = (batch_dims.size() > 0) ? batch_dims[0] : 1;
    uint32_t N = means.size(-2);
    uint32_t C = viewmats.size(-3);
    
    // Create output tensors
    at::DimVector output_shape(batch_dims);
    output_shape.append({C, N});
    
    at::Tensor ranges = at::empty(output_shape, opt);
    at::Tensor azimuths = at::empty(output_shape, opt);
    at::Tensor elevations = at::empty(output_shape, opt);
    
    at::Tensor range_bin_indices = at::empty(output_shape, opt.dtype(at::kInt));
    at::Tensor az_bin_indices = at::empty(output_shape, opt.dtype(at::kInt));
    at::Tensor el_bin_indices = at::empty(output_shape, opt.dtype(at::kInt));
    
    at::Tensor gaussian_weights = at::empty(output_shape, opt);
    
    launch_projection_radar_3dgs_fused_fwd(
        means,
        quats,
        scales,
        viewmats,
        near_range,
        far_range,
        radius_clip,
        range_bins,
        az_bins,
        el_bins,
        num_range_bins,
        num_az_bins,
        num_el_bins,
        ranges,
        azimuths,
        elevations,
        range_bin_indices,
        az_bin_indices,
        el_bin_indices,
        gaussian_weights
    );
    
    return std::make_tuple(
        ranges,
        azimuths,
        elevations,
        range_bin_indices,
        az_bin_indices,
        el_bin_indices,
        gaussian_weights
    );
}

} // namespace gsplat




