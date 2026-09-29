// Modified from gsplat (Apache-2.0) for DyRAD.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

// Forward declarations
namespace gsplat {
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    projection_radar_3dgs_fused_fwd(
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
        const int32_t num_el_bins
    );

    std::tuple<at::Tensor, at::Tensor, at::Tensor> projection_radar_3dgs_fused_bwd(
        const at::Tensor means,
        const at::Tensor quats,
        const at::Tensor scales,
        const at::Tensor viewmats,
        const float near_range,
        const float far_range,
        const at::Tensor ranges,
        const at::Tensor azimuths,
        const at::Tensor elevations,
        const at::Tensor gaussian_weights,
        const at::Tensor range_bin_indices,
        const at::Tensor v_ranges,
        const at::Tensor v_azimuths,
        const at::Tensor v_elevations,
        const at::Tensor v_gaussian_weights
    );
    
    at::Tensor rasterize_radar_rae_fwd(
        const at::Tensor powers,
        const at::Tensor range_bin_indices,
        const at::Tensor az_bin_indices,
        const at::Tensor el_bin_indices,
        const at::Tensor doppler_bin_indices,
        const at::Tensor gaussian_weights,
        const at::Tensor ranges,
        const at::Tensor azimuths,
        const at::Tensor range_bin_centers,
        const at::Tensor az_bin_centers,
        const int32_t num_doppler_bins,
        const int32_t num_range_bins,
        const int32_t num_az_bins,
        const float spread_factor,
        const c10::optional<at::Tensor> v_r_cont,
        const float sigma_D_bins,
        const float doppler_bin_spacing,
        const float psf_w_D,
        const float psf_k_eff_D,
        const float psf_w_R,
        const float psf_k_eff_R,
        const float psf_w_A,
        const float psf_k_eff_A,
        const bool use_physical_dr_psf,
        const bool psf_az_cos_broadening,
        const bool az_tent_delta,
        const bool dop_tent_delta
    );

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rasterize_radar_rae_bwd(
        const at::Tensor powers,
        const at::Tensor range_bin_indices,
        const at::Tensor az_bin_indices,
        const at::Tensor doppler_bin_indices,
        const at::Tensor gaussian_sigmas_m,
        const at::Tensor ranges,
        const at::Tensor azimuths,
        const at::Tensor range_bin_centers,
        const at::Tensor az_bin_centers,
        const int32_t num_doppler_bins,
        const int32_t num_range_bins,
        const int32_t num_az_bins,
        const float spread_factor,
        const at::Tensor v_rad_tensor,
        const c10::optional<at::Tensor> v_r_cont,
        const float sigma_D_bins,
        const float doppler_bin_spacing,
        const float psf_w_D,
        const float psf_k_eff_D,
        const float psf_w_R,
        const float psf_k_eff_R,
        const float psf_w_A,
        const float psf_k_eff_A,
        const bool use_physical_dr_psf,
        const bool psf_az_cos_broadening,
        const bool az_tent_delta,
        const bool dop_tent_delta
    );

}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "gsplat CUDA extensions";
    
    // Radar projection function
    // Note: We need to match the C++ function signature exactly
    // The order matters: tensors, floats, tensors, ints
    m.def(
        "projection_radar_3dgs_fused_fwd",
        [](const at::Tensor& means,
           const at::Tensor& quats,
           const at::Tensor& scales,
           const at::Tensor& viewmats,
           float near_range,
           float far_range,
           float radius_clip,
           const at::Tensor& range_bins,
           const at::Tensor& az_bins,
           const at::Tensor& el_bins,
           int32_t num_range_bins,
           int32_t num_az_bins,
           int32_t num_el_bins) {
            return gsplat::projection_radar_3dgs_fused_fwd(
                means, quats, scales, viewmats,
                near_range, far_range, radius_clip,
                range_bins, az_bins, el_bins,
                num_range_bins, num_az_bins, num_el_bins
            );
        },
        "Radar projection forward pass",
        py::arg("means"),
        py::arg("quats"),
        py::arg("scales"),
        py::arg("viewmats"),
        py::arg("near_range"),
        py::arg("far_range"),
        py::arg("radius_clip"),
        py::arg("range_bins"),
        py::arg("az_bins"),
        py::arg("el_bins"),
        py::arg("num_range_bins"),
        py::arg("num_az_bins"),
        py::arg("num_el_bins")
    );

    m.def(
        "projection_radar_3dgs_fused_bwd",
        &gsplat::projection_radar_3dgs_fused_bwd,
        "Radar projection backward pass (returns grads for means/quats/scales)"
    );
    
    // Radar rasterization
    m.def(
        "rasterize_radar_rae_fwd",
        [](const at::Tensor& powers,
           const at::Tensor& range_bin_indices,
           const at::Tensor& az_bin_indices,
           const at::Tensor& el_bin_indices,
           const at::Tensor& doppler_bin_indices,
           const at::Tensor& gaussian_weights,
           const at::Tensor& ranges,
           const at::Tensor& azimuths,
           const at::Tensor& range_bin_centers,
           const at::Tensor& az_bin_centers,
           int32_t num_doppler_bins,
           int32_t num_range_bins,
           int32_t num_az_bins,
           float spread_factor,
           const c10::optional<at::Tensor> v_r_cont,
           float sigma_D_bins,
           float doppler_bin_spacing,
           float psf_w_D,
           float psf_k_eff_D,
           float psf_w_R,
           float psf_k_eff_R,
           float psf_w_A,
           float psf_k_eff_A,
           bool use_physical_dr_psf,
           bool psf_az_cos_broadening,
           bool az_tent_delta, bool dop_tent_delta) {
            return gsplat::rasterize_radar_rae_fwd(
                powers, range_bin_indices, az_bin_indices, el_bin_indices,
                doppler_bin_indices, gaussian_weights, ranges, azimuths,
                range_bin_centers, az_bin_centers,
                num_doppler_bins, num_range_bins, num_az_bins, spread_factor,
                v_r_cont, sigma_D_bins, doppler_bin_spacing,
                psf_w_D, psf_k_eff_D, psf_w_R, psf_k_eff_R, psf_w_A, psf_k_eff_A,
                use_physical_dr_psf, psf_az_cos_broadening, az_tent_delta, dop_tent_delta
            );
        },
        "Rasterize Gaussians to RAD tensor (with optional 3D sinc-Hann PSF)",
        py::arg("powers"),
        py::arg("range_bin_indices"),
        py::arg("az_bin_indices"),
        py::arg("el_bin_indices"),
        py::arg("doppler_bin_indices"),
        py::arg("gaussian_weights"),
        py::arg("ranges"),
        py::arg("azimuths"),
        py::arg("range_bin_centers"),
        py::arg("az_bin_centers"),
        py::arg("num_doppler_bins"),
        py::arg("num_range_bins"),
        py::arg("num_az_bins"),
        py::arg("spread_factor") = 3.0f,
        py::arg("v_r_cont") = py::none(),
        py::arg("sigma_D_bins") = 1.5f,
        py::arg("doppler_bin_spacing") = 0.06f,
        py::arg("psf_w_D") = 0.0f,
        py::arg("psf_k_eff_D") = 0.0f,
        py::arg("psf_w_R") = 0.0f,
        py::arg("psf_k_eff_R") = 0.0f,
        py::arg("psf_w_A") = 0.0f,
        py::arg("psf_k_eff_A") = 0.0f,
        py::arg("use_physical_dr_psf") = false,
        py::arg("psf_az_cos_broadening") = false,
        py::arg("az_tent_delta") = false,
        py::arg("dop_tent_delta") = false
    );

    m.def(
        "rasterize_radar_rae_bwd",
        [](const at::Tensor& powers,
           const at::Tensor& range_bin_indices,
           const at::Tensor& az_bin_indices,
           const at::Tensor& doppler_bin_indices,
           const at::Tensor& gaussian_sigmas_m,
           const at::Tensor& ranges,
           const at::Tensor& azimuths,
           const at::Tensor& range_bin_centers,
           const at::Tensor& az_bin_centers,
           int32_t num_doppler_bins,
           int32_t num_range_bins,
           int32_t num_az_bins,
           float spread_factor,
           const at::Tensor& v_rad_tensor,
           const c10::optional<at::Tensor> v_r_cont,
           float sigma_D_bins,
           float doppler_bin_spacing,
           float psf_w_D,
           float psf_k_eff_D,
           float psf_w_R,
           float psf_k_eff_R,
           float psf_w_A,
           float psf_k_eff_A,
           bool use_physical_dr_psf,
           bool psf_az_cos_broadening,
           bool az_tent_delta, bool dop_tent_delta) {
            return gsplat::rasterize_radar_rae_bwd(
                powers, range_bin_indices, az_bin_indices, doppler_bin_indices,
                gaussian_sigmas_m, ranges, azimuths, range_bin_centers, az_bin_centers,
                num_doppler_bins, num_range_bins, num_az_bins, spread_factor,
                v_rad_tensor, v_r_cont, sigma_D_bins, doppler_bin_spacing,
                psf_w_D, psf_k_eff_D, psf_w_R, psf_k_eff_R, psf_w_A, psf_k_eff_A,
                use_physical_dr_psf, psf_az_cos_broadening, az_tent_delta, dop_tent_delta
            );
        },
        "Backward pass for radar rasterization (returns grads for powers, ranges, azimuths, sigma_m, v_r_cont)",
        py::arg("powers"),
        py::arg("range_bin_indices"),
        py::arg("az_bin_indices"),
        py::arg("doppler_bin_indices"),
        py::arg("gaussian_sigmas_m"),
        py::arg("ranges"),
        py::arg("azimuths"),
        py::arg("range_bin_centers"),
        py::arg("az_bin_centers"),
        py::arg("num_doppler_bins"),
        py::arg("num_range_bins"),
        py::arg("num_az_bins"),
        py::arg("spread_factor"),
        py::arg("v_rad_tensor"),
        py::arg("v_r_cont") = py::none(),
        py::arg("sigma_D_bins") = 1.5f,
        py::arg("doppler_bin_spacing") = 0.06f,
        py::arg("psf_w_D") = 0.0f,
        py::arg("psf_k_eff_D") = 0.0f,
        py::arg("psf_w_R") = 0.0f,
        py::arg("psf_k_eff_R") = 0.0f,
        py::arg("psf_w_A") = 0.0f,
        py::arg("psf_k_eff_A") = 0.0f,
        py::arg("use_physical_dr_psf") = false,
        py::arg("psf_az_cos_broadening") = false,
        py::arg("az_tent_delta") = false,
        py::arg("dop_tent_delta") = false
    );
}

