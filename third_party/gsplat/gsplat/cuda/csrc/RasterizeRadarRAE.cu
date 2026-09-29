#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cmath>
#include <algorithm>

#include "Common.h"
#include "RadarUtils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Sinc-Hann PSF kernel evaluated at continuous bin offset x.
//   weight(x) = sinc(x/w)^2 * hann(x, k_eff)
//             = 0  when |x| > k_eff
// Uses Taylor series near x=0 to avoid 0/0.
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__device__ __forceinline__ scalar_t sinc_hann_psf(
    scalar_t x,
    scalar_t w,
    scalar_t k_eff
) {
    if (fabsf(x) > k_eff) return (scalar_t)0.0f;
    scalar_t t = (scalar_t)M_PI * x / (w + (scalar_t)1e-8f);
    scalar_t sinc_sq;
    if (fabsf(t) < (scalar_t)1e-4f) {
        sinc_sq = (scalar_t)1.0f - t * t / (scalar_t)3.0f;  // second-order Taylor
    } else {
        scalar_t s = __sinf(t) / t;
        sinc_sq = s * s;
    }
    scalar_t phi = (scalar_t)M_PI * x / (k_eff + (scalar_t)1e-8f);
    scalar_t hann = (scalar_t)0.5f * ((scalar_t)1.0f + __cosf(phi));
    return fmaxf((scalar_t)0.0f, sinc_sq * hann);
}

// ─────────────────────────────────────────────────────────────────────────────
// Hamming-window FFT PSF, used for both range (512-sample window) and Doppler
// (256-sample window). In bin units the response does not depend on the transform
// length, so one function serves both axes. For DDMA, the demux gather
// rd_spectra[:, (d + o_j) mod 256, :] undoes the per-transmitter Doppler shift, so
// the reduced-Doppler response is the plain Hamming-FFT lobe.
//   A(x)   = 0.54*sinc(x) + 0.23*sinc(x-1) + 0.23*sinc(x+1)
//   PSF(x) = A(x)^2 / 0.54^2   (normalised to 1 at x = 0)
// Both shifted terms carry a plus sign (each shifted Dirichlet kernel has a
// linear-phase factor of about -1).
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__device__ __forceinline__ scalar_t hamming_range_psf(scalar_t x) {
    // normalised sinc: sinc(t) = sin(pi*t)/(pi*t)
    auto sinc_n = [](scalar_t t) -> scalar_t {
        scalar_t pi_t = (scalar_t)3.14159265359f * t;
        if (fabsf(t) < (scalar_t)1e-4f) return (scalar_t)1.0f - pi_t*pi_t*(scalar_t)(1.0/6.0);
        return (scalar_t)__sinf((float)pi_t) / pi_t;
    };
    scalar_t A = (scalar_t)0.54f * sinc_n(x)
               + (scalar_t)0.23f * sinc_n(x - (scalar_t)1.0f)
               + (scalar_t)0.23f * sinc_n(x + (scalar_t)1.0f);
    // peak = A(0) = 0.54; normalize PSF = A^2 / 0.54^2
    return fmaxf((scalar_t)0, A * A * (scalar_t)(1.0f / (0.54f * 0.54f)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Rasterize Gaussians to RAD tensor (Range-Azimuth-Doppler).
//
// Two spreading modes controlled by psf_k_eff_D:
//   PSF mode (psf_k_eff_D > 0): separable 3-D PSF applied inside the kernel
//     (Hamming-FFT or sinc-Hann for D/R; sinc-Hann or interpolating tent for A);
//     v_r_cont gives differentiable Doppler placement.
//   Legacy mode (psf_k_eff_D == 0): original Gaussian R/A spreading with
//     optional Gaussian D spreading via sigma_D_bins.
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void rasterize_radar_rae_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ powers,             // [B, C, N]
    const int32_t *__restrict__ range_bin_indices,   // [B, C, N]
    const int32_t *__restrict__ az_bin_indices,      // [B, C, N]
    const int32_t *__restrict__ el_bin_indices,      // [B, C, N] (unused)
    const int32_t *__restrict__ doppler_bin_indices, // [B, C, N]
    const scalar_t *__restrict__ gaussian_weights,   // [B, C, N] sigma metres (legacy)
    const scalar_t *__restrict__ ranges_gauss,       // [B, C, N] continuous range (m)
    const scalar_t *__restrict__ azimuths_gauss,     // [B, C, N] continuous azimuth (rad)
    const scalar_t *__restrict__ range_bin_centers,  // [R]
    const scalar_t *__restrict__ az_bin_centers,     // [A]
    const int32_t num_doppler_bins,
    const int32_t num_range_bins,
    const int32_t num_az_bins,
    const scalar_t spread_factor,                    // (legacy)
    const scalar_t *__restrict__ v_r_cont,           // [B, C, N] or nullptr
    const scalar_t sigma_D_bins,                     // (legacy Gaussian D sigma)
    const scalar_t doppler_bin_spacing,
    // PSF params (all > 0 → PSF mode; 0 → legacy mode)
    const scalar_t psf_w_D,
    const scalar_t psf_k_eff_D,
    const scalar_t psf_w_R,
    const scalar_t psf_k_eff_R,
    const scalar_t psf_w_A,
    const scalar_t psf_k_eff_A,
    // 1 = analytic Hamming-FFT PSF for Doppler and range; 0 = sinc-Hann for D and R
    const int32_t use_physical_dr_psf,
    // 1 = azimuth PSF broadens as wA(az)=wA0/cos(az) (RADIal aperture foreshortening)
    const int32_t psf_az_cos_broadening,
    // 1 = azimuth kernel is a unit-sum interpolating tent over the 2 nearest
    // bins (delta accumulation). The caller then applies the full measured
    // beamformer response as a matrix product over azimuth — mathematically
    // identical to evaluating that response as the per-Gaussian azimuth PSF
    // (rasterization is linear in power), but exact (sidelobes + pedestal
    // included) and ~30x cheaper than a full-axis support loop.
    const int32_t az_tent_delta,
    // Doppler analogue of az_tent_delta: accumulate per-Doppler-bin source power
    // with a unit-sum 2-tap interpolating tent instead of a PSF, so a Doppler
    // response (a shift-invariant convolution in Doppler) can be applied outside
    // the rasterizer. Intended for num_doppler_bins = 256.
    const int32_t dop_tent_delta,
    scalar_t *__restrict__ rad_tensor               // [B, C, D, R, A]
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N) return;

    const uint32_t bid = idx / (C * N);
    const uint32_t cid = (idx / N) % C;

    int32_t range_bin   = range_bin_indices[idx];
    int32_t az_bin      = az_bin_indices[idx];
    int32_t doppler_bin = doppler_bin_indices[idx];

    if (range_bin < 0 || range_bin >= num_range_bins ||
        az_bin   < 0 || az_bin   >= num_az_bins ||
        doppler_bin < 0 || doppler_bin >= num_doppler_bins) {
        return;
    }

    scalar_t power = powers[idx];

    // ── Continuous range/azimuth in bins ─────────────────────────────────────
    scalar_t range_mean    = ranges_gauss[idx];
    scalar_t azimuth_mean  = azimuths_gauss[idx];
    scalar_t range_bin_size = fabsf(range_bin_centers[1] - range_bin_centers[0]);
    scalar_t az_bin_size    = fabsf(az_bin_centers[1]    - az_bin_centers[0]);

    // ── Base tensor offset for this (batch, camera) ───────────────────────────
    const int32_t base = (int32_t)(bid * C * num_doppler_bins * num_range_bins * num_az_bins +
                                   cid * num_doppler_bins * num_range_bins * num_az_bins);

    // =====================================================================
    // PSF mode: separable 3-D spreading (D, R, A) in one pass. Doppler wraps
    // circularly with period D; range and azimuth candidates are clamped.
    // =====================================================================
    if (psf_k_eff_D > (scalar_t)0.0f) {

        scalar_t r_cont  = (range_mean   - range_bin_centers[0]) / (range_bin_size + (scalar_t)1e-8f);
        scalar_t az_cont = (azimuth_mean - az_bin_centers[0])    / (az_bin_size    + (scalar_t)1e-8f);

        scalar_t d_cont;
        if (v_r_cont != nullptr) {
            d_cont = v_r_cont[idx] / (doppler_bin_spacing + (scalar_t)1e-12f)
                   + (scalar_t)num_doppler_bins * (scalar_t)0.5f /* v = 0 maps to bin D/2 (the DC bin of the rolled GT axis) */;
        } else {
            d_cont = (scalar_t)doppler_bin;
        }
        // Wrap d_cont into [0, num_doppler_bins) — FMCW Doppler is circular
        scalar_t D_f = (scalar_t)num_doppler_bins;
        d_cont = fmodf(d_cont + D_f * (scalar_t)1000.0f, D_f);

        // Doppler support. In tent mode the two taps must bracket d_cont, so the
        // base bin is floor(d_cont), not doppler_bin_indices (which the wrapper wraps
        // mod (D-1) and may be off-centre; harmless for a wide PSF, not for a tent).
        int32_t half_kD = (dop_tent_delta != 0)
            ? 1
            : (use_physical_dr_psf
                // Hamming-FFT support: +/-2 bins (the response is <= 6.6e-5 of peak
                // beyond it).
                ? min(2, num_doppler_bins / 2)
                : min((int32_t)ceilf(psf_k_eff_D), num_doppler_bins - 1));
        int32_t d_base = (dop_tent_delta != 0) ? (int32_t)floorf(d_cont) : doppler_bin;
        int32_t dd_lo  = (dop_tent_delta != 0) ? 0 : -half_kD;
        int32_t dd_hi  = (dop_tent_delta != 0) ? 1 :  half_kD;
        int32_t half_kR = use_physical_dr_psf
            ? 2
            : min((int32_t)ceilf(psf_k_eff_R), num_range_bins   - 1);
        // Azimuth PSF broadens with steering angle (aperture foreshortening):
        // FWHM(az) ≈ FWHM0/cos(az). Scale both bandwidth and support; clamp cos to
        // 0.30 (≈±72°) so the lobe stays bounded near the ±75° FOV edge.
        scalar_t w_A_eff   = psf_w_A;
        scalar_t k_eff_A_e = psf_k_eff_A;
        if (psf_az_cos_broadening != 0) {
            scalar_t cos_az = fmaxf(fabsf(__cosf(azimuth_mean)), (scalar_t)0.30f);
            w_A_eff   = psf_w_A     / cos_az;
            k_eff_A_e = psf_k_eff_A / cos_az;
        }
        int32_t half_kA = (az_tent_delta != 0)
            ? 1
            : min((int32_t)ceilf(k_eff_A_e), num_az_bins      - 1);

        for (int32_t dd = dd_lo; dd <= dd_hi; ++dd) {
            int32_t d_cand = ((d_base + dd) % num_doppler_bins + num_doppler_bins) % num_doppler_bins;
            scalar_t d_off = d_cont - (scalar_t)d_cand;
            if (d_off >  D_f * (scalar_t)0.5f) d_off -= D_f;
            if (d_off < -D_f * (scalar_t)0.5f) d_off += D_f;
            scalar_t d_wt = (dop_tent_delta != 0)
                ? fmaxf((scalar_t)0.0f, (scalar_t)1.0f - fabsf(d_off))
                : (use_physical_dr_psf
                    ? hamming_range_psf<scalar_t>(d_off)
                    : sinc_hann_psf<scalar_t>(d_off, psf_w_D, psf_k_eff_D));
            if (d_wt < (scalar_t)1e-8f) continue;

            for (int32_t dr = -half_kR; dr <= half_kR; ++dr) {
                int32_t r_cand = max(0, min(num_range_bins - 1, range_bin + dr));
                scalar_t r_off = r_cont - (scalar_t)r_cand;
                scalar_t r_wt = use_physical_dr_psf
                    ? hamming_range_psf<scalar_t>(r_off)
                    : sinc_hann_psf<scalar_t>(r_off, psf_w_R, psf_k_eff_R);
                if (r_wt < (scalar_t)1e-8f) continue;

                for (int32_t da = -half_kA; da <= half_kA; ++da) {
                    int32_t a_cand = max(0, min(num_az_bins - 1, az_bin + da));
                    scalar_t a_off = az_cont - (scalar_t)a_cand;
                    scalar_t a_wt = (az_tent_delta != 0)
                        ? fmaxf((scalar_t)0.0f, (scalar_t)1.0f - fabsf(a_off))
                        : sinc_hann_psf<scalar_t>(a_off, w_A_eff, k_eff_A_e);
                    if (a_wt < (scalar_t)1e-8f) continue;

                    scalar_t contrib = power * d_wt * r_wt * a_wt;
                    if (contrib > (scalar_t)1e-12f) {
                        int32_t tensor_idx = base
                            + d_cand * num_range_bins * num_az_bins
                            + r_cand * num_az_bins
                            + a_cand;
                        gpuAtomicAdd(&rad_tensor[tensor_idx], contrib);
                    }
                }
            }
        }
        return;  // PSF mode done
    }

    // =====================================================================
    // Legacy mode: Gaussian R/A spreading + optional Gaussian D spreading.
    // =====================================================================
    scalar_t sigma_m = gaussian_weights[idx];
    sigma_m = fmaxf((scalar_t)1e-3f, sigma_m);

    scalar_t estimated_scale_m = fmaxf((scalar_t)0.01f, fminf(sigma_m, (scalar_t)10.0f));

    scalar_t range_spread_bins = estimated_scale_m / (range_bin_size + (scalar_t)1e-6f);
    range_spread_bins = fmaxf((scalar_t)1.0f, fminf(range_spread_bins * spread_factor, (scalar_t)10.0f));

    scalar_t az_bin_sz = (num_az_bins > 1 && az_bin_centers != nullptr)
                         ? fabsf(az_bin_centers[1] - az_bin_centers[0]) : (scalar_t)0.0175f;
    scalar_t angular_spread_rad  = estimated_scale_m / (range_mean + (scalar_t)1e-6f);
    scalar_t az_spread_bins = angular_spread_rad / (az_bin_sz + (scalar_t)1e-6f);
    az_spread_bins = fmaxf((scalar_t)1.0f, fminf(az_spread_bins * spread_factor, (scalar_t)10.0f));

    int32_t range_spread = (int32_t)ceilf(range_spread_bins);
    int32_t az_spread    = (int32_t)ceilf(az_spread_bins);

    bool do_doppler_spread = (v_r_cont != nullptr);
    scalar_t d_cont = do_doppler_spread
        ? (v_r_cont[idx] / doppler_bin_spacing + (scalar_t)num_doppler_bins * (scalar_t)0.5f /* v = 0 maps to bin D/2 (the DC bin of the rolled GT axis) */)
        : (scalar_t)doppler_bin;
    // Wrap d_cont into [0, num_doppler_bins) for FMCW circular Doppler
    scalar_t D_f_leg = (scalar_t)num_doppler_bins;
    if (do_doppler_spread)
        d_cont = fmodf(d_cont + D_f_leg * (scalar_t)1000.0f, D_f_leg);
    int32_t half_d = do_doppler_spread
        ? min((int32_t)ceilf((scalar_t)3.0f * sigma_D_bins), num_doppler_bins - 1)
        : 0;

    for (int32_t dd = -half_d; dd <= half_d; ++dd) {
        int32_t d_cand = ((doppler_bin + dd) % num_doppler_bins + num_doppler_bins) % num_doppler_bins;

        scalar_t d_wt;
        if (do_doppler_spread) {
            scalar_t d_delta = d_cont - (scalar_t)d_cand;
            if (d_delta >  D_f_leg * (scalar_t)0.5f) d_delta -= D_f_leg;
            if (d_delta < -D_f_leg * (scalar_t)0.5f) d_delta += D_f_leg;
            d_wt = __expf(-0.5f * (d_delta / (sigma_D_bins + (scalar_t)1e-8f)) *
                                   (d_delta / (sigma_D_bins + (scalar_t)1e-8f)));
            if (d_wt < (scalar_t)1e-6f) continue;
        } else {
            d_wt = (scalar_t)1.0f;
        }

        for (int32_t dr = -range_spread; dr <= range_spread; ++dr) {
            int32_t r_bin = range_bin + dr;
            if (r_bin < 0 || r_bin >= num_range_bins) continue;

            scalar_t r_bin_center = (range_bin_centers != nullptr) ?
                range_bin_centers[r_bin] : range_mean;
            scalar_t range_dist = fabsf(r_bin_center - range_mean);

            for (int32_t da = -az_spread; da <= az_spread; ++da) {
                int32_t a_bin = az_bin + da;
                if (a_bin < 0 || a_bin >= num_az_bins) continue;

                scalar_t a_bin_center = (az_bin_centers != nullptr) ?
                    az_bin_centers[a_bin] : azimuth_mean;
                scalar_t az_dist = fabsf(a_bin_center - azimuth_mean);
                if (az_dist > (scalar_t)M_PI) az_dist = (scalar_t)(2.0 * M_PI) - az_dist;

                scalar_t range_norm = range_dist / (range_bin_size * range_spread_bins + (scalar_t)1e-6f);
                scalar_t az_norm    = az_dist    / (az_bin_sz    * az_spread_bins    + (scalar_t)1e-6f);
                scalar_t bin_weight = __expf(-0.5f * (range_norm * range_norm + az_norm * az_norm));
                if (bin_weight < (scalar_t)1e-6f) continue;

                int32_t tensor_idx = base
                    + d_cand * num_range_bins * num_az_bins
                    + r_bin  * num_az_bins
                    + a_bin;

                scalar_t contribution = power * bin_weight * d_wt;
                if (contribution > (scalar_t)1e-10f) {
                    gpuAtomicAdd(&rad_tensor[tensor_idx], contribution);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Launch wrapper
// ─────────────────────────────────────────────────────────────────────────────
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
    // PSF params (0 → legacy mode)
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
) {
    DEVICE_GUARD(powers);
    CHECK_INPUT(powers);
    CHECK_INPUT(range_bin_indices);
    CHECK_INPUT(az_bin_indices);
    CHECK_INPUT(el_bin_indices);
    CHECK_INPUT(doppler_bin_indices);
    CHECK_INPUT(gaussian_weights);
    CHECK_INPUT(ranges);
    CHECK_INPUT(azimuths);
    CHECK_INPUT(range_bin_centers);
    CHECK_INPUT(az_bin_centers);

    const uint32_t B = powers.size(0);
    const uint32_t C = powers.size(1);
    const uint32_t N = powers.size(2);

    auto options = powers.options();
    at::Tensor rad_tensor = at::zeros({B, C, num_doppler_bins, num_range_bins, num_az_bins}, options).contiguous();

    const int threads = 256;
    const int blocks  = (B * C * N + threads - 1) / threads;

    at::Tensor v_r_cont_c;
    if (v_r_cont.has_value()) {
        CHECK_INPUT(v_r_cont.value());
        v_r_cont_c = v_r_cont.value().contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES(
        powers.scalar_type(), "rasterize_radar_rae_fwd", [&] {
            const scalar_t* v_r_cont_ptr = v_r_cont.has_value()
                ? v_r_cont_c.data_ptr<scalar_t>() : nullptr;
            rasterize_radar_rae_fwd_kernel<scalar_t><<<blocks, threads>>>(
                B, C, N,
                powers.data_ptr<scalar_t>(),
                range_bin_indices.data_ptr<int32_t>(),
                az_bin_indices.data_ptr<int32_t>(),
                el_bin_indices.data_ptr<int32_t>(),
                doppler_bin_indices.data_ptr<int32_t>(),
                gaussian_weights.data_ptr<scalar_t>(),
                ranges.data_ptr<scalar_t>(),
                azimuths.data_ptr<scalar_t>(),
                range_bin_centers.data_ptr<scalar_t>(),
                az_bin_centers.data_ptr<scalar_t>(),
                num_doppler_bins,
                num_range_bins,
                num_az_bins,
                (scalar_t)spread_factor,
                v_r_cont_ptr,
                (scalar_t)sigma_D_bins,
                (scalar_t)doppler_bin_spacing,
                (scalar_t)psf_w_D,
                (scalar_t)psf_k_eff_D,
                (scalar_t)psf_w_R,
                (scalar_t)psf_k_eff_R,
                (scalar_t)psf_w_A,
                (scalar_t)psf_k_eff_A,
                (int32_t)use_physical_dr_psf,
                (int32_t)psf_az_cos_broadening,
                (int32_t)az_tent_delta,
                (int32_t)dop_tent_delta,
                rad_tensor.data_ptr<scalar_t>()
            );
        });

    return rad_tensor;
}

} // namespace gsplat
