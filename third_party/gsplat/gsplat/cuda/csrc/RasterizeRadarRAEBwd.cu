#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cmath>
#include <tuple>

#include "Common.h"

namespace gsplat {
namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Forward value of sinc-Hann PSF at continuous offset x.
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__device__ __forceinline__ scalar_t sinc_hann_psf_bwd(scalar_t x, scalar_t w, scalar_t k_eff) {
    if (fabsf(x) > k_eff) return (scalar_t)0.0f;
    scalar_t t = (scalar_t)M_PI * x / (w + (scalar_t)1e-8f);
    scalar_t sinc_sq;
    if (fabsf(t) < (scalar_t)1e-4f) {
        sinc_sq = (scalar_t)1.0f - t * t / (scalar_t)3.0f;
    } else {
        scalar_t s = __sinf(t) / t;
        sinc_sq = s * s;
    }
    scalar_t phi  = (scalar_t)M_PI * x / (k_eff + (scalar_t)1e-8f);
    scalar_t hann = (scalar_t)0.5f * ((scalar_t)1.0f + __cosf(phi));
    return fmaxf((scalar_t)0.0f, sinc_sq * hann);
}

// ─────────────────────────────────────────────────────────────────────────────
// Derivative of sinc-Hann PSF w.r.t. x: d(sinc_hann)/dx.
//   Used to propagate gradients through the continuous bin offset.
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sinc_hann_psf_dx(scalar_t x, scalar_t w, scalar_t k_eff) {
    if (fabsf(x) > k_eff) return (scalar_t)0.0f;

    scalar_t pi_over_w = (scalar_t)M_PI / (w + (scalar_t)1e-8f);
    scalar_t t = pi_over_w * x;

    scalar_t sinc_sq, d_sinc_sq_dt;
    if (fabsf(t) < (scalar_t)1e-4f) {
        sinc_sq        = (scalar_t)1.0f - t * t / (scalar_t)3.0f;
        d_sinc_sq_dt   = -(scalar_t)2.0f * t / (scalar_t)3.0f;
    } else {
        scalar_t s = __sinf(t);
        scalar_t c = __cosf(t);
        sinc_sq      = (s / t) * (s / t);
        d_sinc_sq_dt = (scalar_t)2.0f * s * (c * t - s) / (t * t * t);
    }
    scalar_t d_sinc_sq_dx = d_sinc_sq_dt * pi_over_w;

    scalar_t pi_over_k = (scalar_t)M_PI / (k_eff + (scalar_t)1e-8f);
    scalar_t phi       = pi_over_k * x;
    scalar_t hann      = (scalar_t)0.5f * ((scalar_t)1.0f + __cosf(phi));
    scalar_t d_hann_dx = -(scalar_t)0.5f * __sinf(phi) * pi_over_k;

    // product rule: d(sinc_sq * hann)/dx — no clamp on derivative
    return d_sinc_sq_dx * hann + sinc_sq * d_hann_dx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hamming-FFT PSF (range and Doppler), forward value; must match the forward kernel.
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__device__ __forceinline__ scalar_t hamming_range_psf_bwd(scalar_t x) {
    auto sinc_n = [](scalar_t t) -> scalar_t {
        scalar_t pi_t = (scalar_t)3.14159265359f * t;
        if (fabsf(t) < (scalar_t)1e-4f) return (scalar_t)1.0f - pi_t*pi_t*(scalar_t)(1.0/6.0);
        return (scalar_t)__sinf((float)pi_t) / pi_t;
    };
    scalar_t A = (scalar_t)0.54f * sinc_n(x)
               + (scalar_t)0.23f * sinc_n(x - (scalar_t)1.0f)
               + (scalar_t)0.23f * sinc_n(x + (scalar_t)1.0f);
    return fmaxf((scalar_t)0, A * A * (scalar_t)(1.0f / (0.54f * 0.54f)));
}

// d/dx of hamming_range_psf.
//   d_sinc(t)/dt = [πt*cos(πt) - sin(πt)] / (πt²)   (= 0 at t=0)
template <typename scalar_t>
__device__ __forceinline__ scalar_t d_hamming_range_psf_dx(scalar_t x) {
    auto sinc_n = [](scalar_t t) -> scalar_t {
        scalar_t pi_t = (scalar_t)3.14159265359f * t;
        if (fabsf(t) < (scalar_t)1e-4f) return (scalar_t)1.0f - pi_t*pi_t*(scalar_t)(1.0/6.0);
        return (scalar_t)__sinf((float)pi_t) / pi_t;
    };
    auto d_sinc_n = [](scalar_t t) -> scalar_t {
        if (fabsf(t) < (scalar_t)1e-4f) return (scalar_t)0.0f;
        scalar_t pi_t = (scalar_t)3.14159265359f * t;
        return (pi_t * (scalar_t)__cosf((float)pi_t) - (scalar_t)__sinf((float)pi_t))
               / (pi_t * t);
    };
    scalar_t A  = (scalar_t)0.54f * sinc_n(x)
                + (scalar_t)0.23f * sinc_n(x - (scalar_t)1.0f)
                + (scalar_t)0.23f * sinc_n(x + (scalar_t)1.0f);
    scalar_t dA = (scalar_t)0.54f * d_sinc_n(x)
                + (scalar_t)0.23f * d_sinc_n(x - (scalar_t)1.0f)
                + (scalar_t)0.23f * d_sinc_n(x + (scalar_t)1.0f);
    return (scalar_t)2.0f * A * dA * (scalar_t)(1.0f / (0.54f * 0.54f));
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward kernel — two modes matching the forward.
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void rasterize_radar_rae_bwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    // fwd inputs
    const scalar_t *__restrict__ powers,
    const int32_t *__restrict__ range_bin_indices,
    const int32_t *__restrict__ az_bin_indices,
    const int32_t *__restrict__ doppler_bin_indices,
    const scalar_t *__restrict__ gaussian_sigmas_m,  // (legacy)
    const scalar_t *__restrict__ ranges_gauss,
    const scalar_t *__restrict__ azimuths_gauss,
    const scalar_t *__restrict__ range_bin_centers,
    const scalar_t *__restrict__ az_bin_centers,
    const int32_t num_doppler_bins,
    const int32_t num_range_bins,
    const int32_t num_az_bins,
    const scalar_t spread_factor,                    // (legacy)
    const scalar_t *__restrict__ v_r_cont,           // or nullptr
    const scalar_t sigma_D_bins,                     // (legacy)
    const scalar_t doppler_bin_spacing,
    // PSF params
    const scalar_t psf_w_D, const scalar_t psf_k_eff_D,
    const scalar_t psf_w_R, const scalar_t psf_k_eff_R,
    const scalar_t psf_w_A, const scalar_t psf_k_eff_A,
    const int32_t use_physical_dr_psf,
    const int32_t psf_az_cos_broadening,
    // 1 = azimuth kernel is the unit-sum interpolating tent (see forward).
    const int32_t az_tent_delta,
    // Doppler analogue of az_tent_delta (see the forward kernel). tent(x) =
    // max(0, 1-|x|) has d/dx = -sign(x), so the velocity gradient path stays live.
    const int32_t dop_tent_delta,
    // grad output
    const scalar_t *__restrict__ v_rad_tensor,       // [B,C,D,R,A]
    // grad inputs (outputs of this kernel)
    scalar_t *__restrict__ v_powers,
    scalar_t *__restrict__ v_ranges_gauss,
    scalar_t *__restrict__ v_azimuths_gauss,
    scalar_t *__restrict__ v_gaussian_sigmas_m,      // (legacy; zeroed in PSF mode)
    scalar_t *__restrict__ v_v_r_cont                // or nullptr
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N) return;

    const uint32_t bid = idx / (C * N);
    const uint32_t cid = (idx / N) % C;

    const int32_t range_bin   = range_bin_indices[idx];
    const int32_t az_bin      = az_bin_indices[idx];
    const int32_t doppler_bin = doppler_bin_indices[idx];

    if (range_bin < 0 || range_bin >= num_range_bins ||
        az_bin   < 0 || az_bin   >= num_az_bins ||
        doppler_bin < 0 || doppler_bin >= num_doppler_bins) {
        return;
    }

    const scalar_t power       = powers[idx];
    const scalar_t range_mean  = ranges_gauss[idx];
    const scalar_t az_mean     = azimuths_gauss[idx];

    const scalar_t range_bin_size = fabsf(range_bin_centers[1] - range_bin_centers[0]);
    const scalar_t az_bin_size    = fabsf(az_bin_centers[1]    - az_bin_centers[0]);

    const int32_t base = (int32_t)(bid * C * num_doppler_bins * num_range_bins * num_az_bins +
                                   cid * num_doppler_bins * num_range_bins * num_az_bins);

    scalar_t d_power = 0.0f;
    scalar_t d_v_r   = 0.0f;
    scalar_t d_range = 0.0f;
    scalar_t d_az    = 0.0f;
    scalar_t d_sigma = 0.0f;

    // =====================================================================
    // PSF mode
    // =====================================================================
    if (psf_k_eff_D > (scalar_t)0.0f) {

        scalar_t r_cont  = (range_mean - range_bin_centers[0]) / (range_bin_size + (scalar_t)1e-8f);
        scalar_t az_cont = (az_mean    - az_bin_centers[0])    / (az_bin_size    + (scalar_t)1e-8f);

        bool do_v_r = (v_r_cont != nullptr);
        scalar_t d_cont = do_v_r
            ? (v_r_cont[idx] / (doppler_bin_spacing + (scalar_t)1e-12f)
               + (scalar_t)num_doppler_bins * (scalar_t)0.5f /* v = 0 maps to bin D/2 (the DC bin of the rolled GT axis) */)
            : (scalar_t)doppler_bin;
        // Wrap d_cont into [0, num_doppler_bins) — FMCW Doppler is circular
        scalar_t D_f = (scalar_t)num_doppler_bins;
        d_cont = fmodf(d_cont + D_f * (scalar_t)1000.0f, D_f);

        // Tent mode brackets d_cont itself (base = floor(d_cont)); doppler_bin_indices
        // is wrapped mod (D-1) by the wrapper and is not centred. Must match the forward.
        int32_t half_kD = (dop_tent_delta != 0)
            ? 1
            : (use_physical_dr_psf
                // Hamming-FFT support: +/-2 bins; must match the forward kernel.
                ? min(2, num_doppler_bins / 2)
                : min((int32_t)ceilf(psf_k_eff_D), num_doppler_bins - 1));
        int32_t d_base = (dop_tent_delta != 0) ? (int32_t)floorf(d_cont) : doppler_bin;
        int32_t dd_lo  = (dop_tent_delta != 0) ? 0 : -half_kD;
        int32_t dd_hi  = (dop_tent_delta != 0) ? 1 :  half_kD;
        int32_t half_kR = use_physical_dr_psf
            ? 2
            : min((int32_t)ceilf(psf_k_eff_R), num_range_bins   - 1);
        // Azimuth PSF broadening (must match forward): wA(az)=wA0/cos(az). The
        // width's az-dependence is held fixed (stop-grad); only the bin offset is
        // differentiated, consistent with the forward weight evaluation.
        scalar_t w_A_eff   = psf_w_A;
        scalar_t k_eff_A_e = psf_k_eff_A;
        if (psf_az_cos_broadening != 0) {
            scalar_t cos_az = fmaxf(fabsf(__cosf(az_mean)), (scalar_t)0.30f);
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
            scalar_t d_wt;
            if (dop_tent_delta != 0) {
                // tent(x) = max(0, 1 - |x|);  d/dx = -sign(x) inside (mirrors azimuth)
                d_wt = fmaxf((scalar_t)0.0f, (scalar_t)1.0f - fabsf(d_off));
            } else {
                d_wt = use_physical_dr_psf
                    ? hamming_range_psf_bwd<scalar_t>(d_off)
                    : sinc_hann_psf_bwd<scalar_t>(d_off, psf_w_D, psf_k_eff_D);
            }
            if (d_wt < (scalar_t)1e-8f && !do_v_r) continue;
            scalar_t d_d_wt_dx = (scalar_t)0.0f;
            if (do_v_r) {
                if (dop_tent_delta != 0) {
                    d_d_wt_dx = (fabsf(d_off) < (scalar_t)1.0f)
                        ? ((d_off > (scalar_t)0.0f) ? (scalar_t)-1.0f : (scalar_t)1.0f)
                        : (scalar_t)0.0f;
                } else {
                    d_d_wt_dx = use_physical_dr_psf
                        ? d_hamming_range_psf_dx<scalar_t>(d_off)
                        : d_sinc_hann_psf_dx<scalar_t>(d_off, psf_w_D, psf_k_eff_D);
                }
            }

            for (int32_t dr = -half_kR; dr <= half_kR; ++dr) {
                int32_t r_cand = max(0, min(num_range_bins - 1, range_bin + dr));
                scalar_t r_off = r_cont - (scalar_t)r_cand;
                scalar_t r_wt  = use_physical_dr_psf
                    ? hamming_range_psf_bwd<scalar_t>(r_off)
                    : sinc_hann_psf_bwd<scalar_t>(r_off, psf_w_R, psf_k_eff_R);
                if (r_wt < (scalar_t)1e-8f) continue;
                scalar_t d_r_wt_dx = use_physical_dr_psf
                    ? d_hamming_range_psf_dx<scalar_t>(r_off)
                    : d_sinc_hann_psf_dx<scalar_t>(r_off, psf_w_R, psf_k_eff_R);

                for (int32_t da = -half_kA; da <= half_kA; ++da) {
                    int32_t a_cand = max(0, min(num_az_bins - 1, az_bin + da));
                    scalar_t a_off = az_cont - (scalar_t)a_cand;
                    scalar_t a_wt, d_a_wt_dx;
                    if (az_tent_delta != 0) {
                        // tent(x) = max(0, 1 - |x|);  d/dx = -sign(x) inside
                        a_wt = fmaxf((scalar_t)0.0f, (scalar_t)1.0f - fabsf(a_off));
                        d_a_wt_dx = (fabsf(a_off) < (scalar_t)1.0f)
                            ? ((a_off > (scalar_t)0.0f) ? (scalar_t)-1.0f : (scalar_t)1.0f)
                            : (scalar_t)0.0f;
                    } else {
                        a_wt = sinc_hann_psf_bwd<scalar_t>(a_off, w_A_eff, k_eff_A_e);
                        d_a_wt_dx = d_sinc_hann_psf_dx<scalar_t>(a_off, w_A_eff, k_eff_A_e);
                    }
                    if (a_wt < (scalar_t)1e-8f) continue;

                    int32_t tensor_idx = base
                        + d_cand * num_range_bins * num_az_bins
                        + r_cand * num_az_bins
                        + a_cand;
                    const scalar_t g_out_raw = v_rad_tensor[tensor_idx];
                    if (g_out_raw == (scalar_t)0.0f) continue;
                    // Cap g_out: log10(near-zero pred_linear) gives 1e18+ upstream
                    // gradients. Normal signal bins produce |g_out| < 1e3. This cap
                    // kills the explosion while preserving all useful signal.
                    const scalar_t g_out = fmaxf(fminf(g_out_raw, (scalar_t)1e4f),
                                                 -(scalar_t)1e4f);

                    // dL/d(power)
                    d_power += g_out * d_wt * r_wt * a_wt;

                    // dL/d(v_r_cont) via d_cont:  d(d_off)/d(d_cont)=1, d(d_cont)/d(v_r)=1/dv
                    if (do_v_r) {
                        d_v_r += g_out * power * d_d_wt_dx * r_wt * a_wt
                                 / (doppler_bin_spacing + (scalar_t)1e-12f);
                    }

                    // dL/d(ranges_gauss) via r_cont:  d(r_off)/d(range_mean) = 1/range_bin_size
                    d_range += g_out * power * d_wt * d_r_wt_dx * a_wt
                               / (range_bin_size + (scalar_t)1e-8f);

                    // dL/d(azimuths_gauss) via az_cont:  d(a_off)/d(az_mean) = 1/az_bin_size
                    d_az += g_out * power * d_wt * r_wt * d_a_wt_dx
                            / (az_bin_size + (scalar_t)1e-8f);
                }
            }
        }

        v_powers[idx]           = d_power;
        v_ranges_gauss[idx]     = d_range;
        v_azimuths_gauss[idx]   = d_az;
        v_gaussian_sigmas_m[idx] = (scalar_t)0.0f;  // not used in PSF mode
        if (v_v_r_cont != nullptr) v_v_r_cont[idx] = d_v_r;
        return;
    }

    // =====================================================================
    // Legacy mode: Gaussian spreading backward.
    // =====================================================================
    scalar_t sigma_m = gaussian_sigmas_m[idx];
    sigma_m = fmaxf((scalar_t)1e-3f, sigma_m);

    scalar_t estimated_scale_m = fmaxf((scalar_t)0.01f, fminf(sigma_m, (scalar_t)10.0f));

    const scalar_t range_bin_size_leg = (num_range_bins > 1 && range_bin_centers != nullptr)
                                        ? fabsf(range_bin_centers[1] - range_bin_centers[0])
                                        : (scalar_t)0.0596f;
    scalar_t range_spread_bins = estimated_scale_m / (range_bin_size_leg + (scalar_t)1e-6f);
    range_spread_bins = fmaxf((scalar_t)1.0f, fminf(range_spread_bins * spread_factor, (scalar_t)10.0f));

    const scalar_t az_bin_sz = (num_az_bins > 1 && az_bin_centers != nullptr)
                               ? fabsf(az_bin_centers[1] - az_bin_centers[0])
                               : (scalar_t)0.0175f;
    scalar_t angular_spread_rad = estimated_scale_m / (range_mean + (scalar_t)1e-6f);
    scalar_t az_spread_bins = angular_spread_rad / (az_bin_sz + (scalar_t)1e-6f);
    az_spread_bins = fmaxf((scalar_t)1.0f, fminf(az_spread_bins * spread_factor, (scalar_t)10.0f));

    const int32_t range_spread = (int32_t)ceilf(range_spread_bins);
    const int32_t az_spread    = (int32_t)ceilf(az_spread_bins);
    const scalar_t den_r = range_bin_size_leg * range_spread_bins + (scalar_t)1e-6f;
    const scalar_t den_a = az_bin_sz          * az_spread_bins    + (scalar_t)1e-6f;

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

        scalar_t d_wt, d_d_wt_d_v_r;
        if (do_doppler_spread) {
            scalar_t d_delta = d_cont - (scalar_t)d_cand;
            if (d_delta >  D_f_leg * (scalar_t)0.5f) d_delta -= D_f_leg;
            if (d_delta < -D_f_leg * (scalar_t)0.5f) d_delta += D_f_leg;
            scalar_t inv_s   = (scalar_t)1.0f / (sigma_D_bins + (scalar_t)1e-8f);
            d_wt = __expf(-0.5f * (d_delta * inv_s) * (d_delta * inv_s));
            if (d_wt < (scalar_t)1e-6f) continue;
            d_d_wt_d_v_r = d_wt * (-d_delta) * inv_s * inv_s
                           / (doppler_bin_spacing + (scalar_t)1e-12f);
        } else {
            d_wt = (scalar_t)1.0f;
            d_d_wt_d_v_r = (scalar_t)0.0f;
        }

        for (int32_t dr = -range_spread; dr <= range_spread; ++dr) {
            const int32_t r_bin = range_bin + dr;
            if (r_bin < 0 || r_bin >= num_range_bins) continue;
            const scalar_t r_center = (range_bin_centers != nullptr) ? range_bin_centers[r_bin] : range_mean;
            const scalar_t delta_r  = r_center - range_mean;
            const scalar_t sign_r   = (delta_r >= (scalar_t)0.0f) ? (scalar_t)1.0f : (scalar_t)-1.0f;

            for (int32_t da = -az_spread; da <= az_spread; ++da) {
                const int32_t a_bin = az_bin + da;
                if (a_bin < 0 || a_bin >= num_az_bins) continue;

                const scalar_t a_center = (az_bin_centers != nullptr) ? az_bin_centers[a_bin] : az_mean;
                scalar_t delta_a = a_center - az_mean;
                while (delta_a > (scalar_t)M_PI)  delta_a -= (scalar_t)(2.0 * M_PI);
                while (delta_a < (scalar_t)-M_PI) delta_a += (scalar_t)(2.0 * M_PI);
                const scalar_t sign_a = (delta_a >= (scalar_t)0.0f) ? (scalar_t)1.0f : (scalar_t)-1.0f;

                const scalar_t range_norm = fabsf(delta_r) / den_r;
                const scalar_t az_norm    = fabsf(delta_a) / den_a;
                const scalar_t dist_sq    = range_norm * range_norm + az_norm * az_norm;
                const scalar_t w = __expf(-0.5f * dist_sq);
                if (w < (scalar_t)1e-6f) continue;

                const int32_t tensor_idx = base
                    + d_cand * num_range_bins * num_az_bins
                    + r_bin  * num_az_bins
                    + a_bin;
                const scalar_t g_out = v_rad_tensor[tensor_idx];
                if (g_out == (scalar_t)0.0f) continue;

                d_power += g_out * w * d_wt;

                const scalar_t d_w      = g_out * power * d_wt;
                const scalar_t d_dist_sq = d_w * (-0.5f) * w;
                const scalar_t d_range_dist = d_dist_sq * 2.0f * fabsf(delta_r) / (den_r * den_r);
                const scalar_t d_az_dist    = d_dist_sq * 2.0f * fabsf(delta_a) / (den_a * den_a);
                d_range += d_range_dist * (-sign_r);
                d_az    += d_az_dist    * (-sign_a);

                const scalar_t d_den_r = d_dist_sq * (-2.0f) * (fabsf(delta_r) * fabsf(delta_r)) / (den_r * den_r * den_r);
                const scalar_t d_den_a = d_dist_sq * (-2.0f) * (fabsf(delta_a) * fabsf(delta_a)) / (den_a * den_a * den_a);
                const scalar_t d_rspb  = d_den_r * range_bin_size_leg;
                const scalar_t d_aspb  = d_den_a * az_bin_sz;

                const scalar_t rspb_raw = (estimated_scale_m / (range_bin_size_leg + (scalar_t)1e-6f)) * spread_factor;
                if (rspb_raw > (scalar_t)1.0f && rspb_raw < (scalar_t)10.0f)
                    d_sigma += d_rspb * (spread_factor / (range_bin_size_leg + (scalar_t)1e-6f));

                const scalar_t aspb_raw = (estimated_scale_m / (range_mean + (scalar_t)1e-6f))
                                          / (az_bin_sz + (scalar_t)1e-6f) * spread_factor;
                if (aspb_raw > (scalar_t)1.0f && aspb_raw < (scalar_t)10.0f)
                    d_sigma += d_aspb * (spread_factor / ((range_mean + (scalar_t)1e-6f) * (az_bin_sz + (scalar_t)1e-6f)));

                if (sigma_m > (scalar_t)0.01f && sigma_m < (scalar_t)10.0f) {
                    /* d_sigma already accumulated above */
                }
                if (do_doppler_spread) {
                    d_v_r += g_out * power * w * d_d_wt_d_v_r;
                }
            }
        }
    }

    v_powers[idx]            = d_power;
    v_ranges_gauss[idx]      = d_range;
    v_azimuths_gauss[idx]    = d_az;
    v_gaussian_sigmas_m[idx] = (sigma_m > (scalar_t)0.01f && sigma_m < (scalar_t)10.0f) ? d_sigma : (scalar_t)0.0f;
    if (v_v_r_cont != nullptr) v_v_r_cont[idx] = d_v_r;
}


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
) {
    DEVICE_GUARD(powers);
    CHECK_INPUT(powers);
    CHECK_INPUT(range_bin_indices);
    CHECK_INPUT(az_bin_indices);
    CHECK_INPUT(doppler_bin_indices);
    CHECK_INPUT(gaussian_sigmas_m);
    CHECK_INPUT(ranges);
    CHECK_INPUT(azimuths);
    CHECK_INPUT(range_bin_centers);
    CHECK_INPUT(az_bin_centers);
    CHECK_INPUT(v_rad_tensor);

    const uint32_t B = powers.size(0);
    const uint32_t C = powers.size(1);
    const uint32_t N = powers.size(2);

    at::Tensor v_powers  = at::zeros_like(powers);
    at::Tensor v_ranges  = at::zeros_like(ranges);
    at::Tensor v_az      = at::zeros_like(azimuths);
    at::Tensor v_sigmas  = at::zeros_like(gaussian_sigmas_m);
    at::Tensor v_vr_out  = at::zeros_like(powers);

    at::Tensor v_r_cont_c;
    if (v_r_cont.has_value()) {
        CHECK_INPUT(v_r_cont.value());
        v_r_cont_c = v_r_cont.value().contiguous();
    }

    const int threads = 256;
    const int blocks  = (B * C * N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        powers.scalar_type(), "rasterize_radar_rae_bwd", [&] {
            const scalar_t* v_r_cont_ptr = v_r_cont.has_value()
                ? v_r_cont_c.data_ptr<scalar_t>() : nullptr;
            scalar_t* v_vr_ptr = v_r_cont.has_value()
                ? v_vr_out.data_ptr<scalar_t>() : nullptr;

            rasterize_radar_rae_bwd_kernel<scalar_t><<<blocks, threads>>>(
                B, C, N,
                powers.data_ptr<scalar_t>(),
                range_bin_indices.data_ptr<int32_t>(),
                az_bin_indices.data_ptr<int32_t>(),
                doppler_bin_indices.data_ptr<int32_t>(),
                gaussian_sigmas_m.data_ptr<scalar_t>(),
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
                (scalar_t)psf_w_D,    (scalar_t)psf_k_eff_D,
                (scalar_t)psf_w_R,    (scalar_t)psf_k_eff_R,
                (scalar_t)psf_w_A,    (scalar_t)psf_k_eff_A,
                (int32_t)use_physical_dr_psf,
                (int32_t)psf_az_cos_broadening,
                (int32_t)az_tent_delta,
                (int32_t)dop_tent_delta,
                v_rad_tensor.data_ptr<scalar_t>(),
                v_powers.data_ptr<scalar_t>(),
                v_ranges.data_ptr<scalar_t>(),
                v_az.data_ptr<scalar_t>(),
                v_sigmas.data_ptr<scalar_t>(),
                v_vr_ptr
            );
        });

    return std::make_tuple(v_powers, v_ranges, v_az, v_sigmas, v_vr_out);
}

} // namespace gsplat
