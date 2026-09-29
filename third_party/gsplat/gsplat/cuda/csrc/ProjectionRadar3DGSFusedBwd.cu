#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cmath>
#
#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"
#
namespace gsplat {
namespace cg = cooperative_groups;
#
template <typename scalar_t>
__global__ void projection_radar_3dgs_fused_bwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    // fwd inputs
    const scalar_t *__restrict__ means,    // [B,N,3]
    const scalar_t *__restrict__ quats,    // [B,N,4]
    const scalar_t *__restrict__ scales,   // [B,N,3]
    const scalar_t *__restrict__ viewmats, // [B,C,4,4]
    const scalar_t near_range,
    const scalar_t far_range,
    // fwd outputs (needed for stable backward)
    const scalar_t *__restrict__ ranges,     // [B,C,N]
    const scalar_t *__restrict__ azimuths,   // [B,C,N]
    const scalar_t *__restrict__ elevations, // [B,C,N]
    const scalar_t *__restrict__ sigma_m,    // [B,C,N] (gaussian_weights)
    const int32_t *__restrict__ range_bin_indices, // [B,C,N] (for validity)
    // upstream gradients
    const scalar_t *__restrict__ v_ranges,     // [B,C,N]
    const scalar_t *__restrict__ v_azimuths,   // [B,C,N]
    const scalar_t *__restrict__ v_elevations, // [B,C,N]
    const scalar_t *__restrict__ v_sigma_m,    // [B,C,N]
    // outputs (grads)
    scalar_t *__restrict__ v_means,  // [B,N,3]
    scalar_t *__restrict__ v_quats,  // [B,N,4]
    scalar_t *__restrict__ v_scales  // [B,N,3]
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N) return;
#
    const uint32_t bid = idx / (C * N);
    const uint32_t cid = (idx / N) % C;
    const uint32_t gid = idx % N;
#
    // If fwd marked invalid (out of range), skip (no contribution).
    if (range_bin_indices[idx] < 0) {
        return;
    }
#
    // Load view matrix for this (b,c)
    const scalar_t *vm = viewmats + bid * C * 16 + cid * 16;
    // glm is column-major but input is row-major
    mat3 R = mat3(
        vm[0], vm[4], vm[8],
        vm[1], vm[5], vm[9],
        vm[2], vm[6], vm[10]
    );
    vec3 t = vec3(vm[3], vm[7], vm[11]);
#
    // mean in world
    const scalar_t *m_ptr = means + bid * N * 3 + gid * 3;
    vec3 mean_w = glm::make_vec3(m_ptr);
    vec3 mean_r;
    posW2C(R, t, mean_w, mean_r);
#
    const scalar_t x = (scalar_t)mean_r.x;
    const scalar_t y = (scalar_t)mean_r.y;
    const scalar_t z = (scalar_t)mean_r.z;
#
    const scalar_t r = ranges[idx]; // already computed in fwd
    if (r < near_range || r > far_range) return;
#
    // Upstream grads
    const scalar_t gr = v_ranges ? v_ranges[idx] : (scalar_t)0.0;
    const scalar_t gaz = v_azimuths ? v_azimuths[idx] : (scalar_t)0.0;
    const scalar_t gel = v_elevations ? v_elevations[idx] : (scalar_t)0.0;
#
    // Backprop spherical coords -> mean_r
    // range = sqrt(x^2+y^2+z^2)
    const scalar_t inv_r = (scalar_t)1.0 / (r + (scalar_t)1e-10);
#
    // azimuth = atan2(y, x)
    const scalar_t inv_xy2 = (scalar_t)1.0 / (x * x + y * y + (scalar_t)1e-10);
#
    // elevation = asin(z/r)
    const scalar_t u = z * inv_r; // z/r
    const scalar_t inv_sqrt_1_u2 = rsqrtf(max((scalar_t)1e-10, (scalar_t)1.0 - u * u));
#
    // d range / d (x,y,z) = (x,y,z)/r
    scalar_t vx = gr * x * inv_r;
    scalar_t vy = gr * y * inv_r;
    scalar_t vz = gr * z * inv_r;
#
    // d az / d x = -y/(x^2+y^2), d az / d y = x/(x^2+y^2)
    vx += gaz * (-y) * inv_xy2;
    vy += gaz * (x) * inv_xy2;
#
    // elevation = asin(u), u=z/r
    // du/dx = -z*x/r^3, du/dy = -z*y/r^3, du/dz = (x^2+y^2)/r^3
    const scalar_t inv_r3 = inv_r * inv_r * inv_r;
    const scalar_t du_dx = -z * x * inv_r3;
    const scalar_t du_dy = -z * y * inv_r3;
    const scalar_t du_dz = (x * x + y * y) * inv_r3;
    vx += gel * inv_sqrt_1_u2 * du_dx;
    vy += gel * inv_sqrt_1_u2 * du_dy;
    vz += gel * inv_sqrt_1_u2 * du_dz;
#
    // mean_r = R * mean_w + t  => v_mean_w = R^T * v_mean_r
    vec3 v_mean_r(vx, vy, vz);
    vec3 v_mean_w = glm::transpose(R) * v_mean_r;
#
    // Accumulate into v_means (across cameras) with atomic adds
    scalar_t *v_means_ptr = v_means + bid * N * 3 + gid * 3;
    atomicAdd(&v_means_ptr[0], (scalar_t)v_mean_w.x);
    atomicAdd(&v_means_ptr[1], (scalar_t)v_mean_w.y);
    atomicAdd(&v_means_ptr[2], (scalar_t)v_mean_w.z);
#
    // Backprop sigma_m (meters) -> (quat, scale) via covariance trace
    const scalar_t gs = v_sigma_m ? v_sigma_m[idx] : (scalar_t)0.0;
    if (gs != (scalar_t)0.0) {
        const scalar_t sig = max(sigma_m[idx], (scalar_t)1e-6);
        // sigma = sqrt(trace/3)  => d sigma / d trace = 1/(6*sigma)
        const scalar_t v_trace = gs * ((scalar_t)1.0 / ((scalar_t)6.0 * sig));
#
        // v_covar_r = v_trace * I
        mat3 v_covar_r = mat3(v_trace, 0.f, 0.f,
                              0.f, v_trace, 0.f,
                              0.f, 0.f, v_trace);
#
        // covar_r = R * covar_w * R^T => v_covar_w = R^T * v_covar_r * R
        mat3 v_covar_w = glm::transpose(R) * v_covar_r * R;
#
        // Load quat/scale for this gaussian
        const scalar_t *q_ptr = quats + bid * N * 4 + gid * 4;
        const scalar_t *s_ptr = scales + bid * N * 3 + gid * 3;
        vec4 quat = glm::make_vec4(q_ptr);
        vec3 scale = glm::make_vec3(s_ptr);
#
        // Match fwd normalization
        float qn = glm::length(quat);
        if (qn > 1e-8f) {
            quat = quat / qn;
        }
        mat3 rotmat = quat_to_rotmat(quat);
#
        vec4 v_quat_local(0.f);
        vec3 v_scale_local(0.f);
        quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar_w, v_quat_local, v_scale_local);
#
        // Accumulate to outputs (across cameras) with atomics
        scalar_t *v_q_ptr = v_quats + bid * N * 4 + gid * 4;
        scalar_t *v_s_ptr = v_scales + bid * N * 3 + gid * 3;
        atomicAdd(&v_s_ptr[0], (scalar_t)v_scale_local.x);
        atomicAdd(&v_s_ptr[1], (scalar_t)v_scale_local.y);
        atomicAdd(&v_s_ptr[2], (scalar_t)v_scale_local.z);
        atomicAdd(&v_q_ptr[0], (scalar_t)v_quat_local.x);
        atomicAdd(&v_q_ptr[1], (scalar_t)v_quat_local.y);
        atomicAdd(&v_q_ptr[2], (scalar_t)v_quat_local.z);
        atomicAdd(&v_q_ptr[3], (scalar_t)v_quat_local.w);
    }
}
#
std::tuple<at::Tensor, at::Tensor, at::Tensor> projection_radar_3dgs_fused_bwd(
    const at::Tensor means,           // [B,N,3]
    const at::Tensor quats,           // [B,N,4]
    const at::Tensor scales,          // [B,N,3]
    const at::Tensor viewmats,        // [B,C,4,4]
    const float near_range,
    const float far_range,
    const at::Tensor ranges,          // [B,C,N]
    const at::Tensor azimuths,        // [B,C,N]
    const at::Tensor elevations,      // [B,C,N]
    const at::Tensor sigma_m,         // [B,C,N]
    const at::Tensor range_bin_indices, // [B,C,N]
    const at::Tensor v_ranges,        // [B,C,N]
    const at::Tensor v_azimuths,      // [B,C,N]
    const at::Tensor v_elevations,    // [B,C,N]
    const at::Tensor v_sigma_m        // [B,C,N]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(ranges);
    CHECK_INPUT(azimuths);
    CHECK_INPUT(elevations);
    CHECK_INPUT(sigma_m);
    CHECK_INPUT(range_bin_indices);
    CHECK_INPUT(v_ranges);
    CHECK_INPUT(v_azimuths);
    CHECK_INPUT(v_elevations);
    CHECK_INPUT(v_sigma_m);
#
    const uint32_t B = means.size(0);
    const uint32_t C = viewmats.size(1);
    const uint32_t N = means.size(1);
#
    at::Tensor v_means = at::zeros_like(means);
    at::Tensor v_quats = at::zeros_like(quats);
    at::Tensor v_scales = at::zeros_like(scales);
#
    const int threads = 256;
    const int blocks = (B * C * N + threads - 1) / threads;
#
    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(), "projection_radar_3dgs_fused_bwd", [&] {
            projection_radar_3dgs_fused_bwd_kernel<scalar_t><<<blocks, threads>>>(
                B, C, N,
                means.data_ptr<scalar_t>(),
                quats.data_ptr<scalar_t>(),
                scales.data_ptr<scalar_t>(),
                viewmats.data_ptr<scalar_t>(),
                (scalar_t)near_range,
                (scalar_t)far_range,
                ranges.data_ptr<scalar_t>(),
                azimuths.data_ptr<scalar_t>(),
                elevations.data_ptr<scalar_t>(),
                sigma_m.data_ptr<scalar_t>(),
                range_bin_indices.data_ptr<int32_t>(),
                v_ranges.data_ptr<scalar_t>(),
                v_azimuths.data_ptr<scalar_t>(),
                v_elevations.data_ptr<scalar_t>(),
                v_sigma_m.data_ptr<scalar_t>(),
                v_means.data_ptr<scalar_t>(),
                v_quats.data_ptr<scalar_t>(),
                v_scales.data_ptr<scalar_t>()
            );
        });
#
    return std::make_tuple(v_means, v_quats, v_scales);
}
#
} // namespace gsplat



