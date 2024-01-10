/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;    //线性组合系数
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];    //每一项都是一个基

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f; //为什么要加0.5

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass. 【反向传播时，这部分梯度为0】
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.


	float3 t = transformPoint4x3(mean, viewmatrix); //转换到相机坐标系，后面是转换到ray coordinates,ray coordinates就是NDC空间，直接为正交投影

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;    //出界了也考虑？出界了直接放在边界上
	t.y = min(limy, max(-limy, tytz)) * t.z;

    //相机坐标系到ray坐标系
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);   //投影后第三行用不到，这里简化

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

    //两个矩阵相乘的意义是将右边矩阵中的每一列列向量变换到左边矩阵中每一行行向量为基所表示的空间中去。
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;    //glm::transpose求转置

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.[投影是直接去掉第三维度]
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,   //高斯原点
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,              //geomState.clamped,为了反向传播，clamped部分导数为0[out]
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,    //相机外参数:world_view_transform
	const float* projmatrix,    //投影矩阵:full_proj_transform
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii, //输出，该高斯在当前图像上的半径[out]
	float2* points_xy_image,  //像素坐标[out]
	float* depths,  //深度[out]
	float* cov3Ds,  //3d协方差[out]
	float* rgb,     //颜色[out]
	float4* conic_opacity,  //2d协方差逆+opa[out]
	const dim3 grid,
	uint32_t* tiles_touched,    //覆盖到的瓦片数量[out]
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();   //当前线程在grid的序号
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside. 去掉near<0.2,只根据距离进行判断，而非锥体
	float3 p_view;  //相机坐标系下的坐标
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);   //full_proj_transform, 世界坐标到NDC
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };    //齐次坐标到欧式坐标【NDC空间】

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);    //rotations使用四元数存储，论文公式（6）
		cov3D = cov3Ds + idx * 6; //对称矩阵，存上6个
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix); //像素平面.这个函数的原理没有看太懂

	// Invert covariance (EWA algorithm)，高斯函数协方差矩阵是逆
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det)); //特征值1：长轴
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det)); //特征值2：短轴
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2))); //3.f将高斯分布正负3个标准差全都覆盖
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };  //转换到像素坐标????p_proj并没有转换到相机坐标系啊？？？
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);  //计算每个圆形覆盖了哪些像素[最小最大tail的index]
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z; //相机坐标系
	radii[idx] = my_radius; //覆盖半径
	points_xy_image[idx] = point_image; //图像上的位置
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);//计算出覆盖了几个tiles,只记录个数？
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features, //颜色
	const float4* __restrict__ conic_opacity,
    const float* __restrict__ depths,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
    float* __restrict__ depth_map,
    float* __restrict__ weight_map )
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();   //当前线程的thread_block
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

    //dim3 group_index();  //网格grid内3维索引（当前线程所在block索引(x,y,z)）
    //dim3 thread_index();  //块block内3维索引（当前线程索引（x,y,z））
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };   //
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y }; //整个图像层面
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;    //边界处理，边缘tail会补全，但是这部分没有像素，因此需要跳过
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;    //对于外部的像素，直接认为是done

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];    //当前block（瓦片）对应的全部高斯
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);     //每轮256个线程，N个高斯需要处理多少轮，每个线程一个高斯
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];    //同一个block
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;             //累计透明度
	uint32_t contributor = 0;   //总共经过多少个gaussian，用于早停
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };  //最终颜色
    float Dep = 0;
    float Wei = 0;

	// Iterate over batches until all done or range is complete[每次256个像素]
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)    //每个线程
	{
		// End if entire block votes that it is done rasterizing
        //将GS读取到共享内存中
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE) //全部都计算完成了，done；这里的num_done指定是像素层面
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank(); //当前线程的 序号：处理第几个GS;thread_rank：将（x,y,z）转换为序号
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];   //根据线程号找到对应的高斯
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];   //高斯在像素上的位置
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();   //同步：将256个高斯都读到共享内存中农

        //该像素处理完共享显存中的全部高斯
		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)    //此时j为tils-gs,线程为像素，每次256个线程处理同一个高斯【覆盖tile内多个像素】
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;    //高斯的概率值
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;  //太小，认为是透明
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) //早停
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
            Dep += depths[collected_id[j]] * alpha * T;
            Wei += alpha * T;
			T = test_T;   //累计透明度

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		Wei += 1e-6;
        depth_map[pix_id]= Dep/Wei;
        weight_map[pix_id] = Wei;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
    const float* depths,
	float* final_T, //imgState.accum_alpha,
	uint32_t* n_contrib,    //imgState.n_contrib,
	const float* bg_color,
	float* out_color,
    float* depth_map,
    float* weight_map
)
{
        //grid = (tile_x,tile_y),block = (block_x,block_y,1)
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
        depths,
		final_T,
		n_contrib,
		bg_color,
		out_color,
        depth_map,
        weight_map );
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,  //geomState.clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,     // geomState.internal_radii
	float2* means2D,    //geomState.means2D,
	float* depths,      //geomState.depths,
	float* cov3Ds,      //geomState.cov3D,
	float* rgb,         //geomState.rgb,
	float4* conic_opacity,  //geomState.conic_opacity,
	const dim3 grid,    //tile_grid,
	uint32_t* tiles_touched,    //geomState.tiles_touched,
	bool prefiltered)
{
    //预处理阶段：3d椭球到2d椭圆，计算覆盖哪些像素【逐高斯处理】
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (    //(按照高斯数量进行划分)
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}