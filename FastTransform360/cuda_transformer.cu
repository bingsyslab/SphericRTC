#include "transformer.h"

texture<uint8_t, 2, cudaReadModeElementType> tex_ref; // TEXTURE REFERENCE MUST BE A GLOBAL VARIABLE
const int THREAD_LIMIT = 960;
const int PARAM_NUM = 9;
__constant__ float mat[2 * PARAM_NUM];
__constant__ float cube[6 * PARAM_NUM] = {0.5, 0.5, 0.5, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0,
                                          0.5, 0.5, -0.5, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
                                          -0.5, 0.5, -0.5, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0,
                                          -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                          -0.5, -0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                                          -0.5, 0.5, 0.5, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0};

__device__ float CUDADividedByPi(float val) {return 0.3183098861837907 * val;}

__device__ float CUDATimesPi(float val) {return 3.141592653589793 * val;}

__global__ void CUDAScaleKernel(uint8_t* transformed_data,
                                int w,
                                float rw,
                                float rh,
                                int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    transformed_data[i] = tex2D(tex_ref,
        (__int2float_rn(i % w) + 0.5) * rw,
        (__int2float_rn(i / w) + 0.5) * rh);
  }
}

__global__ void CUDARotateKernel(uint8_t* transformed_data,
                                 int w,
                                 float rw,
                                 float rh,
                                 int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float su, cu, sv, cv;
    sincospif(__fmaf_rn(__fmaf_rn(2.0, __int2float_rn(i % w), 1.0), rw, -1.0), &su, &cu);
    sincospif(-__fmaf_rn(__fmaf_rn(1.0, __int2float_rn(i / w), 0.5), rh, -0.5), &sv, &cv);
    float x = -su * cv, y = sv, z = -cu * cv;
    transformed_data[i] = tex2D(tex_ref,
        CUDADividedByPi(0.5 * atan2f(__fmaf_rn(mat[0], x,
                                     __fmaf_rn(mat[1], y,
                                               mat[2] * z)),
                                     __fmaf_rn(mat[6], x,
                                     __fmaf_rn(mat[7], y,
                                               mat[8] * z)))),
        CUDADividedByPi(acosf(__fmaf_rn(mat[3], x,
                              __fmaf_rn(mat[4], y,
                                        mat[5] * z)))));
  }
}

__global__ void CUDATransformKernelOERP(uint8_t* transformed_data,
                                        int w,
                                        float rw,
                                        float rh,
                                        float dx,
                                        float dy,
                                        float dz,
                                        int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float su, cu, sv, cv;
    sincospif(__fmaf_rn(__fmaf_rn(2.0, __int2float_rn(i % w), 1.0), rw, -1.0), &su, &cu);
    sincospif(-__fmaf_rn(__fmaf_rn(1.0, __int2float_rn(i / w), 0.5), rh, -0.5), &sv, &cv);
    float x = -su * cv, y = sv, z = -cu * cv;
    float m = __fmaf_rn(x, dx, __fmaf_rn(y, dy, z * dz));
    float t = m + __fsqrt_rn(__fmaf_rn(m, m, 1.0 -
                             __fmaf_rn(dx, dx,
                             __fmaf_rn(dy, dy, dz * dz))));
    x = __fmaf_rn(t, x, -dx);
    y = __fmaf_rn(t, y, -dy);
    z = __fmaf_rn(t, z, -dz);
    float l = __fmaf_rn(x, x, __fmaf_rn(y, y, z * z));
    transformed_data[i] = tex2D(tex_ref,
        CUDADividedByPi(0.5 * atan2f(__fmaf_rn(mat[0], x,
                                     __fmaf_rn(mat[1], y,
                                               mat[2] * z)),
                                     __fmaf_rn(mat[6], x,
                                     __fmaf_rn(mat[7], y,
                                               mat[8] * z)))),
        CUDADividedByPi(acosf(__frsqrt_rn(l) *
                              __fmaf_rn(mat[3], x,
                              __fmaf_rn(mat[4], y,
                                        mat[5] * z)))));
  }
}

__global__ void CUDATransformKernelCMP(uint8_t* transformed_data,
                                       int w,
                                       float rw,
                                       float rh,
                                       float ecoef,
                                       int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float u = (__int2float_rn(i % w) + 0.5) * rw,
          v = (__int2float_rn(i / w) + 0.5) * rh;
    int fu = __float2int_rd(3.0 * u),
        fv = __float2int_rd(2.0 * v);
    u = __fmaf_rn(__fmaf_rn(3.0, u, -__int2float_rn(fu)), ecoef, -__fmaf_rn(0.5, ecoef, -0.5));
    v = __fmaf_rn(__fmaf_rn(2.0, v, -__int2float_rn(fv)), ecoef, -__fmaf_rn(0.5, ecoef, -0.5));
    int s = 9 * (3 * fv + fu);
    float x = __fmaf_rn(cube[s + 6], v,
              __fmaf_rn(cube[s + 3], u,
                        cube[s])),
          y = __fmaf_rn(cube[s + 7], v,
              __fmaf_rn(cube[s + 4], u,
                        cube[s + 1])),
          z = __fmaf_rn(cube[s + 8], v,
              __fmaf_rn(cube[s + 5], u,
                        cube[s + 2]));
    float l = __frsqrt_rn(__fmaf_rn(x, x, __fmaf_rn(y, y, z * z)));
    x *= l; y *= l; z *= l;
    transformed_data[i] = tex2D(tex_ref,
        CUDADividedByPi(0.5 * atan2f(__fmaf_rn(mat[0], x,
                                     __fmaf_rn(mat[1], y,
                                               mat[2] * z)),
                                     __fmaf_rn(mat[6], x,
                                     __fmaf_rn(mat[7], y,
                                               mat[8] * z)))),
        CUDADividedByPi(acosf(__fmaf_rn(mat[3], x,
                              __fmaf_rn(mat[4], y,
                                        mat[5] * z)))));
  }
}

__global__ void CUDATransformKernelOCMP(uint8_t* transformed_data,
                                        int w,
                                        float rw,
                                        float rh,
                                        float ecoef,
                                        float dx,
                                        float dy,
                                        float dz,
                                        int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float u = (__int2float_rn(i % w) + 0.5) * rw,
          v = (__int2float_rn(i / w) + 0.5) * rh;
    int fu = __float2int_rd(3.0 * u),
        fv = __float2int_rd(2.0 * v);
    u = __fmaf_rn(__fmaf_rn(3.0, u, -__int2float_rn(fu)), ecoef, -__fmaf_rn(0.5, ecoef, -0.5));
    v = __fmaf_rn(__fmaf_rn(2.0, v, -__int2float_rn(fv)), ecoef, -__fmaf_rn(0.5, ecoef, -0.5));
    int s = 9 * (3 * fv + fu);
    float x = __fmaf_rn(cube[s + 6], v,
              __fmaf_rn(cube[s + 3], u,
                        cube[s])),
          y = __fmaf_rn(cube[s + 7], v,
              __fmaf_rn(cube[s + 4], u,
                        cube[s + 1])),
          z = __fmaf_rn(cube[s + 8], v,
              __fmaf_rn(cube[s + 5], u,
                        cube[s + 2]));
    float l = __frsqrt_rn(__fmaf_rn(x, x, __fmaf_rn(y, y, z * z)));
    x *= l; y *= l; z *= l;
    float m = __fmaf_rn(x, dx, __fmaf_rn(y, dy, z * dz));
    float t = m + __fsqrt_rn(__fmaf_rn(m, m, 1.0 -
                             __fmaf_rn(dx, dx,
                             __fmaf_rn(dy, dy, dz * dz))));
    x = __fmaf_rn(t, x, -dx);
    y = __fmaf_rn(t, y, -dy);
    z = __fmaf_rn(t, z, -dz);
    l = __fmaf_rn(x, x, __fmaf_rn(y, y, z * z));
    transformed_data[i] = tex2D(tex_ref,
        CUDADividedByPi(0.5 * atan2f(__fmaf_rn(mat[0], x,
                                     __fmaf_rn(mat[1], y,
                                               mat[2] * z)),
                                     __fmaf_rn(mat[6], x,
                                     __fmaf_rn(mat[7], y,
                                               mat[8] * z)))),
        CUDADividedByPi(acosf(__frsqrt_rn(l) *
                              __fmaf_rn(mat[3], x,
                              __fmaf_rn(mat[4], y,
                                        mat[5] * z)))));
  }
}

__global__ void CUDATransformKernelEAC(uint8_t* transformed_data,
                                       int w,
                                       float rw,
                                       float rh,
                                       float ecoef,
                                       int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float u = (__int2float_rn(i % w) + 0.5) * rw,
          v = (__int2float_rn(i / w) + 0.5) * rh;
    int fu = __float2int_rd(3.0 * u),
        fv = __float2int_rd(2.0 * v);
    u = __fmaf_rn(0.5 * ecoef, tanf(CUDATimesPi(__fmaf_rn(1.5, u, -__fmaf_rn(0.5, __int2float_rn(fu), 0.25)))), 0.5);
    v = __fmaf_rn(0.5 * ecoef, tanf(CUDATimesPi(__fmaf_rn(1.0, v, -__fmaf_rn(0.5, __int2float_rn(fv), 0.25)))), 0.5);
    int s = 9 * (3 * fv + fu);
    float x = __fmaf_rn(cube[s + 6], v,
              __fmaf_rn(cube[s + 3], u,
                        cube[s])),
          y = __fmaf_rn(cube[s + 7], v,
              __fmaf_rn(cube[s + 4], u,
                        cube[s + 1])),
          z = __fmaf_rn(cube[s + 8], v,
              __fmaf_rn(cube[s + 5], u,
                        cube[s + 2]));
    float l = __frsqrt_rn(__fmaf_rn(x, x, __fmaf_rn(y, y, z * z)));
    x *= l; y *= l; z *= l;
    transformed_data[i] = tex2D(tex_ref,
        CUDADividedByPi(0.5 * atan2f(__fmaf_rn(mat[0], x,
                                     __fmaf_rn(mat[1], y,
                                               mat[2] * z)),
                                     __fmaf_rn(mat[6], x,
                                     __fmaf_rn(mat[7], y,
                                               mat[8] * z)))),
        CUDADividedByPi(acosf(__fmaf_rn(mat[3], x,
                              __fmaf_rn(mat[4], y,
                                        mat[5] * z)))));
  }
}

__global__ void CUDATransformKernelOEAC(uint8_t* transformed_data,
                                        int w,
                                        float rw,
                                        float rh,
                                        float ecoef,
                                        float dx,
                                        float dy,
                                        float dz,
                                        int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float u = (__int2float_rn(i % w) + 0.5) * rw,
          v = (__int2float_rn(i / w) + 0.5) * rh;
    int fu = __float2int_rd(3.0 * u),
        fv = __float2int_rd(2.0 * v);
    u = __fmaf_rn(0.5 * ecoef, tanf(CUDATimesPi(__fmaf_rn(1.5, u, -__fmaf_rn(0.5, __int2float_rn(fu), 0.25)))), 0.5);
    v = __fmaf_rn(0.5 * ecoef, tanf(CUDATimesPi(__fmaf_rn(1.0, v, -__fmaf_rn(0.5, __int2float_rn(fv), 0.25)))), 0.5);
    int s = 9 * (3 * fv + fu);
    float x = __fmaf_rn(cube[s + 6], v,
              __fmaf_rn(cube[s + 3], u,
                        cube[s])),
          y = __fmaf_rn(cube[s + 7], v,
              __fmaf_rn(cube[s + 4], u,
                        cube[s + 1])),
          z = __fmaf_rn(cube[s + 8], v,
              __fmaf_rn(cube[s + 5], u,
                        cube[s + 2]));
    float l = __frsqrt_rn(__fmaf_rn(x, x, __fmaf_rn(y, y, z * z)));
    x *= l; y *= l; z *= l;
    float m = __fmaf_rn(x, dx, __fmaf_rn(y, dy, z * dz));
    float t = m + __fsqrt_rn(__fmaf_rn(m, m, 1.0 -
                             __fmaf_rn(dx, dx,
                             __fmaf_rn(dy, dy, dz * dz))));
    x = __fmaf_rn(t, x, -dx);
    y = __fmaf_rn(t, y, -dy);
    z = __fmaf_rn(t, z, -dz);
    l = __fmaf_rn(x, x, __fmaf_rn(y, y, z * z));
    transformed_data[i] = tex2D(tex_ref,
        CUDADividedByPi(0.5 * atan2f(__fmaf_rn(mat[0], x,
                                     __fmaf_rn(mat[1], y,
                                               mat[2] * z)),
                                     __fmaf_rn(mat[6], x,
                                     __fmaf_rn(mat[7], y,
                                               mat[8] * z)))),
        CUDADividedByPi(acosf(__frsqrt_rn(l) *
                              __fmaf_rn(mat[3], x,
                              __fmaf_rn(mat[4], y,
                                        mat[5] * z)))));
  }
}

__global__ void CUDATransformKernelBEP(uint8_t* transformed_data,
                                       int w,
                                       float rw,
                                       float rh,
                                       float ecoef,
                                       int s,
                                       int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float su, cu, sv, cv;
    sincospif(__fmaf_rn(__fmaf_rn(1.5, __int2float_rn(i % w), 0.75), rw, -0.75), &su, &cu);
    sincospif(__fmaf_rn(__fmaf_rn(0.5, __int2float_rn(i / w), 0.25), rh, -0.25) * -ecoef, &sv, &cv);
    float x = -su * cv, y = sv, z = -cu * cv;
    transformed_data[i] = tex2D(tex_ref,
        CUDADividedByPi(0.5 * atan2f(__fmaf_rn(mat[s], x,
                                     __fmaf_rn(mat[s + 1], y,
                                               mat[s + 2] * z)),
                                     __fmaf_rn(mat[s + 6], x,
                                     __fmaf_rn(mat[s + 7], y,
                                               mat[s + 8] * z)))),
        CUDADividedByPi(acosf(__fmaf_rn(mat[s + 3], x,
                              __fmaf_rn(mat[s + 4], y,
                                        mat[s + 5] * z)))));
  }
}

__global__ void CUDARenderKernel() {}

float Radians(float degrees) {return 0.017453292519943295 * degrees;}

void RotationMatrix(float y, float p, float r, float* out) {
  float sy = sin(y), cy = cos(y),
        sp = sin(p), cp = cos(p),
        sr = sin(r), cr = cos(r);
  *out++ = sy * sp * sr + cy * cr;
  *out++ = sy * sp * cr - cy * sr;
  *out++ = sy * cp;
  *out++ = cp * sr;
  *out++ = cp * cr;
  *out++ = -sp;
  *out++ = cy * sp * sr - sy * cr;
  *out++ = cy * sp * cr + sy * sr;
  *out = cy * cp;
}

int CUDAScaleWrapper(const uint8_t* data,
                     int width,
                     int height,
                     int target_width,
                     int target_height,
                     uint8_t* out_data) {
  cudaArray* cuArray;
  cudaMallocArray(&cuArray, &tex_ref.channelDesc, width, height);
  cudaMemcpyToArray(cuArray, 0, 0, data, width * height, cudaMemcpyHostToDevice);
  cudaBindTextureToArray(tex_ref, cuArray);
  tex_ref.addressMode[0] = cudaAddressModeWrap;
  tex_ref.addressMode[1] = cudaAddressModeWrap;
  tex_ref.filterMode = cudaFilterModePoint; // "cudaFilterModeLinear" ONLY USED WITH FLOAT TEXTURE
  tex_ref.normalized = true;
  int n = target_width * target_height;
  float rw = 1.0 / target_width, rh = 1.0 / target_height;
  uint8_t* cuda_data = NULL;
  cudaMalloc((void**)&cuda_data, n);
  int nt = min(max(target_width >> 1, target_height), THREAD_LIMIT);
  CUDAScaleKernel<<<n / nt, nt>>>(cuda_data,
                                  target_width,
                                  rw, rh, n);
  cudaDeviceSynchronize();
  // fprintf(stderr, "%s\n", cudaGetErrorName(cudaGetLastError())); // DEBUG
  cudaMemcpy(out_data, cuda_data, n, cudaMemcpyDeviceToHost);
  cudaFree(cuda_data);
  cudaUnbindTexture(tex_ref);
  cudaFreeArray(cuArray);
  return 0;
}

int CUDARotateWrapper(const uint8_t* data,
                      int width,
                      int height,
                      int target_width,
                      int target_height,
                      float yaw,
                      float pitch,
                      float roll,
                      uint8_t* out_data) {
  cudaArray* cuArray;
  cudaMallocArray(&cuArray, &tex_ref.channelDesc, width, height);
  cudaMemcpyToArray(cuArray, 0, 0, data, width * height, cudaMemcpyHostToDevice);
  cudaBindTextureToArray(tex_ref, cuArray);
  tex_ref.addressMode[0] = cudaAddressModeWrap;
  tex_ref.addressMode[1] = cudaAddressModeWrap;
  tex_ref.filterMode = cudaFilterModePoint; // "cudaFilterModeLinear" ONLY USED WITH FLOAT TEXTURE
  tex_ref.normalized = true;
  float* host_mat = (float*)malloc(PARAM_NUM * sizeof(float));
  RotationMatrix(Radians(yaw), Radians(pitch), Radians(-roll), host_mat);
  cudaMemcpyToSymbol(mat, host_mat, PARAM_NUM * sizeof(float));
  int n = target_width * target_height;
  float rw = 1.0 / target_width, rh = 1.0 / target_height;
  uint8_t* cuda_data = NULL;
  cudaMalloc((void**)&cuda_data, n);
  int nt = min(max(target_width >> 1, target_height), THREAD_LIMIT);
  CUDARotateKernel<<<n / nt, nt>>>(cuda_data,
                                   target_width,
                                   rw, rh, n);
  cudaDeviceSynchronize();
  // fprintf(stderr, "%s\n", cudaGetErrorName(cudaGetLastError())); // DEBUG
  cudaMemcpy(out_data, cuda_data, n, cudaMemcpyDeviceToHost);
  cudaFree(cuda_data);
  free(host_mat);
  cudaUnbindTexture(tex_ref);
  cudaFreeArray(cuArray);
  return 0;
}

int CUDATransformWrapper(const uint8_t* data,
                         int width,
                         int height,
                         int target_width,
                         int target_height,
                         float yaw,
                         float pitch,
                         float roll,
                         float x,
                         float y,
                         float z,
                         float ecoef,
                         uint8_t* out_data,
                         int flag) {
  switch (flag) {
    case OFFSET_EQUIRECT_21:
    case OFFSET_EQUIRECT:
    {
      cudaArray* cuArray;
      cudaMallocArray(&cuArray, &tex_ref.channelDesc, width, height);
      cudaMemcpyToArray(cuArray, 0, 0, data, width * height, cudaMemcpyHostToDevice);
      cudaBindTextureToArray(tex_ref, cuArray);
      tex_ref.addressMode[0] = cudaAddressModeWrap;
      tex_ref.addressMode[1] = cudaAddressModeWrap;
      tex_ref.filterMode = cudaFilterModePoint; // "cudaFilterModeLinear" ONLY USED WITH FLOAT TEXTURE
      tex_ref.normalized = true;
      float* host_mat = (float*)malloc(PARAM_NUM * sizeof(float));
      RotationMatrix(Radians(yaw), Radians(pitch), Radians(-roll), host_mat);
      cudaMemcpyToSymbol(mat, host_mat, PARAM_NUM * sizeof(float));
      int n = target_width * target_height;
      float rw = 1.0 / target_width, rh = 1.0 / target_height;
      uint8_t* cuda_data = NULL;
      cudaMalloc((void**)&cuda_data, n);
      int nt = min(max(target_width >> 1, target_height), THREAD_LIMIT);
      CUDATransformKernelOERP<<<n / nt, nt>>>(cuda_data,
                                              target_width,
                                              rw, rh,
                                              x, y, z, n);
      cudaDeviceSynchronize();
      // fprintf(stderr, "%s\n", cudaGetErrorName(cudaGetLastError())); // DEBUG
      cudaMemcpy(out_data, cuda_data, n, cudaMemcpyDeviceToHost);
      cudaFree(cuda_data);
      free(host_mat);
      cudaUnbindTexture(tex_ref);
      cudaFreeArray(cuArray);
      break;
    }
    case CUBEMAP_32:
    case CUBEMAP:
    {
      cudaArray* cuArray;
      cudaMallocArray(&cuArray, &tex_ref.channelDesc, width, height);
      cudaMemcpyToArray(cuArray, 0, 0, data, width * height, cudaMemcpyHostToDevice);
      cudaBindTextureToArray(tex_ref, cuArray);
      tex_ref.addressMode[0] = cudaAddressModeWrap;
      tex_ref.addressMode[1] = cudaAddressModeWrap;
      tex_ref.filterMode = cudaFilterModePoint; // "cudaFilterModeLinear" ONLY USED WITH FLOAT TEXTURE
      tex_ref.normalized = true;
      float* host_mat = (float*)malloc(PARAM_NUM * sizeof(float));
      RotationMatrix(Radians(yaw), Radians(pitch), Radians(-roll), host_mat);
      cudaMemcpyToSymbol(mat, host_mat, PARAM_NUM * sizeof(float));
      int n = target_width * target_height;
      float rw = 1.0 / target_width, rh = 1.0 / target_height;
      uint8_t* cuda_data = NULL;
      cudaMalloc((void**)&cuda_data, n);
      int nt = min(max(target_width / 3, target_height >> 1), THREAD_LIMIT);
      CUDATransformKernelCMP<<<n / nt, nt>>>(cuda_data,
                                             target_width,
                                             rw, rh,
                                             ecoef, n);
      cudaDeviceSynchronize();
      // fprintf(stderr, "%s\n", cudaGetErrorName(cudaGetLastError())); // DEBUG
      cudaMemcpy(out_data, cuda_data, n, cudaMemcpyDeviceToHost);
      cudaFree(cuda_data);
      free(host_mat);
      cudaUnbindTexture(tex_ref);
      cudaFreeArray(cuArray);
      break;
    }
    case OFFSET_CUBEMAP_32:
    case OFFSET_CUBEMAP:
    {
      cudaArray* cuArray;
      cudaMallocArray(&cuArray, &tex_ref.channelDesc, width, height);
      cudaMemcpyToArray(cuArray, 0, 0, data, width * height, cudaMemcpyHostToDevice);
      cudaBindTextureToArray(tex_ref, cuArray);
      tex_ref.addressMode[0] = cudaAddressModeWrap;
      tex_ref.addressMode[1] = cudaAddressModeWrap;
      tex_ref.filterMode = cudaFilterModePoint; // "cudaFilterModeLinear" ONLY USED WITH FLOAT TEXTURE
      tex_ref.normalized = true;
      float* host_mat = (float*)malloc(PARAM_NUM * sizeof(float));
      RotationMatrix(Radians(yaw), Radians(pitch), Radians(-roll), host_mat);
      cudaMemcpyToSymbol(mat, host_mat, PARAM_NUM * sizeof(float));
      int n = target_width * target_height;
      float rw = 1.0 / target_width, rh = 1.0 / target_height;
      uint8_t* cuda_data = NULL;
      cudaMalloc((void**)&cuda_data, n);
      int nt = min(max(target_width / 3, target_height >> 1), THREAD_LIMIT);
      CUDATransformKernelOCMP<<<n / nt, nt>>>(cuda_data,
                                              target_width,
                                              rw, rh, ecoef,
                                              x, y, z, n);
      cudaDeviceSynchronize();
      // fprintf(stderr, "%s\n", cudaGetErrorName(cudaGetLastError())); // DEBUG
      cudaMemcpy(out_data, cuda_data, n, cudaMemcpyDeviceToHost);
      cudaFree(cuda_data);
      free(host_mat);
      cudaUnbindTexture(tex_ref);
      cudaFreeArray(cuArray);
      break;
    }
    case EAC_32:
    case EAC:
    {
      cudaArray* cuArray;
      cudaMallocArray(&cuArray, &tex_ref.channelDesc, width, height);
      cudaMemcpyToArray(cuArray, 0, 0, data, width * height, cudaMemcpyHostToDevice);
      cudaBindTextureToArray(tex_ref, cuArray);
      tex_ref.addressMode[0] = cudaAddressModeWrap;
      tex_ref.addressMode[1] = cudaAddressModeWrap;
      tex_ref.filterMode = cudaFilterModePoint; // "cudaFilterModeLinear" ONLY USED WITH FLOAT TEXTURE
      tex_ref.normalized = true;
      float* host_mat = (float*)malloc(PARAM_NUM * sizeof(float));
      RotationMatrix(Radians(yaw), Radians(pitch), Radians(-roll), host_mat);
      cudaMemcpyToSymbol(mat, host_mat, PARAM_NUM * sizeof(float));
      int n = target_width * target_height;
      float rw = 1.0 / target_width, rh = 1.0 / target_height;
      uint8_t* cuda_data = NULL;
      cudaMalloc((void**)&cuda_data, n);
      int nt = min(max(target_width / 3, target_height >> 1), THREAD_LIMIT);
      CUDATransformKernelEAC<<<n / nt, nt>>>(cuda_data,
                                             target_width,
                                             rw, rh,
                                             ecoef, n);
      cudaDeviceSynchronize();
      // fprintf(stderr, "%s\n", cudaGetErrorName(cudaGetLastError())); // DEBUG
      cudaMemcpy(out_data, cuda_data, n, cudaMemcpyDeviceToHost);
      cudaFree(cuda_data);
      free(host_mat);
      cudaUnbindTexture(tex_ref);
      cudaFreeArray(cuArray);
      break;
    }
    case OFFSET_EAC_32:
    case OFFSET_EAC:
    {
      cudaArray* cuArray;
      cudaMallocArray(&cuArray, &tex_ref.channelDesc, width, height);
      cudaMemcpyToArray(cuArray, 0, 0, data, width * height, cudaMemcpyHostToDevice);
      cudaBindTextureToArray(tex_ref, cuArray);
      tex_ref.addressMode[0] = cudaAddressModeWrap;
      tex_ref.addressMode[1] = cudaAddressModeWrap;
      tex_ref.filterMode = cudaFilterModePoint; // "cudaFilterModeLinear" ONLY USED WITH FLOAT TEXTURE
      tex_ref.normalized = true;
      float* host_mat = (float*)malloc(PARAM_NUM * sizeof(float));
      RotationMatrix(Radians(yaw), Radians(pitch), Radians(-roll), host_mat);
      cudaMemcpyToSymbol(mat, host_mat, PARAM_NUM * sizeof(float));
      int n = target_width * target_height;
      float rw = 1.0 / target_width, rh = 1.0 / target_height;
      uint8_t* cuda_data = NULL;
      cudaMalloc((void**)&cuda_data, n);
      int nt = min(max(target_width / 3, target_height >> 1), THREAD_LIMIT);
      CUDATransformKernelOEAC<<<n / nt, nt>>>(cuda_data,
                                              target_width,
                                              rw, rh, ecoef,
                                              x, y, z, n);
      cudaDeviceSynchronize();
      // fprintf(stderr, "%s\n", cudaGetErrorName(cudaGetLastError())); // DEBUG
      cudaMemcpy(out_data, cuda_data, n, cudaMemcpyDeviceToHost);
      cudaFree(cuda_data);
      free(host_mat);
      cudaUnbindTexture(tex_ref);
      cudaFreeArray(cuArray);
      break;
    }
    case BASEBALL_EQUIRECT_32:
    case BASEBALL_EQUIRECT:
    {
      cudaArray* cuArray;
      cudaMallocArray(&cuArray, &tex_ref.channelDesc, width, height);
      cudaMemcpyToArray(cuArray, 0, 0, data, width * height, cudaMemcpyHostToDevice);
      cudaBindTextureToArray(tex_ref, cuArray);
      tex_ref.addressMode[0] = cudaAddressModeWrap;
      tex_ref.addressMode[1] = cudaAddressModeWrap;
      tex_ref.filterMode = cudaFilterModePoint; // "cudaFilterModeLinear" ONLY USED WITH FLOAT TEXTURE
      tex_ref.normalized = true;
      float* host_mat = (float*)malloc(2 * PARAM_NUM * sizeof(float));
      RotationMatrix(Radians(yaw), Radians(pitch), Radians(-roll), host_mat);
      RotationMatrix(Radians(180.0 + yaw), Radians(-pitch), Radians(-90.0 + roll), PARAM_NUM + host_mat);
      cudaMemcpyToSymbol(mat, host_mat, 2 * PARAM_NUM * sizeof(float));
      int nh = target_width * target_height >> 1;
      float rw = 1.0 / target_width, rhh = 2.0 / target_height;
      uint8_t* cuda_data = NULL;
      cudaMalloc((void**)&cuda_data, nh << 1);
      int nt = min(max(target_width / 3, target_height >> 1), THREAD_LIMIT);
      CUDATransformKernelBEP<<<nh / nt, nt>>>(cuda_data,
                                              target_width,
                                              rw, rhh, ecoef,
                                              0, nh);
      CUDATransformKernelBEP<<<nh / nt, nt>>>(nh + cuda_data,
                                              target_width,
                                              rw, rhh, ecoef,
                                              PARAM_NUM, nh);
      cudaDeviceSynchronize();
      // fprintf(stderr, "%s\n", cudaGetErrorName(cudaGetLastError())); // DEBUG
      cudaMemcpy(out_data, cuda_data, nh << 1, cudaMemcpyDeviceToHost);
      cudaFree(cuda_data);
      free(host_mat);
      cudaUnbindTexture(tex_ref);
      cudaFreeArray(cuArray);
      break;
    }
    case SHIFT_BASEBALL_EQUIRECT_21:
    {
      cudaArray* cuArray;
      cudaMallocArray(&cuArray, &tex_ref.channelDesc, width, height);
      cudaMemcpyToArray(cuArray, 0, 0, data, width * height, cudaMemcpyHostToDevice);
      cudaBindTextureToArray(tex_ref, cuArray);
      tex_ref.addressMode[0] = cudaAddressModeWrap;
      tex_ref.addressMode[1] = cudaAddressModeWrap;
      tex_ref.filterMode = cudaFilterModePoint; // "cudaFilterModeLinear" ONLY USED WITH FLOAT TEXTURE
      tex_ref.normalized = true;
      float* host_mat = (float*)malloc(2 * PARAM_NUM * sizeof(float));
      RotationMatrix(Radians(yaw), Radians(pitch), Radians(-roll), host_mat);
      RotationMatrix(Radians(180.0 + yaw), Radians(-pitch), Radians(-90.0 + roll), PARAM_NUM + host_mat);
      cudaMemcpyToSymbol(mat, host_mat, 2 * PARAM_NUM * sizeof(float));
      int target_height_up = target_width / 3;
      int nu = target_width * target_height_up;
      int target_height_dn = target_height - target_height_up;
      int nd = target_width * target_height_dn;
      float rw = 1.0 / target_width, rhu = 1.0 / target_height_up, rhd = 1.0 / target_height_dn;
      uint8_t* cuda_data = NULL;
      cudaMalloc((void**)&cuda_data, nu + nd);
      int nt = min(max(target_width / 3, target_height_up), THREAD_LIMIT);
      CUDATransformKernelBEP<<<nu / nt, nt>>>(cuda_data,
                                              target_width,
                                              rw, rhu, ecoef,
                                              0, nu);
      CUDATransformKernelBEP<<<nd / nt, nt>>>(nu + cuda_data,
                                              target_width,
                                              rw, rhd, ecoef,
                                              PARAM_NUM, nd);
      cudaDeviceSynchronize();
      // fprintf(stderr, "%s\n", cudaGetErrorName(cudaGetLastError())); // DEBUG
      cudaMemcpy(out_data, cuda_data, nu + nd, cudaMemcpyDeviceToHost);
      cudaFree(cuda_data);
      free(host_mat);
      cudaUnbindTexture(tex_ref);
      cudaFreeArray(cuArray);
      break;
    }
    default:
      fprintf(stderr, "CUDATransformWrapper(): Invalid Flag\n");
      return 1;
  }
  return 0;
}

int CUDARenderWrapper(const uint8_t* data,
                      int width,
                      int height,
                      int target_width,
                      int target_height,
                      float yaw,
                      float pitch,
                      float roll,
                      float x,
                      float y,
                      float z,
                      float ecoef,
                      uint8_t* out_data,
                      int flag) {
  return 0;
}
