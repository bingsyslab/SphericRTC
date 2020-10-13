#ifndef FAST_TRANSFORM_360_H_
#define FAST_TRANSFORM_360_H_

#include <stdint.h>
#include <stddef.h> // NULL
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include <cuda_runtime.h> // CUDA RUNTIME APIS
/* NOT COMPILED WITH CHROME
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // cv::imwrite()
#include <opencv2/imgproc.hpp> // cv::cvtColor()
// NOT COMPILED WITH CHROME */

#define EQUIRECT_21 0
#define EQUIRECT 1
#define OFFSET_EQUIRECT_21 2
#define OFFSET_EQUIRECT 3
#define CUBEMAP_32 4
#define CUBEMAP 5
#define OFFSET_CUBEMAP_32 6
#define OFFSET_CUBEMAP 7
#define EAC_32 8
#define EAC 9
#define OFFSET_EAC_32 10
#define OFFSET_EAC 11
#define BASEBALL_EQUIRECT_32 12
#define BASEBALL_EQUIRECT 13
#define SHIFT_BASEBALL_EQUIRECT_21 14

#define IN 0
#define OUT 1

extern int CUDAScaleWrapper(const uint8_t* data,
                            int width,
                            int height,
                            int target_width,
                            int target_height,
                            uint8_t* out_data);

extern int CUDARotateWrapper(const uint8_t* data,
                             int width,
                             int height,
                             int target_width,
                             int target_height,
                             float yaw,
                             float pitch,
                             float roll,
                             uint8_t* out_data);

extern int CUDATransformWrapper(const uint8_t* data,
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
                                int flag);

extern int CUDARenderWrapper(const uint8_t* data,
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
                             int flag);

namespace ft360 {

int Consult(int& target_width, int& target_height, int flag);

struct Channel {
  const uint8_t* data;
  int width;
  int height;
  int target_width;
  int target_height;
  uint8_t* transformed_data;
};

class Transformer {
 public:
  Transformer(const uint8_t* y,
              const uint8_t* u,
              const uint8_t* v,
              int width,
              int height,
              int target_width,
              int target_height);
  ~Transformer();
  int Scale();
  int Rotate(float yaw, float pitch, float roll);
  int Transform(float yaw,
                float pitch,
                float roll,
                float x,
                float y,
                float z,
                float ecoef,
                int flag);
  int Render(float yaw,
             float pitch,
             float roll,
             float x,
             float y,
             float z,
             float ecoef,
             int flag);
  int Save(const char* file_name, int flag); // NOT COMPILED WITH CHROME
  void Get(uint8_t* transformed_y,
           uint8_t* transformed_u,
           uint8_t* transformed_v);

 private:
  int SetChannel(const uint8_t* data,
                 char id,
                 int width,
                 int height,
                 int target_width,
                 int target_height,
                 Channel& channel);
  Channel y_;
  Channel u_;
  Channel v_;
};

} // namespace ft360

#endif
