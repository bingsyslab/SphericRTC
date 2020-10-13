#include "transformer.h"

namespace ft360 {

int Consult(int& target_width, int& target_height, int flag) {
  int bandwidth = target_width * target_height;
  switch (flag) {
    case EQUIRECT_21:
    case OFFSET_EQUIRECT_21:
    {
      int unit = (int)round(0.125 * sqrt(0.5 * (float)bandwidth)) << 3;
      target_width = unit << 1;
      target_height = unit;
      break;
    }
    case CUBEMAP_32:
    case OFFSET_CUBEMAP_32:
    case EAC_32:
    case OFFSET_EAC_32:
    case BASEBALL_EQUIRECT_32:
    {
      int unit = (int)round(0.125 * sqrt(0.16666666666666666 * (float)bandwidth)) << 3;
      target_width = 3 * unit;
      target_height = unit << 1;
      break;
    }
    case SHIFT_BASEBALL_EQUIRECT_21:
    {
      int unit = (int)round(0.125 * sqrt(0.05555555555555555 * (float)bandwidth)) << 3;
      target_width = 6 * unit;
      target_height = target_width >> 1;
      break;
    }
    case EQUIRECT:
    case OFFSET_EQUIRECT:
    case CUBEMAP:
    case OFFSET_CUBEMAP:
    case EAC:
    case OFFSET_EAC:
    case BASEBALL_EQUIRECT:
      target_width = (int)round(0.125 * (float)target_width) << 3;
      target_height = (int)(0.125 * (float)(bandwidth / target_width)) << 3;
      break;
    default:
      fprintf(stderr, "Consult(): Invalid Flag\n");
      return 1;
  }
  return 0;
}

Transformer::Transformer(const uint8_t* y,
                         const uint8_t* u,
                         const uint8_t* v,
                         int width,
                         int height,
                         int target_width,
                         int target_height) {
  if (SetChannel(y, 'y',
                 width,
                 height,
                 target_width,
                 target_height,
                 y_) != 0)
    fprintf(stderr, "SetChannel() Failed: y_\n");
  if (SetChannel(u, 'u',
                 width >> 1,
                 height >> 1,
                 target_width >> 1,
                 target_height >> 1,
                 u_) != 0)
    fprintf(stderr, "SetChannel() Failed: u_\n");
  if (SetChannel(v, 'v',
                 width >> 1,
                 height >> 1,
                 target_width >> 1,
                 target_height >> 1,
                 v_) != 0)
    fprintf(stderr, "SetChannel() Failed: v_\n");
}

Transformer::~Transformer() {
  y_.data = NULL;
  free(y_.transformed_data);
  u_.data = NULL;
  free(u_.transformed_data);
  v_.data = NULL;
  free(v_.transformed_data);
}

int Transformer::Scale() {
  if (CUDAScaleWrapper(y_.data,
                       y_.width,
                       y_.height,
                       y_.target_width,
                       y_.target_height,
                       y_.transformed_data) != 0) {
    fprintf(stderr, "CUDAScaleWrapper() Failed: y_\n");
    return 1;
  }
  if (CUDAScaleWrapper(u_.data,
                       u_.width,
                       u_.height,
                       u_.target_width,
                       u_.target_height,
                       u_.transformed_data) != 0) {
    fprintf(stderr, "CUDAScaleWrapper() Failed: u_\n");
    return 1;
  }
  if (CUDAScaleWrapper(v_.data,
                       v_.width,
                       v_.height,
                       v_.target_width,
                       v_.target_height,
                       v_.transformed_data) != 0) {
    fprintf(stderr, "CUDAScaleWrapper() Failed: v_\n");
    return 1;
  }
  return 0;
}

int Transformer::Rotate(float yaw, float pitch, float roll) {
  if (CUDARotateWrapper(y_.data,
                        y_.width,
                        y_.height,
                        y_.target_width,
                        y_.target_height,
                        yaw,
                        pitch,
                        roll,
                        y_.transformed_data) != 0) {
    fprintf(stderr, "CUDARotateWrapper() Failed: y_\n");
    return 1;
  }
  if (CUDARotateWrapper(u_.data,
                        u_.width,
                        u_.height,
                        u_.target_width,
                        u_.target_height,
                        yaw,
                        pitch,
                        roll,
                        u_.transformed_data) != 0) {
    fprintf(stderr, "CUDARotateWrapper() Failed: u_\n");
    return 1;
  }
  if (CUDARotateWrapper(v_.data,
                        v_.width,
                        v_.height,
                        v_.target_width,
                        v_.target_height,
                        yaw,
                        pitch,
                        roll,
                        v_.transformed_data) != 0) {
    fprintf(stderr, "CUDARotateWrapper() Failed: v_\n");
    return 1;
  }
  return 0;
}

int Transformer::Transform(float yaw,
                           float pitch,
                           float roll,
                           float x,
                           float y,
                           float z,
                           float ecoef,
                           int flag) {
  switch (flag) {
    case OFFSET_EQUIRECT_21:
    case OFFSET_EQUIRECT:
    case CUBEMAP_32:
    case CUBEMAP:
    case OFFSET_CUBEMAP_32:
    case OFFSET_CUBEMAP:
    case EAC_32:
    case EAC:
    case OFFSET_EAC_32:
    case OFFSET_EAC:
    case BASEBALL_EQUIRECT_32:
    case BASEBALL_EQUIRECT:
    case SHIFT_BASEBALL_EQUIRECT_21:
      if (CUDATransformWrapper(y_.data,
                               y_.width,
                               y_.height,
                               y_.target_width,
                               y_.target_height,
                               yaw,
                               pitch,
                               roll,
                               x, y, z,
                               ecoef,
                               y_.transformed_data,
                               flag) != 0) {
        fprintf(stderr, "CUDATransformWrapper() Failed: y_\n");
        return 1;
      }
      if (CUDATransformWrapper(u_.data,
                               u_.width,
                               u_.height,
                               u_.target_width,
                               u_.target_height,
                               yaw,
                               pitch,
                               roll,
                               x, y, z,
                               ecoef,
                               u_.transformed_data,
                               flag) != 0) {
        fprintf(stderr, "CUDATransformWrapper() Failed: u_\n");
        return 1;
      }
      if (CUDATransformWrapper(v_.data,
                               v_.width,
                               v_.height,
                               v_.target_width,
                               v_.target_height,
                               yaw,
                               pitch,
                               roll,
                               x, y, z,
                               ecoef,
                               v_.transformed_data,
                               flag) != 0) {
        fprintf(stderr, "CUDATransformWrapper() Failed: v_\n");
        return 1;
      }
      break;
    default:
      fprintf(stderr, "Transform(): Invalid Flag\n");
      return 1;
  }
  return 0;
}

int Transformer::Render(float yaw,
                        float pitch,
                        float roll,
                        float x,
                        float y,
                        float z,
                        float ecoef,
                        int flag) {
  switch (flag) {
    case EQUIRECT_21:
    case EQUIRECT:
    case OFFSET_EQUIRECT_21:
    case OFFSET_EQUIRECT:
    case CUBEMAP_32:
    case CUBEMAP:
    case OFFSET_CUBEMAP_32:
    case OFFSET_CUBEMAP:
    case EAC_32:
    case EAC:
    case OFFSET_EAC_32:
    case OFFSET_EAC:
    case BASEBALL_EQUIRECT_32:
    case BASEBALL_EQUIRECT:
    case SHIFT_BASEBALL_EQUIRECT_21:
      if (CUDARenderWrapper(y_.data,
                            y_.width,
                            y_.height,
                            y_.target_width,
                            y_.target_height,
                            yaw,
                            pitch,
                            roll,
                            x, y, z,
                            ecoef,
                            y_.transformed_data,
                            flag) != 0) {
        fprintf(stderr, "CUDARenderWrapper() Failed: y_\n");
        return 1;
      }
      if (CUDARenderWrapper(u_.data,
                            u_.width,
                            u_.height,
                            u_.target_width,
                            u_.target_height,
                            yaw,
                            pitch,
                            roll,
                            x, y, z,
                            ecoef,
                            u_.transformed_data,
                            flag) != 0) {
        fprintf(stderr, "CUDARenderWrapper() Failed: u_\n");
        return 1;
      }
      if (CUDARenderWrapper(v_.data,
                            v_.width,
                            v_.height,
                            v_.target_width,
                            v_.target_height,
                            yaw,
                            pitch,
                            roll,
                            x, y, z,
                            ecoef,
                            v_.transformed_data,
                            flag) != 0) {
        fprintf(stderr, "CUDARenderWrapper() Failed: v_\n");
        return 1;
      }
      break;
    default:
      fprintf(stderr, "Render(): Invalid Flag\n");
      return 1;
  }
  return 0;
}

int Transformer::Save(const char* file_name, int flag) {
  /* NOT COMPILED WITH CHROME
  cv::Mat yuv, rgb;
  switch (flag) {
    case IN:
    {
      yuv.create(y_.height + (y_.height >> 1), y_.width, CV_8UC1);
      int stride = y_.width * y_.height;
      memcpy(yuv.data, y_.data, stride);
      memcpy(yuv.data + stride, u_.data, stride >> 2);
      memcpy(yuv.data + stride + (stride >> 2), v_.data, stride >> 2);
      rgb.create(y_.height, y_.width, CV_8UC3);
      cv::cvtColor(yuv, rgb, CV_YUV2BGR_I420, 3);
      break;
    }
    case OUT:
    {
      yuv.create(y_.target_height + (y_.target_height >> 1), y_.target_width, CV_8UC1);
      int stride = y_.target_width * y_.target_height;
      memcpy(yuv.data, y_.transformed_data, stride);
      memcpy(yuv.data + stride, u_.transformed_data, stride >> 2);
      memcpy(yuv.data + stride + (stride >> 2), v_.transformed_data, stride >> 2);
      rgb.create(y_.target_height, y_.target_width, CV_8UC3);
      cv::cvtColor(yuv, rgb, CV_YUV2BGR_I420, 3);
      break;
    }
    default:
      fprintf(stderr, "Save(): Invalid Flag\n");
      return 1;
  }
  if (!cv::imwrite(file_name, rgb)) {
    fprintf(stderr, "Save(): cv::imwrite() Failed\n");
    return 1;
  } // NOT COMPILED WITH CHROME */
  return 0;
}

void Transformer::Get(uint8_t* transformed_y,
                      uint8_t* transformed_u,
                      uint8_t* transformed_v) {
  memcpy(transformed_y, y_.transformed_data, y_.target_width * y_.target_height);
  memcpy(transformed_u, u_.transformed_data, u_.target_width * u_.target_height);
  memcpy(transformed_v, v_.transformed_data, v_.target_width * v_.target_height);
}

int Transformer::SetChannel(const uint8_t* data,
                            char id,
                            int width,
                            int height,
                            int target_width,
                            int target_height,
                            Channel& channel) {
  channel.data = data;
  channel.width = width;
  channel.height = height;
  channel.target_width = target_width;
  channel.target_height = target_height;
  channel.transformed_data = (uint8_t*)malloc(target_width * target_height);
  if (channel.transformed_data == NULL) {
    fprintf(stderr, "SetChannel(): Memory Allocation Failed: %c_\n", id);
    return 1;
  }
  return 0;
}

} // namespace ft360
