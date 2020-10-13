#include <iostream>
#include <time.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // cv::imwrite()
#include <opencv2/imgproc.hpp> // cv::cvtColor()

#include "transformer.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
  if (argc != 14) {
    cerr << "Invalid Argument Number" << endl;
    exit(1);
  }
  Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if (image.data == NULL) {
    cerr << "Loading Image Failed: " << argv[1] << endl;
    exit(1);
  }
  int i420_y_size = image.rows * image.cols;
  int i420_uv_size = i420_y_size >> 2;
  Mat i420_image;
  cvtColor(image, i420_image, CV_BGR2YUV_I420);
  uint8_t* y = i420_image.data;
  uint8_t* u = y + i420_y_size;
  uint8_t* v = u + i420_uv_size;
  int target_width = atof(argv[2]),
      target_height = atof(argv[3]);
  float yaw = atof(argv[4]),
        pitch = atof(argv[5]),
        roll = atof(argv[6]);
  float dx = atof(argv[7]),
        dy = atof(argv[8]),
        dz = atof(argv[9]);
  float ecoef = atof(argv[10]);
  int flag = atoi(argv[11]);
  if (ft360::Consult(target_width, target_height, flag) != 0) {
    cerr << "ft360::Consult() Failed" << endl;
    exit(1);
  }
  cout << "Consulted Resolution: "
       << target_width << "x" << target_height << endl;
  ft360::Transformer transformer(y, u, v,
                                 image.cols,
                                 image.rows,
                                 target_width,
                                 target_height);
  clock_t t_start, t_end;
  t_start = clock();
  if (transformer.Transform(yaw, pitch, roll,
                            dx, dy, dz,
                            ecoef, flag) != 0) {
    cerr << "Transformer::Transform() Failed" << endl;
    exit(1);
  }
  t_end = clock();
  cout << "#1 Running Time: "
       << 1000 * (float)(t_end - t_start) / CLOCKS_PER_SEC
       << "(ms)" << endl;
  // WHEN NEED TO DEBUG
  t_start = clock();
  for (int i = 0; i < 30; i++) {
    if (transformer.Transform(yaw, pitch, roll,
                              dx, dy, dz,
                              ecoef, flag) != 0) {
      cerr << "Transformer::Transform() Failed" << endl;
      exit(1);
    }
  }
  t_end = clock();
  cout << "Average Running Time(1s): "
       << 1000 * (float)(t_end - t_start) / (CLOCKS_PER_SEC * 30)
       << "(ms)" << endl;
  t_start = clock();
  for (int i = 0; i < 300; i++) {
    if (transformer.Transform(yaw, pitch, roll,
                              dx, dy, dz,
                              ecoef, flag) != 0) {
      cerr << "Transformer::Transform() Failed" << endl;
      exit(1);
    }
  }
  t_end = clock();
  cout << "Average Running Time(10s): "
       << 1000 * (float)(t_end - t_start) / (CLOCKS_PER_SEC * 300)
       << "(ms)" << endl;
  t_start = clock();
  for (int i = 0; i < 1800; i++) {
    if (transformer.Transform(yaw, pitch, roll,
                              dx, dy, dz,
                              ecoef, flag) != 0) {
      cerr << "Transformer::Transform() Failed" << endl;
      exit(1);
    }
  }
  t_end = clock();
  cout << "Average Running Time(60s): "
       << 1000 * (float)(t_end - t_start) / (CLOCKS_PER_SEC * 1800)
       << "(ms)" << endl;
  // WHEN NEED TO DEBUG */
  if (transformer.Save(argv[12], atoi(argv[13])) != 0) {
    cerr << "Transformer::Save() Failed" << endl;
    exit(1);
  }
  exit(0);
}
