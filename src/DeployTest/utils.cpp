#include "utils.h"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <math.h>

const std::string readable_fs(const unsigned int sz /*in bytes*/) {
  float size = (float)sz;
  unsigned int kb = 1024;
  unsigned int mb = kb * 1024;
  unsigned int gb = mb * 1024;
  std::string s = "";
  float minus = 0;
  if (size > gb) {
    float a = floor(size / gb);
    minus += a * gb;
    s += std::to_string((int)a);
    s += "GB, ";
  }
  if (size > mb) {
    float a = floor((size - minus) / mb);
    minus += a * mb;
    s += std::to_string((int)a);
    s += "MB, ";
  }
  if (size > kb) {
    float a = floor((size - minus) / kb);
    minus += a * kb;
    s += std::to_string((int)a);
    s += "KB, ";
  }
  s += std::to_string((int)(size - minus));
  s += "B (";
  s += std::to_string(sz);
  s += ")";
  return s;
}

const std::string current_time_and_date() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  // std::stringstream ss;
  // ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
  // return ss.str();
  char mbstr[100];
  time_t rawtime;
  time(&rawtime);
  struct tm *timeinfo = localtime(&rawtime);
  strftime(mbstr, 100, "%Y-%m-%d_%H-%M-%S", timeinfo);
  return std::string(mbstr);
}