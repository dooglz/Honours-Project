#pragma once

#define DASH50 "--------------------------------------------------"
static const void clearOut() {
  // cout << string(100, '\n');
}

static const char Spinner(uint8_t t) {
  char spinners[] = {
      '|', '/', '-', '\\',
  };
  return (spinners[t % 4]);
}

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