#pragma once
#include <string>


#define DASH50 "--------------------------------------------------"
static const void clearOut() {
  // cout << string(100, '\n');
}

static const char Spinner(const unsigned int t) {
  char spinners[] = {
      '|', '/', '-', '\\',
  };
  return (spinners[t % 4]);
}

const std::string readable_fs(const unsigned int sz /*in bytes*/) ;