#pragma once
#include <string>

#define DASH50 "--------------------------------------------------"
#define CLEARN "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"

static const char Spinner(const unsigned int t) {
  char spinners[] = {
      '|', '/', '-', '\\',
  };
  return (spinners[t % 4]);
}

const std::string readable_fs(const unsigned int sz /*in bytes*/);

const std::string current_time_and_date();