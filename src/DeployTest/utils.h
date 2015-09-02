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