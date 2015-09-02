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

static const char Loading(uint8_t t, uint8_t c) {
  char spinners[] = {'◐', '◓', '◑',  '◒', '|', '/', '-', '\\', '.', 'o',
                     'O', '0', '\'', '-', '.', '-', '◢', '◣',  '◤', '◥'};
  return (spinners[(c % 4) * 4 + (t % 4)]);
}
