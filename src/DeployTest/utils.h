#pragma once
#include <string>
#include <iostream>
#include <sstream>
#include <functional>

#define DASH50 "--------------------------------------------------"
#define CLEARN "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"

using namespace std;

static const char Spinner(const unsigned int t) {
  char spinners[] = {
      '|', '/', '-', '\\',
  };
  return (spinners[t % 4]);
}

const std::string readable_fs(const unsigned int sz /*in bytes*/);

const std::string current_time_and_date();

template <class T> bool lexical_cast(T &result, const std::string &str) {
  std::stringstream s(str);
  return (s >> result && s.rdbuf()->in_avail() == 0);
}

template <class T, class U>
T promptValidated(const std::string &message,
                  std::function<bool(U)> condition = [](...) { return true; }) {
  T input;
  std::string buf;
  while (!(std::cout << message,
           std::getline(std::cin, buf) && lexical_cast<T>(input, buf) && condition(input))) {
    if (std::cin.eof())
      throw std::runtime_error("End of file reached!");
  }
  return input;
}
template <class T, class U>
T nopromptValidated(std::function<bool(U)> condition = [](...) { return true; }) {
  T input;
  std::string buf;
  while (!(std::getline(std::cin, buf) && lexical_cast<T>(input, buf) && condition(input))) {
    if (std::cin.eof())
      throw std::runtime_error("End of file reached!");
  }
  return input;
}