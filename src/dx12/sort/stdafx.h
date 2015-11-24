#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <string>
#include <iostream>
#include <chrono>
#include <vector>

#include <windows.h>

#include <dxgi1_4.h>
#include <d3d12.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include "d3dx12.h"

// Windows Runtime Library.
#include <wrl.h>

//#include <shellapi.h>

inline void ThrowIfFailed(HRESULT hr) {
  if (FAILED(hr)) {
    throw;
  }
}

template <typename T> const bool CheckArrayOrder(const T *a, const size_t size, const bool order) {
  if (size < 1) {
    return true;
  }
  for (size_t i = 1; i < size; i++) {
    if ((order && a[i] < a[i - 1]) || (!order && a[i] > a[i - 1])) {
      return false;
    }
  }
  return true;
}