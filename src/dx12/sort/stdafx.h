//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently.

#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers.
#endif

#include <windows.h>

#include <dxgi1_4.h>
#include <d3d12.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>

#include "d3dx12.h"

#include <wrl.h>
#include <vector>
#include <shellapi.h>

#include <string>
#include <iostream>

inline void ThrowIfFailed(HRESULT hr)
{
  if (FAILED(hr))
  {
    throw;
  }
}

template <typename T>
const bool CheckArrayOrder(const T *a, const size_t size, const bool order) {
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