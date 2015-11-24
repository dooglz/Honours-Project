#include "stdafx.h"

//#include <LM.h>
//#include <tchar.h>
//#include <wrl/client.h>
//#include <clocale>
//#include <stdexcept>
#include <chrono>

using namespace std;
using Microsoft::WRL::ComPtr;
//#define NSIZE 8
//#define NSIZE 32
//#define NSIZE 64
//#define NSIZE 128
//#define NSIZE 256
//#define NSIZE 1024
//#define NSIZE 2048
//#define NSIZE 16384
//#define NSIZE 32768
#define NSIZE 65536
//#define NSIZE 131072
//#define NSIZE 1048576
#define NSIZEBYTES NSIZE * 4
#define CARDS 2
#define NSIZEPC NSIZE / CARDS
#define NSIZEBYTESPC NSIZEBYTES / CARDS
#define USEMAPPEDUNIFORM true

namespace {
ComPtr<ID3D12Device> gDev;
}

void CHK(HRESULT hr) {
  if (FAILED(hr))
    throw runtime_error("HRESULT is failed value.");
}

static const char Spinner(const unsigned int t) {
  char spinners[] = {
      '|', '/', '-', '\\',
  };
  return (spinners[t % 4]);
}

void setResourceBarrier(ID3D12GraphicsCommandList *commandList, ID3D12Resource *res,
                        D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
  D3D12_RESOURCE_BARRIER desc = {};
  desc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  desc.Transition.pResource = res;
  desc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  desc.Transition.StateBefore = before;
  desc.Transition.StateAfter = after;
  desc.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  commandList->ResourceBarrier(1, &desc);
}

struct ConstantBufferCS {
  UINT j;
  UINT k;
  UINT yDimSize;
};
const UINT CBSIZE = sizeof(ConstantBufferCS);

ComPtr<ID3D12Resource> m_constantBufferCS;
ComPtr<ID3D12Resource> constantBufferCSUpload;

UINT m_fenceValue = 1;
ComPtr<ID3D12Fence> m_fence;

void ExecuteAndWait(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12CommandQueue> cmdQueue,
                    ComPtr<ID3D12CommandAllocator> cmdAlloc, ComPtr<ID3D12PipelineState> pso,
                    ComPtr<ID3D12Fence> fence) {
  // Execute
  ThrowIfFailed(cmdList->Close());
  ID3D12CommandList *cmds = cmdList.Get();
  cmdQueue->ExecuteCommandLists(1, &cmds);

  // Tell teh queue to set the Fence to m_fenceValue
  ThrowIfFailed(cmdQueue->Signal(m_fence.Get(), m_fenceValue));

  // Wait until the previous frame is finished.
  if (m_fence->GetCompletedValue() < m_fenceValue) {
    HANDLE m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent));
    WaitForSingleObject(m_fenceEvent, INFINITE);
    CloseHandle(m_fenceEvent);
  }
  m_fenceValue++;

  /*
   HANDLE fenceEveneHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
   // Wait until GPU finished
   CHK(fence->SetEventOnCompletion(1, fenceEveneHandle));
   CHK(cmdQueue->Signal(fence.Get(), 1));
   auto wait = WaitForSingleObject(fenceEveneHandle, 10000);
   if (wait != WAIT_OBJECT_0)
     throw runtime_error("Failed WaitForSingleObject().");
 */

  // Cleanup command
  ThrowIfFailed(cmdAlloc->Reset());
  ThrowIfFailed(cmdList->Reset(cmdAlloc.Get(), pso.Get()));
}

UINT8 *m_pConstantBufferGSData;
void createConstBuf(ComPtr<ID3D12Device> m_device) {
  const UINT bufferSize = sizeof(ConstantBufferCS);

#if USEMAPPEDUNIFORM
  ThrowIfFailed(m_device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
      IID_PPV_ARGS(&m_constantBufferCS)));

  CD3DX12_RANGE readRange(0, 0); // We do not intend to read from this resource on the CPU.
  ThrowIfFailed(
      m_constantBufferCS->Map(0, &readRange, reinterpret_cast<void **>(&m_pConstantBufferGSData)));
  ZeroMemory(m_pConstantBufferGSData, bufferSize);

#else
  ThrowIfFailed(m_device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
      nullptr, IID_PPV_ARGS(&m_constantBufferCS)));

  ThrowIfFailed(m_device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
      IID_PPV_ARGS(&constantBufferCSUpload)));
#endif
}

void setConstBuf(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12Device> m_device,
                 ConstantBufferCS cb) {

#if USEMAPPEDUNIFORM
  UINT8 *destination = m_pConstantBufferGSData;
  memcpy(destination, &cb, sizeof(ConstantBufferCS));

#else
  // ttransition back to a writable state
  setResourceBarrier(cmdList.Get(), m_constantBufferCS.Get(),
                     D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                     D3D12_RESOURCE_STATE_COPY_DEST);

  D3D12_SUBRESOURCE_DATA computeCBData = {};
  computeCBData.pData = reinterpret_cast<UINT8 *>(&cb);
  computeCBData.RowPitch = CBSIZE;
  computeCBData.SlicePitch = computeCBData.RowPitch;

  ThrowIfFailed(UpdateSubresources<1>(cmdList.Get(), m_constantBufferCS.Get(),
                                      constantBufferCSUpload.Get(), 0, 0, 1, &computeCBData));
  setResourceBarrier(cmdList.Get(), m_constantBufferCS.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                     D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
#endif
}

void Sort(int size, ComPtr<ID3D12GraphicsCommandList> cmdList,
          ComPtr<ID3D12RootSignature> rootSignature, ComPtr<ID3D12PipelineState> pso,
          ComPtr<ID3D12DescriptorHeap> descHeapUav, ComPtr<ID3D12Device> m_device,
          ComPtr<ID3D12CommandQueue> cmdQueue, ComPtr<ID3D12CommandAllocator> cmdAlloc,
          ComPtr<ID3D12Fence> fence) {
  cmdList->SetComputeRootSignature(rootSignature.Get());
  cmdList->SetPipelineState(pso.Get());
  cmdList->SetDescriptorHeaps(1, descHeapUav.GetAddressOf());
  cmdList->SetComputeRootConstantBufferView(1, m_constantBufferCS->GetGPUVirtualAddress());
  cmdList->SetComputeRootDescriptorTable(0, descHeapUav->GetGPUDescriptorHandleForHeapStart());

  UINT j, k, ret, dx, dy, dz;

  dx = size / 2;
  dy = 1;
  dz = 1;
  if (dx > 65535) {
    for (dy = 2; dy < 65535; dy += 2) {
      if ((dx / dy) < 35535) {
        dx = (dx / dy);
        break;
      }
    }
    if (dx > 65535) {
      cout << "too many numbers" << endl;
      return;
    }
    cout << dx << "," << dy << endl;
  }

  int phase;
  bool lookDirection;
  for (int phase = 0; phase < (size); ++phase) {
    lookDirection = (phase % 2 == 0);

    setConstBuf(cmdList, m_device, {lookDirection, 0, dy});

    ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);
    cmdList->SetComputeRootSignature(rootSignature.Get());
    cmdList->SetComputeRootConstantBufferView(1, m_constantBufferCS->GetGPUVirtualAddress());

    cmdList->SetComputeRootSignature(rootSignature.Get());
    cmdList->SetPipelineState(pso.Get());
    cmdList->SetDescriptorHeaps(1, descHeapUav.GetAddressOf());
    cmdList->SetComputeRootConstantBufferView(1, m_constantBufferCS->GetGPUVirtualAddress());
    cmdList->SetComputeRootDescriptorTable(0, descHeapUav->GetGPUDescriptorHandleForHeapStart());

    cmdList->Dispatch(dx, dy, 1);
    ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);
    if (phase % 512 == 0) {
      static int aa = 0;
      cout << "\r" << Spinner(aa++) << "\tPercent Done:\t" << int((float)phase * 100 / (float)size)
           << "% " << std::flush;
    }
  }

  /*
    //  Major step
    for (k = 2; k <= size; k <<= 1) {
      //  Minor step
      for (j = k >> 1; j > 0; j = j >> 1) {
        setConstBuf(cmdList, m_device, {j, k, dy});

        ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);
        cmdList->SetComputeRootSignature(rootSignature.Get());
        cmdList->SetComputeRootConstantBufferView(1, m_constantBufferCS->GetGPUVirtualAddress());

        cmdList->SetComputeRootSignature(rootSignature.Get());
        cmdList->SetPipelineState(pso.Get());
        cmdList->SetDescriptorHeaps(1, descHeapUav.GetAddressOf());
        cmdList->SetComputeRootConstantBufferView(1, m_constantBufferCS->GetGPUVirtualAddress());
        cmdList->SetComputeRootDescriptorTable(0,
    descHeapUav->GetGPUDescriptorHandleForHeapStart());

        cmdList->Dispatch(dx, dy, 1);
        ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);
      }
      cout << k << endl;
    }
    */
  cout << "Sort done" << endl;
}

void SendRandom(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12Resource> bufferUpload,
                ComPtr<ID3D12Resource> bufferTarget) {

  UINT *rndData = new UINT[NSIZE];
  cout << endl;
  for (size_t i = 0; i < NSIZE; i++) {
    UINT x = (UINT)0;
    rndData[i] = (x << 14) | ((UINT)rand() & 0x3FFF);
    rndData[i] = (x << 14) | ((UINT)rand() & 0x3FFF);
    if (i < 12) {
      cout << rndData[i] << ",";
    }
  }
  cout << endl;
  D3D12_SUBRESOURCE_DATA rnd = {};
  rnd.pData = (rndData);
  rnd.RowPitch = NSIZEBYTES;
  rnd.SlicePitch = rnd.RowPitch;

  setResourceBarrier(cmdList.Get(), bufferTarget.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COPY_DEST);
  ThrowIfFailed(
      UpdateSubresources<1>(cmdList.Get(), bufferTarget.Get(), bufferUpload.Get(), 0, 0, 1, &rnd));
  setResourceBarrier(cmdList.Get(), bufferTarget.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  delete[] rndData;
}

void proc() {
  ComPtr<ID3D12CommandAllocator> cmdAlloc;
  ComPtr<ID3D12CommandQueue> cmdQueue;
  ComPtr<ID3D12GraphicsCommandList> cmdList;
  ComPtr<ID3D12Fence> fence;
  HANDLE fenceEveneHandle;
  ComPtr<ID3D12DescriptorHeap> descHeapUav;
  ComPtr<ID3D12RootSignature> rootSignature;
  ComPtr<ID3D12PipelineState> pso;
  ComPtr<ID3D12Resource> bufferDefault;
  ComPtr<ID3D12Resource> bufferReadback;
  ComPtr<ID3D12Resource> bufferUpload;

#if _DEBUG
  ID3D12Debug *debug = nullptr;
  D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
  if (debug) {
    debug->EnableDebugLayer();
    debug->Release();
    debug = nullptr;
  }
#endif /* _DEBUG */

  std::vector<IDXGIAdapter *> vAdapters;
  {

    UINT i = 0;
    ComPtr<IDXGIFactory4> pFactory;
    ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&pFactory)));
    IDXGIAdapter *pAdapter;

    while (pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
      vAdapters.push_back(pAdapter);
      DXGI_ADAPTER_DESC adapterDesc;
      pAdapter->GetDesc(&adapterDesc);
      wcout << "Adapter " << i << "\t" << adapterDesc.Description << endl;
      ++i;
    }
  }

  ID3D12Device *dev;
  CHK(D3D12CreateDevice(vAdapters[0], D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)));
  gDev = dev;

  CHK(gDev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                   IID_PPV_ARGS(cmdAlloc.ReleaseAndGetAddressOf())));

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  CHK(gDev->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(cmdQueue.ReleaseAndGetAddressOf())));

  CHK(gDev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAlloc.Get(), nullptr,
                              IID_PPV_ARGS(cmdList.ReleaseAndGetAddressOf())));

  CHK(gDev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.ReleaseAndGetAddressOf())));

  fenceEveneHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);

  // Create root signature
  {
    CD3DX12_DESCRIPTOR_RANGE descRange[1];
    descRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);

    CD3DX12_ROOT_PARAMETER rootParam[2];
    rootParam[0].InitAsDescriptorTable(ARRAYSIZE(descRange), descRange);
    rootParam[1].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);

    ID3D10Blob *sig, *info;
    auto rootSigDesc = D3D12_ROOT_SIGNATURE_DESC();
    rootSigDesc.NumParameters = 2;
    rootSigDesc.NumStaticSamplers = 0;
    rootSigDesc.pParameters = rootParam;
    rootSigDesc.pStaticSamplers = nullptr;
    rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    CHK(D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &info));
    gDev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                              IID_PPV_ARGS(rootSignature.ReleaseAndGetAddressOf()));
    sig->Release();
  }

  // Create shader
  ID3D10Blob *cs;
  {
    ID3D10Blob *info;
    UINT flag = 0;
#if _DEBUG
    flag |= D3DCOMPILE_DEBUG;
#endif /* _DEBUG */
    CHK(D3DCompileFromFile(L"sort.hlsl", nullptr, nullptr, "CSMain", "cs_5_0", flag, 0, &cs,
                           &info));
  }

  ThrowIfFailed(gDev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));

  // Create PSO
  D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
  psoDesc.CS.BytecodeLength = cs->GetBufferSize();
  psoDesc.CS.pShaderBytecode = cs->GetBufferPointer();
  psoDesc.pRootSignature = rootSignature.Get();
  CHK(gDev->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso.ReleaseAndGetAddressOf())));
  cs->Release();

  // Create DescriptorHeap for UAV
  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  desc.NumDescriptors = 10;
  desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  CHK(gDev->CreateDescriptorHeap(&desc, IID_PPV_ARGS(descHeapUav.ReleaseAndGetAddressOf())));

  // Create buffer on device memory
  auto resourceDesc =
      CD3DX12_RESOURCE_DESC::Buffer(NSIZEBYTES, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  CHK(gDev->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                                    D3D12_HEAP_FLAG_NONE, &resourceDesc,
                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                    IID_PPV_ARGS(bufferDefault.ReleaseAndGetAddressOf())));
  bufferDefault->SetName(L"BufferDefault");

  // Create buffer on system memory
  resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(NSIZEBYTES);
  CHK(gDev->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
                                    D3D12_HEAP_FLAG_NONE, &resourceDesc,
                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                    IID_PPV_ARGS(bufferReadback.ReleaseAndGetAddressOf())));
  bufferReadback->SetName(L"BufferReadback");

  ThrowIfFailed(gDev->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(NSIZEBYTES), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
      IID_PPV_ARGS(&bufferUpload)));
  bufferUpload->SetName(L"BufferUpload");

  // Setup UAV
  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
  uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  uavDesc.Format = DXGI_FORMAT_UNKNOWN;
  uavDesc.Buffer.NumElements = NSIZE;
  uavDesc.Buffer.StructureByteStride = 4;
  gDev->CreateUnorderedAccessView(bufferDefault.Get(), nullptr, &uavDesc,
                                  descHeapUav->GetCPUDescriptorHandleForHeapStart());

  createConstBuf(gDev);
  ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);

  SendRandom(cmdList, bufferUpload, bufferDefault);
  ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);
  cout << "Numbers copied " << endl;

  auto start = chrono::high_resolution_clock::now();

  Sort(NSIZE, cmdList, rootSignature, pso, descHeapUav, gDev, cmdQueue, cmdAlloc, fence);
  ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);
  auto end = chrono::high_resolution_clock::now();
  cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count() << "ns" << endl;
  cout << "Numbers Sorted" << endl;

  setResourceBarrier(cmdList.Get(), bufferDefault.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COPY_SOURCE);
  ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);

  cmdList->CopyResource(bufferReadback.Get(), bufferDefault.Get());

  // Execute
  ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);
  cout << "Numbers Copied to readback" << endl;

  // Get system memory pointer
  void *data;
  CHK(bufferReadback->Map(0, nullptr, &data));

  UINT *rndData = new UINT[NSIZE];
  int *dpointer = (int *)data;
  for (size_t i = 0; i < NSIZE; i++) {
    rndData[i] = *dpointer;
    dpointer++;
  }
  bufferReadback->Unmap(0, nullptr);
  cout << "readback done" << endl;

  for (size_t i = 0; i < NSIZE && i < 12; i++) {
    cout << rndData[i] << ",";
  }
  cout << endl;
  delete[] rndData;

  // Output
}

int wmain(int argc, wchar_t **argv) {
  proc();
//  _getwch();

  return 0;
}