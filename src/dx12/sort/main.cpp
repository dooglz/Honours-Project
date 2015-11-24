#include "stdafx.h"

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
//#define NSIZE 4194304
#define NSIZEBYTES NSIZE * 4
#define CARDS 2
#define NSIZEPC NSIZE / CARDS
#define NSIZEBYTESPC NSIZEBYTES / CARDS
#define USEMAPPEDUNIFORM true

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



void ExecuteAndWait(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12CommandQueue> cmdQueue,
                    ComPtr<ID3D12CommandAllocator> cmdAlloc, ComPtr<ID3D12PipelineState> pso, ComPtr<ID3D12Device> device) {
  // Execute
  ThrowIfFailed(cmdList->Close());
  ID3D12CommandList *cmds = cmdList.Get();
  cmdQueue->ExecuteCommandLists(1, &cmds);

  ComPtr<ID3D12Fence> m_fence;
  ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));

  // Tell the queue to set the Fence to 1
  ThrowIfFailed(cmdQueue->Signal(m_fence.Get(), 1));

  // Wait until the fence gets set
  if (m_fence->GetCompletedValue() < 1) {
    HANDLE m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ThrowIfFailed(m_fence->SetEventOnCompletion(1, m_fenceEvent));
    WaitForSingleObject(m_fenceEvent, INFINITE);
    CloseHandle(m_fenceEvent);
  }

  // Cleanup command
  ThrowIfFailed(cmdAlloc->Reset());
  ThrowIfFailed(cmdList->Reset(cmdAlloc.Get(), pso.Get()));
}
struct constantBufStuff {
  UINT8 *bufferData;
  ComPtr<ID3D12Resource> buffer;
  ComPtr<ID3D12Resource> bufferUpload;
};
constantBufStuff createConstBuf(ComPtr<ID3D12Device> m_device) {
  const UINT bufferSize = sizeof(ConstantBufferCS);
  constantBufStuff cbf;
#if USEMAPPEDUNIFORM
  ThrowIfFailed(m_device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
      IID_PPV_ARGS(&cbf.buffer)));

  CD3DX12_RANGE readRange(0, 0); // We do not intend to read from this resource on the CPU.
  ThrowIfFailed(cbf.buffer->Map(0, &readRange, reinterpret_cast<void **>(&cbf.bufferData)));
  ZeroMemory(cbf.bufferData, bufferSize);

#else
  ThrowIfFailed(m_device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
      nullptr, IID_PPV_ARGS(&cbf.buffer)));

  ThrowIfFailed(m_device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
      IID_PPV_ARGS(&cbf.bufferUpload)));
#endif
return cbf;
}

void setConstBuf(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12Device> m_device, constantBufStuff cbf, ConstantBufferCS cb) {

#if USEMAPPEDUNIFORM
  memcpy(cbf.bufferData, &cb, sizeof(ConstantBufferCS));
#else
  // ttransition back to a writable state
  setResourceBarrier(cmdList.Get(), cbf.buffer.Get(),
                     D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                     D3D12_RESOURCE_STATE_COPY_DEST);

  D3D12_SUBRESOURCE_DATA computeCBData = {};
  computeCBData.pData = reinterpret_cast<UINT8 *>(&cb);
  computeCBData.RowPitch = CBSIZE;
  computeCBData.SlicePitch = computeCBData.RowPitch;

  ThrowIfFailed(UpdateSubresources<1>(cmdList.Get(), cbf.buffer.Get(),
    cbf.bufferUpload.Get(), 0, 0, 1, &computeCBData));
  setResourceBarrier(cmdList.Get(), cbf.buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                     D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
#endif
}

void Sort(int size, ComPtr<ID3D12GraphicsCommandList> cmdList,
          ComPtr<ID3D12RootSignature> rootSignature, ComPtr<ID3D12PipelineState> pso,
          ComPtr<ID3D12DescriptorHeap> descHeapUav, ComPtr<ID3D12Device> m_device,
          ComPtr<ID3D12CommandQueue> cmdQueue, ComPtr<ID3D12CommandAllocator> cmdAlloc, ComPtr<ID3D12Device> device,constantBufStuff cbf) {
  cmdList->SetComputeRootSignature(rootSignature.Get());
  cmdList->SetPipelineState(pso.Get());
  cmdList->SetDescriptorHeaps(1, descHeapUav.GetAddressOf());
  cmdList->SetComputeRootConstantBufferView(1, cbf.buffer->GetGPUVirtualAddress());
  cmdList->SetComputeRootDescriptorTable(0, descHeapUav->GetGPUDescriptorHandleForHeapStart());

  UINT j, k, ret, dx, dy, dz;

  dx = size;
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
  //  Major step
  for (k = 2; k <= size; k <<= 1) {
    //  Minor step
    for (j = k >> 1; j > 0; j = j >> 1) {
      setConstBuf(cmdList, m_device,cbf, {j, k, dy});
      ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso,device);

      cmdList->SetComputeRootSignature(rootSignature.Get());
      cmdList->SetPipelineState(pso.Get());
      cmdList->SetDescriptorHeaps(1, descHeapUav.GetAddressOf());
      cmdList->SetComputeRootConstantBufferView(1, cbf.buffer->GetGPUVirtualAddress());
      cmdList->SetComputeRootDescriptorTable(0, descHeapUav->GetGPUDescriptorHandleForHeapStart());
      cmdList->Dispatch(dx, dy, 1);
      ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, device);
    }
  }

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
  ComPtr<ID3D12CommandAllocator> cmdAlloc[CARDS];
  ComPtr<ID3D12CommandQueue> cmdQueue[CARDS];
  ComPtr<ID3D12GraphicsCommandList> cmdList[CARDS];
  ComPtr<ID3D12DescriptorHeap> descHeapUav[CARDS];
  ComPtr<ID3D12RootSignature> rootSignature[CARDS];
  ComPtr<ID3D12PipelineState> pso[CARDS];
  ComPtr<ID3D12Resource> bufferDefault[CARDS];
  ComPtr<ID3D12Resource> bufferReadback[CARDS];
  ComPtr<ID3D12Resource> bufferUpload[CARDS];
  ComPtr<ID3D12Device> devices[CARDS];
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


  // Create shader
  ID3D10Blob *cs;
  {
    ID3D10Blob *info;
    UINT flag = 0;
#if _DEBUG
    flag |= D3DCOMPILE_DEBUG;
#endif /* _DEBUG */
    ThrowIfFailed(D3DCompileFromFile(L"sort.hlsl", nullptr, nullptr, "CSMain", "cs_5_0", flag, 0,
      &cs, &info));
  }

  for (size_t i = 0; i < CARDS; ++i) {
    ThrowIfFailed(
        D3D12CreateDevice(vAdapters[0], D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&devices[i])));
    ThrowIfFailed(devices[i]->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(cmdAlloc[i].ReleaseAndGetAddressOf())));

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ThrowIfFailed(devices[i]->CreateCommandQueue(
        &queueDesc, IID_PPV_ARGS(cmdQueue[i].ReleaseAndGetAddressOf())));

    ThrowIfFailed(devices[i]->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                cmdAlloc[i].Get(), nullptr,
                                                IID_PPV_ARGS(cmdList[i].ReleaseAndGetAddressOf())));
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
      ThrowIfFailed(
          D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &info));
      devices[i]->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                      IID_PPV_ARGS(rootSignature[i].ReleaseAndGetAddressOf()));
      sig->Release();
    }
    // Create PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.CS.BytecodeLength = cs->GetBufferSize();
    psoDesc.CS.pShaderBytecode = cs->GetBufferPointer();
    psoDesc.pRootSignature = rootSignature[i].Get();
    ThrowIfFailed(
      devices[i]->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso[i].ReleaseAndGetAddressOf())));

    // Create DescriptorHeap for UAV
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 10;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(
      devices[i]->CreateDescriptorHeap(&desc, IID_PPV_ARGS(descHeapUav[i].ReleaseAndGetAddressOf())));

    // Create buffer on device memory
    auto resourceDesc =
      CD3DX12_RESOURCE_DESC::Buffer(NSIZEBYTES, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ThrowIfFailed(devices[i]->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &resourceDesc,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
      IID_PPV_ARGS(bufferDefault[i].ReleaseAndGetAddressOf())));
    bufferDefault[i]->SetName(L"BufferDefault");

    // Create buffer on system memory
    resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(NSIZEBYTES);
    ThrowIfFailed(devices[i]->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE, &resourceDesc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
      IID_PPV_ARGS(bufferReadback[i].ReleaseAndGetAddressOf())));
    bufferReadback[i]->SetName(L"BufferReadback");

    ThrowIfFailed(devices[i]->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(NSIZEBYTES), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
      IID_PPV_ARGS(&bufferUpload[i])));
    bufferUpload[i]->SetName(L"BufferUpload");

    // Setup UAV
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.Buffer.NumElements = NSIZE;
    uavDesc.Buffer.StructureByteStride = 4;
    devices[i]->CreateUnorderedAccessView(bufferDefault[i].Get(), nullptr, &uavDesc,
      descHeapUav[i]->GetCPUDescriptorHandleForHeapStart());

    createConstBuf(devices[i]);
    ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, devices[i]);

  }

  cs->Release();

  for (size_t i = 0; i < CARDS; ++i) {
    SendRandom(cmdList, bufferUpload, bufferDefault);
    ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, devices[i]);
  }
  cout << "Numbers copied " << endl;

  for (size_t i = 0; i < CARDS; ++i) {
    auto start = chrono::high_resolution_clock::now();

    Sort(NSIZE, cmdList, rootSignature, pso, descHeapUav, devices[i], cmdQueue, cmdAlloc);
    ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso);
    auto end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count() << "ns" << endl;
    cout << "Numbers Sorted" << endl;
  }
  setResourceBarrier(cmdList.Get(), bufferDefault.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COPY_SOURCE);
  ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso);

  cmdList->CopyResource(bufferReadback.Get(), bufferDefault.Get());

  // Execute
  ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso);
  cout << "Numbers Copied to readback" << endl;

  // Get system memory pointer
  void *data;
  ThrowIfFailed(bufferReadback->Map(0, nullptr, &data));

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

  CheckArrayOrder(rndData, NSIZE, true);
  cout << "Validation done" << endl;

  delete[] rndData;
}

int wmain(int argc, wchar_t **argv) {
  proc();
  _getwch();

  return 0;
}