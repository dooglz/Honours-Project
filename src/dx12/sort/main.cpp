#include "stdafx.h"

using namespace std;
using Microsoft::WRL::ComPtr;
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

struct constantBufStuff {
  UINT8 *bufferData;
  ComPtr<ID3D12Resource> buffer;
  ComPtr<ID3D12Resource> bufferUpload;
};

void Execute(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12CommandQueue> cmdQueue) {
  // Execute
  ThrowIfFailed(cmdList->Close());
  ID3D12CommandList *cmds = cmdList.Get();
  cmdQueue->ExecuteCommandLists(1, &cmds);
}
void Wait(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12CommandQueue> cmdQueue,
          ComPtr<ID3D12CommandAllocator> cmdAlloc, ComPtr<ID3D12PipelineState> pso,
          ComPtr<ID3D12Device> device) {
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

void ExecuteAndWait(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12CommandQueue> cmdQueue,
                    ComPtr<ID3D12CommandAllocator> cmdAlloc, ComPtr<ID3D12PipelineState> pso,
                    ComPtr<ID3D12Device> device) {
  Execute(cmdList, cmdQueue);
  Wait(cmdList, cmdQueue, cmdAlloc, pso, device);
}

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

void setConstBuf(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12Device> m_device,
                 constantBufStuff cbf, ConstantBufferCS cb) {

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

  ThrowIfFailed(UpdateSubresources<1>(cmdList.Get(), cbf.buffer.Get(), cbf.bufferUpload.Get(), 0, 0,
                                      1, &computeCBData));
  setResourceBarrier(cmdList.Get(), cbf.buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                     D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
#endif
}

void Sort(int size, ComPtr<ID3D12GraphicsCommandList> cmdList,
          ComPtr<ID3D12RootSignature> rootSignature, ComPtr<ID3D12PipelineState> pso,
          ComPtr<ID3D12DescriptorHeap> descHeapUav, ComPtr<ID3D12Device> device,
          ComPtr<ID3D12CommandQueue> cmdQueue, ComPtr<ID3D12CommandAllocator> cmdAlloc,
          constantBufStuff cbf) {
  cmdList->SetComputeRootSignature(rootSignature.Get());
  cmdList->SetPipelineState(pso.Get());
  cmdList->SetDescriptorHeaps(1, descHeapUav.GetAddressOf());
  cmdList->SetComputeRootConstantBufferView(1, cbf.buffer->GetGPUVirtualAddress());
  cmdList->SetComputeRootDescriptorTable(0, descHeapUav->GetGPUDescriptorHandleForHeapStart());

  UINT dx = size;
  UINT dy = 1;
  UINT dz = 1;
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
  for (UINT k = 2; k <= size; k <<= 1) {
    //  Minor step
    for (UINT j = k >> 1; j > 0; j = j >> 1) {
      setConstBuf(cmdList, device, cbf, {j, k, dy});
      ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, device);

      cmdList->SetComputeRootSignature(rootSignature.Get());
      cmdList->SetPipelineState(pso.Get());
      cmdList->SetDescriptorHeaps(1, descHeapUav.GetAddressOf());
      cmdList->SetComputeRootConstantBufferView(1, cbf.buffer->GetGPUVirtualAddress());
      cmdList->SetComputeRootDescriptorTable(0, descHeapUav->GetGPUDescriptorHandleForHeapStart());
      cmdList->Dispatch(dx, dy, 1);
      ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, device);
    }
  }
}

void SortN(int size, const int GPU_N, ComPtr<ID3D12GraphicsCommandList> *cmdList,
           ComPtr<ID3D12RootSignature> *rootSignature, ComPtr<ID3D12PipelineState> *pso,
           ComPtr<ID3D12DescriptorHeap> *descHeapUav, ComPtr<ID3D12Device> *device,
           ComPtr<ID3D12CommandQueue> *cmdQueue, ComPtr<ID3D12CommandAllocator> *cmdAlloc,
           constantBufStuff *cbf) {

  for (size_t i = 0; i < GPU_N; ++i) {
    cmdList[i]->SetComputeRootSignature(rootSignature[i].Get());
    cmdList[i]->SetPipelineState(pso[i].Get());
    cmdList[i]->SetDescriptorHeaps(1, descHeapUav[i].GetAddressOf());
    cmdList[i]->SetComputeRootConstantBufferView(1, cbf[i].buffer->GetGPUVirtualAddress());
    cmdList[i]->SetComputeRootDescriptorTable(0,
                                              descHeapUav[i]->GetGPUDescriptorHandleForHeapStart());
  }

  UINT dx = size;
  UINT dy = 1;
  UINT dz = 1;
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
  for (UINT k = 2; k <= size; k <<= 1) {
    //  Minor step
    for (UINT j = k >> 1; j > 0; j = j >> 1) {
      for (size_t i = 0; i < GPU_N; ++i) {
        setConstBuf(cmdList[i], device[i], cbf[i], {j, k, dy});
        ExecuteAndWait(cmdList[i], cmdQueue[i], cmdAlloc[i], pso[i], device[i]);

        cmdList[i]->SetComputeRootSignature(rootSignature[i].Get());
        cmdList[i]->SetPipelineState(pso[i].Get());
        cmdList[i]->SetDescriptorHeaps(1, descHeapUav[i].GetAddressOf());
        cmdList[i]->SetComputeRootConstantBufferView(1, cbf[i].buffer->GetGPUVirtualAddress());
        cmdList[i]->SetComputeRootDescriptorTable(
            0, descHeapUav[i]->GetGPUDescriptorHandleForHeapStart());
        cmdList[i]->Dispatch(dx, dy, 1);
        ExecuteAndWait(cmdList[i], cmdQueue[i], cmdAlloc[i], pso[i], device[i]);
      }
    }
  }
}

void proc() {

  const int GPU_N = 2;
  UINT maxN = 1 << 12;
  UINT maxNPC = (UINT)floor(maxN / GPU_N);
  size_t sz = maxN * sizeof(UINT);
  size_t szPC = maxNPC * sizeof(UINT);
  UINT *rndData = new UINT[maxN];
  for (UINT i = 0; i < maxN; i++) {
    UINT x = 0;
    rndData[i] = (x << 14) | ((UINT)rand() & 0x3FFF);
    rndData[i] = (x << 14) | ((UINT)rand() & 0x3FFF);
  }

  ComPtr<ID3D12CommandAllocator> cmdAlloc[GPU_N];
  ComPtr<ID3D12CommandQueue> cmdQueue[GPU_N];
  ComPtr<ID3D12GraphicsCommandList> cmdList[GPU_N];
  ComPtr<ID3D12DescriptorHeap> descHeapUav[GPU_N];
  ComPtr<ID3D12RootSignature> rootSignature[GPU_N];
  ComPtr<ID3D12PipelineState> pso[GPU_N];
  ComPtr<ID3D12Resource> bufferDefault[GPU_N];
  ComPtr<ID3D12Resource> bufferReadback[GPU_N];
  ComPtr<ID3D12Resource> bufferUpload[GPU_N];
  ComPtr<ID3D12Device> devices[GPU_N];
  constantBufStuff cbs[GPU_N];
  ComPtr<ID3D12Resource> m_crossAdapterResource[GPU_N];
  ComPtr<ID3D12Heap> shared_heap[GPU_N];

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

  for (size_t i = 0; i < GPU_N; ++i) {
    ThrowIfFailed(
        D3D12CreateDevice(vAdapters[i], D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&devices[i])));
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
    ThrowIfFailed(devices[i]->CreateComputePipelineState(
        &psoDesc, IID_PPV_ARGS(pso[i].ReleaseAndGetAddressOf())));

    // Create DescriptorHeap for UAV
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 10;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(devices[i]->CreateDescriptorHeap(
        &desc, IID_PPV_ARGS(descHeapUav[i].ReleaseAndGetAddressOf())));

    // Create buffer on device memory
    auto resourceDesc =
        CD3DX12_RESOURCE_DESC::Buffer(szPC, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ThrowIfFailed(devices[i]->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &resourceDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(bufferDefault[i].ReleaseAndGetAddressOf())));
    bufferDefault[i]->SetName(L"BufferDefault");

    // Create buffer on system memory
    resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(szPC);
    ThrowIfFailed(devices[i]->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE, &resourceDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(bufferReadback[i].ReleaseAndGetAddressOf())));
    bufferReadback[i]->SetName(L"BufferReadback");

    ThrowIfFailed(devices[i]->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(szPC), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&bufferUpload[i])));
    bufferUpload[i]->SetName(L"BufferUpload");

    // Setup UAV
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.Buffer.NumElements = maxNPC;
    uavDesc.Buffer.StructureByteStride = 4;
    devices[i]->CreateUnorderedAccessView(bufferDefault[i].Get(), nullptr, &uavDesc,
                                          descHeapUav[i]->GetCPUDescriptorHandleForHeapStart());

    cbs[i] = createConstBuf(devices[i]);
    ExecuteAndWait(cmdList[i], cmdQueue[i], cmdAlloc[i], pso[i], devices[i]);
  }

  cs->Release();

  if (GPU_N > 1) {
    // Create cross-adapter shared resources on the primary adapter,
    //  and open the shared handles on  the secondary adapter.
    D3D12_RESOURCE_DESC crossAdapterDesc = CD3DX12_RESOURCE_DESC::Buffer(szPC);
    crossAdapterDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;

    CD3DX12_HEAP_DESC heapDesc(szPC, D3D12_HEAP_TYPE_DEFAULT, 0,
                               D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER);

    ThrowIfFailed(devices[0]->CreateHeap(&heapDesc, IID_PPV_ARGS(&shared_heap[0])));

    HANDLE heapHandle = nullptr;
    auto a = devices[0]->CreateSharedHandle(shared_heap[0].Get(), nullptr, GENERIC_ALL, nullptr,
                                            &heapHandle);

    HRESULT openSharedHandleResult =
        devices[1]->OpenSharedHandle(heapHandle, IID_PPV_ARGS(&shared_heap[1]));

    // We can close the handle after opening the cross-adapter shared resource.
    CloseHandle(heapHandle);

    ThrowIfFailed(openSharedHandleResult);

    ThrowIfFailed(devices[0]->CreatePlacedResource(shared_heap[0].Get(), 0, &crossAdapterDesc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&m_crossAdapterResource[0])));
    ThrowIfFailed(devices[1]->CreatePlacedResource(shared_heap[1].Get(), 0, &crossAdapterDesc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&m_crossAdapterResource[1])));
    ExecuteAndWait(cmdList[0], cmdQueue[0], cmdAlloc[0], pso[0], devices[0]);
    ExecuteAndWait(cmdList[1], cmdQueue[1], cmdAlloc[1], pso[1], devices[1]);
  }

  // upload random data
  for (size_t i = 0; i < GPU_N; ++i) {
    D3D12_SUBRESOURCE_DATA rnd = {};
    rnd.pData = (&rndData[i * maxNPC]);
    rnd.RowPitch = szPC;
    rnd.SlicePitch = rnd.RowPitch;

    setResourceBarrier(cmdList[i].Get(), bufferDefault[i].Get(),
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    ThrowIfFailed(UpdateSubresources<1>(cmdList[i].Get(), bufferDefault[i].Get(),
                                        bufferUpload[i].Get(), 0, 0, 1, &rnd));
    setResourceBarrier(cmdList[i].Get(), bufferDefault[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Execute(cmdList[i], cmdQueue[i]);
  }

  for (size_t i = 0; i < GPU_N; ++i) {
    Wait(cmdList[i], cmdQueue[i], cmdAlloc[i], pso[i], devices[i]);
  }

  if (GPU_N > 1) {
    for (size_t swapsize = maxNPC / 2; swapsize > 0; swapsize /= 2) {
      SortN(maxNPC, GPU_N, cmdList, rootSignature, pso, descHeapUav, devices, cmdQueue, cmdAlloc,
            cbs);
      const size_t swapByteSize = swapsize * sizeof(UINT);

      // now swap
      // read in the bottom of card 1 to bottom of swapspace
      setResourceBarrier(cmdList[1].Get(), bufferDefault[1].Get(),
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);

      cmdList[1]->CopyBufferRegion(m_crossAdapterResource[1].Get(), 0, bufferDefault[1].Get(), 0,
                                   swapByteSize);

      setResourceBarrier(cmdList[1].Get(), bufferDefault[1].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                         D3D12_RESOURCE_STATE_COPY_DEST);

      ExecuteAndWait(cmdList[1], cmdQueue[1], cmdAlloc[1], pso[1], devices[1]);

      // read in the top of card 0 to top of swapspace
      setResourceBarrier(cmdList[0].Get(), bufferDefault[0].Get(),
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);

      cmdList[0]->CopyBufferRegion(m_crossAdapterResource[0].Get(), szPC - swapByteSize,
                                   bufferDefault[0].Get(), szPC - swapByteSize, swapByteSize);

      setResourceBarrier(cmdList[0].Get(), bufferDefault[0].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                         D3D12_RESOURCE_STATE_COPY_DEST);

      // execute
      ExecuteAndWait(cmdList[0], cmdQueue[0], cmdAlloc[0], pso[0], devices[0]);

      // setup swapspace for reading
      setResourceBarrier(cmdList[0].Get(), m_crossAdapterResource[0].Get(),
                         D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
      setResourceBarrier(cmdList[1].Get(), m_crossAdapterResource[1].Get(),
                         D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);

      // read in the bottom of swapspace to top of card 0
      cmdList[0]->CopyBufferRegion(bufferDefault[0].Get(), szPC - swapByteSize,
                                   m_crossAdapterResource[0].Get(), 0, swapByteSize);
      // read in the top of swapspace to bottom of card 1
      cmdList[1]->CopyBufferRegion(bufferDefault[1].Get(), 0, m_crossAdapterResource[1].Get(),
                                   szPC - swapByteSize, swapByteSize);

      // setup swapspace for writing
      setResourceBarrier(cmdList[0].Get(), m_crossAdapterResource[0].Get(),
                         D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
      setResourceBarrier(cmdList[1].Get(), m_crossAdapterResource[1].Get(),
                         D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);

      ExecuteAndWait(cmdList[0], cmdQueue[0], cmdAlloc[0], pso[0], devices[0]);
      ExecuteAndWait(cmdList[1], cmdQueue[1], cmdAlloc[1], pso[1], devices[1]);
    }
  }

  cout << "Numbers Sorted" << endl;
  for (size_t i = 0; i < GPU_N; ++i) {
    setResourceBarrier(cmdList[i].Get(), bufferDefault[i].Get(),
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    ExecuteAndWait(cmdList[i], cmdQueue[i], cmdAlloc[i], pso[i], devices[i]);

    cmdList[i]->CopyResource(bufferReadback[i].Get(), bufferDefault[i].Get());

    // Execute
    ExecuteAndWait(cmdList[i], cmdQueue[i], cmdAlloc[i], pso[i], devices[i]);
  }
  cout << "Numbers Copied to readback" << endl;

  // Get system memory pointer
  UINT *outData = new UINT[maxN];
  for (size_t i = 0; i < GPU_N; ++i) {
    void *data;
    ThrowIfFailed(bufferReadback[i]->Map(0, nullptr, &data));
    UINT *dpointer = (UINT *)data;
    for (size_t j = 0; j < maxNPC; j++) {
      outData[(i * maxNPC) + j] = *dpointer;
      dpointer++;
    }
    bufferReadback[i]->Unmap(0, nullptr);
    cout << "readback done" << endl;
  }

  CheckArrayOrder(rndData, maxN, true);
  delete[] rndData;
  cout << "Validation done" << endl;
}

int wmain(int argc, wchar_t **argv) {
  proc();
  _getwch();

  return 0;
}