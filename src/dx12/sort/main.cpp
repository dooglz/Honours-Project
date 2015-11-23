#include "stdafx.h"

//#include <LM.h>
//#include <tchar.h>
//#include <wrl/client.h>
//#include <clocale>
//#include <stdexcept>

using namespace std;
using Microsoft::WRL::ComPtr;
#define NSIZE 8
#define NSIZEBYTES NSIZE * 4
namespace {
ComPtr<ID3D12Device> gDev;
}

void CHK(HRESULT hr) {
  if (FAILED(hr))
    throw runtime_error("HRESULT is failed value.");
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
};
const UINT CBSIZE = sizeof(ConstantBufferCS);

ComPtr<ID3D12Resource> m_constantBufferCS;
ComPtr<ID3D12Resource> constantBufferCSUpload;

void ExecuteAndWait(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12CommandQueue> cmdQueue,
                    ComPtr<ID3D12CommandAllocator> cmdAlloc, ComPtr<ID3D12PipelineState> pso,
                    ComPtr<ID3D12Fence> fence) {
  // Execute
  CHK(cmdList->Close());
  ID3D12CommandList *cmds = cmdList.Get();
  cmdQueue->ExecuteCommandLists(1, &cmds);

  ;
  HANDLE fenceEveneHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  // Wait until GPU finished
  CHK(fence->SetEventOnCompletion(1, fenceEveneHandle));
  CHK(cmdQueue->Signal(fence.Get(), 1));
  auto wait = WaitForSingleObject(fenceEveneHandle, 10000);
  if (wait != WAIT_OBJECT_0)
    throw runtime_error("Failed WaitForSingleObject().");

  // Cleanup command
  ThrowIfFailed(cmdAlloc->Reset());
  ThrowIfFailed(cmdList->Reset(cmdAlloc.Get(), pso.Get()));
  CloseHandle(fenceEveneHandle);
}

void createConstBuf(ComPtr<ID3D12Device> m_device) {
  const UINT bufferSize = sizeof(ConstantBufferCS);

  ThrowIfFailed(m_device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
      nullptr, IID_PPV_ARGS(&m_constantBufferCS)));

  ThrowIfFailed(m_device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
      IID_PPV_ARGS(&constantBufferCSUpload)));
}

void setConstBuf(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12Device> m_device,
                 ConstantBufferCS cb) {
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

  UINT j, k, ret;
  //  Major step
  for (k = 2; k <= size; k <<= 1) {
    //  Minor step
    for (j = k >> 1; j > 0; j = j >> 1) {
      setConstBuf(cmdList, m_device, {j, k});

      ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);
      cmdList->SetComputeRootSignature(rootSignature.Get());
      cmdList->SetComputeRootConstantBufferView(1, m_constantBufferCS->GetGPUVirtualAddress());
      cmdList->Dispatch(size, 1, 1);
      // ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);
    }
  }
}

void SendRandom(ComPtr<ID3D12GraphicsCommandList> cmdList, ComPtr<ID3D12Resource> bufferUpload,
                ComPtr<ID3D12Resource> bufferTarget) {

  UINT rndData[NSIZE];
  cout << endl;
  for (size_t i = 0; i < NSIZE; i++) {
    UINT x = (UINT)0;
    rndData[i] = (x << 14) | ((UINT)rand() & 0x3FFF);
    rndData[i] = (x << 14) | ((UINT)rand() & 0x3FFF);
    cout << rndData[i] << ",";
  }
  cout << endl;
  D3D12_SUBRESOURCE_DATA rnd = {};
  rnd.pData = (&rndData);
  rnd.RowPitch = NSIZEBYTES;
  rnd.SlicePitch = rnd.RowPitch;

  setResourceBarrier(cmdList.Get(), bufferTarget.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COPY_DEST);
  ThrowIfFailed(
      UpdateSubresources<1>(cmdList.Get(), bufferTarget.Get(), bufferUpload.Get(), 0, 0, 1, &rnd));
  setResourceBarrier(cmdList.Get(), bufferTarget.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
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
  ID3D12Device *dev;
  CHK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)));
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

  Sort(NSIZE, cmdList, rootSignature, pso, descHeapUav, gDev, cmdQueue, cmdAlloc, fence);
  ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);

  setResourceBarrier(cmdList.Get(), bufferDefault.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COPY_SOURCE);
  cmdList->CopyResource(bufferReadback.Get(), bufferDefault.Get());

  // Execute
  ExecuteAndWait(cmdList, cmdQueue, cmdAlloc, pso, fence);

  // Get system memory pointer
  void *data;
  CHK(bufferReadback->Map(0, nullptr, &data));

  int *dpointer = (int *)data;
  string str = "";
  for (size_t i = 0; i < NSIZE + 3; i++) {
    str += to_string(*dpointer) + ",";
    dpointer++;
  }
  bufferReadback->Unmap(0, nullptr);

  // Output
  cout << str << endl;
}

int wmain(int argc, wchar_t **argv) {
  proc();
  _getwch();

  return 0;
}