#pragma once

#include <climits>
#include <cstdint>

#ifndef NO_PYTORCH
#include <torch/csrc/stable/tensor.h>
#endif

namespace feather_attn {

#ifndef NO_PYTORCH

// Stride check for a tensor with exactly `ndim` dimensions. Size-1
// dimensions are ignored, matching Tensor::is_contiguous().
inline bool IsContiguousND(const torch::stable::Tensor& tensor, int64_t ndim)
{
    __int128 expected_stride = 1;
    for(int64_t dim = ndim - 1; dim >= 0; --dim)
    {
        if(expected_stride > INT64_MAX)
            return false;
        if(tensor.size(dim) != 1 &&
           tensor.stride(dim) != static_cast<int64_t>(expected_stride))
            return false;
        expected_stride *= tensor.size(dim);
    }
    return true;
}

inline bool IsInt32Addressable(
    const torch::stable::Tensor& tensor,
    int64_t element_size)
{
    __int128 max_element_offset = 0;
    for(int64_t dim = 0; dim < tensor.dim(); ++dim)
    {
        const int64_t stride = tensor.stride(dim);
        if(stride < 0 || stride > INT32_MAX)
            return false;
        max_element_offset +=
            static_cast<__int128>(tensor.size(dim) - 1) * stride;
    }
    return max_element_offset * element_size <= INT32_MAX;
}

#endif // NO_PYTORCH

// Shared NHD D64 head-partition heuristic. Both the forward strided launch
// path and the backward NHD path use the same LLC-aware group count.
inline int64_t ResolveNhdD64StridedGroupCount(int64_t heads, int64_t n_kv)
{
    constexpr __int128 kLlcBytes = static_cast<__int128>(32) * 1024 * 1024;
    constexpr int64_t kHeadDim  = 64;
    const __int128 kv_bytes_per_head =
        static_cast<__int128>(n_kv) * kHeadDim * 2 * sizeof(uint16_t);
    const __int128 total_kv_bytes =
        static_cast<__int128>(heads) * kv_bytes_per_head;
    if(total_kv_bytes * 2 < kLlcBytes * 3 || kv_bytes_per_head > kLlcBytes)
        return 1;

    const int64_t group_size = static_cast<int64_t>(kLlcBytes / kv_bytes_per_head);
    if(group_size < 4 || group_size >= heads)
        return 1;

    int64_t group_count = (heads + group_size - 1) / group_size;
    if(group_count < 3)
        return 1;
    // A physical-head stride divisible by eight recreates the NHD partition cliff.
    if(group_count % 8 == 0)
        ++group_count;
    return group_count;
}

} // namespace feather_attn
