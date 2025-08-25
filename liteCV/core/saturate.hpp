#pragma once
#ifndef LCV_CORE_SATURATE_HPP
#define LCV_CORE_SATURATE_HPP
#include <utility>

#include "lcvdef.hpp"
#include "lcvmath.hpp"


namespace lcv
{
    template<typename T>
    T inline saturate_cast(uchar v)
    {
        return T(v);
    }

    template<typename T>
    T inline saturate_cast(schar v)
    {
        return T(v);
    }

    template<typename T>
    T inline saturate_cast(ushort v)
    {
        return T(v);
    }

    template<typename T>
    T inline saturate_cast(short v)
    {
        return T(v);
    }

    template<typename T>
    T inline saturate_cast(uint v)
    {
        return T(v);
    }

    template<typename T>
    T inline saturate_cast(int v)
    {
        return T(v);
    }

    template<typename T>
    T inline saturate_cast(float32 v)
    {
        return T(v);
    }

    template<typename T>
    T inline saturate_cast(float64 v)
    {
        return T(v);
    }

    template<typename T>
    T inline saturate_cast(int64 v)
    {
        return T(v);
    }

    template<typename T>
    T inline saturate_cast(uint64 v)
    {
        return T(v);
    }

    template<>
    uchar inline saturate_cast<uchar>(schar v)
    {
        return (uchar)std::max((int)v, 0);
    }

    template<>
    uchar inline saturate_cast<uchar>(ushort v)
    {
        return (uchar)std::min((uint)v, (uint)UCHAR_MAX);
    }

    template<>
    uchar inline saturate_cast<uchar>(int v)
    {
        return (uchar)((uint)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
    }

    template<>
    uchar inline saturate_cast<uchar>(short v)
    {
        return saturate_cast<uchar>((int)v);
    }

    template<>
    uchar inline saturate_cast<uchar>(uint v)
    {
        return (uchar)std::min(v, (uint)UCHAR_MAX);
    }

    template<>
    uchar inline saturate_cast<uchar>(float32 v)
    {
        int iv = lcvRound(v);
        return saturate_cast<uchar>(iv);
    }

    template<>
    uchar inline saturate_cast<uchar>(float64 v)
    {
        int iv = lcvRound(v);
        return saturate_cast<uchar>(iv);
    }

    template<>
    uchar inline saturate_cast<uchar>(int64 v)
    {
        return (uchar)((uint64)v <= (uint64)UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
    }

    template<>
    uchar inline saturate_cast<uchar>(uint64 v)
    {
        return (uchar)std::min(v, (uint64)UCHAR_MAX);
    }

    template<>
    schar inline saturate_cast<schar>(uchar v)
    {
        return (schar)std::min((int)v, SCHAR_MAX);
    }

    template<>
    schar inline saturate_cast<schar>(ushort v)
    {
        return (schar)std::min((uint)v, (uint)SCHAR_MAX);
    }

    template<>
    schar inline saturate_cast<schar>(int v)
    {
        return (schar)((uint)(v-SCHAR_MIN) <= (uint)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN);
    }

    template<>
    schar inline saturate_cast<schar>(short v)
    {
        return saturate_cast<schar>((int)v);
    }

    template<>
    schar inline saturate_cast<schar>(uint v)
    {
        return (schar)std::min(v, (uint)SCHAR_MAX);
    }

    template<>
    schar inline saturate_cast<schar>(float32 v)
    {
        int iv = lcvRound(v);
        return saturate_cast<schar>(iv);
    }

    template<>
    schar inline saturate_cast<schar>(float64 v)
    {
        int iv = lcvRound(v);
        return saturate_cast<schar>(iv);
    }

    template<>
    schar inline saturate_cast<schar>(int64 v)
    {
        return (schar)((uint64)((int64)v-SCHAR_MIN) <= (uint64)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN);
    }

    template<>
    schar inline saturate_cast<schar>(uint64 v)
    {
        return (schar)std::min(v, (uint64)SCHAR_MAX);
    }

    template<>
    ushort inline saturate_cast<ushort>(schar v)
    {
        return (ushort)std::max((int)v, 0);
    }

    template<>
    ushort inline saturate_cast<ushort>(short v)
    {
        return (ushort)std::max((int)v, 0);
    }

    template<>
    ushort inline saturate_cast<ushort>(uint v)
    {
        return (ushort)std::min(v, (uint)USHRT_MAX);
    }

    template<>
    ushort inline saturate_cast<ushort>(int v)
    {
        return (ushort)((uint)v <= (uint)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0);
    }

    template<>
    ushort inline saturate_cast<ushort>(float32 v)
    {
        int iv = lcvRound(v);
        return saturate_cast<ushort>(iv);
    }

    template<>
    ushort inline saturate_cast<ushort>(float64 v)
    {
        int iv = lcvRound(v);
        return saturate_cast<ushort>(iv);
    }

    template<>
    ushort inline saturate_cast<ushort>(int64 v)
    {
        return (ushort)((uint64)v <= (uint64)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0);
    }

    template<>
    ushort inline saturate_cast<ushort>(uint64 v)
    {
        return (ushort)std::min(v, (uint64)USHRT_MAX);
    }

    template<>
    short inline saturate_cast<short>(ushort v)
    {
        return (short)std::min((int)v, SHRT_MAX);
    }

    template<>
    short inline saturate_cast<short>(uint v)
    {
        return (short)std::min(v, (uint)SHRT_MAX);
    }

    template<>
    short inline saturate_cast<short>(int v)
    {
        return (short)((uint)(v - SHRT_MIN) <= (uint)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN);
    }

    template<>
    short inline saturate_cast<short>(float32 v)
    {
        int iv = lcvRound(v);
        return saturate_cast<short>(iv);
    }

    template<>
    short inline saturate_cast<short>(float64 v)
    {
        int iv = lcvRound(v);
        return saturate_cast<short>(iv);
    }

    template<>
    short inline saturate_cast<short>(int64 v)
    {
        return (short)((uint64)((int64)v - SHRT_MIN) <= (uint64)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN);
    }

    template<>
    short inline saturate_cast<short>(uint64 v)
    {
        return (short)std::min(v, (uint64)SHRT_MAX);
    }

    template<>
    uint inline saturate_cast<uint>(schar v)
    {
        return (uint)std::max(v, (schar)0);
    }

    template<>
    uint inline saturate_cast<uint>(short v)
    {
        return (uint)std::max(v, (short)0);
    }

    template<>
    uint inline saturate_cast<uint>(int v)
    {
        return (uint)std::max(v, (int)0);
    }

    template<>
    uint inline saturate_cast<uint>(float32 v)
    {
        return static_cast<uint>(lcvRound(v));
    }

    template<>
    uint inline saturate_cast<uint>(float64 v)
    {
        return static_cast<uint>(lcvRound(v));
    }

    template<>
    uint inline saturate_cast<uint>(int64 v)
    {
        return (uint)((uint64)v <= (uint64)UINT_MAX ? v : v > 0 ? UINT_MAX : 0);
    }

    template<>
    uint inline saturate_cast<uint>(uint64 v)
    {
        return (uint)std::min(v, (uint64)UINT_MAX);
    }

    template<>
    int inline saturate_cast<int>(uint v)
    {
        return (int)std::min(v, (uint)INT_MAX);
    }

    template<>
    int inline saturate_cast<int>(float32 v)
    {
        return lcvRound(v);
    }

    template<>
    int inline saturate_cast<int>(float64 v)
    {
        return lcvRound(v);
    }

    template<>
    int inline saturate_cast<int>(int64 v)
    {
        return (int)((uint64)(v - INT_MIN) <= (uint64)UINT_MAX ? v : v > 0 ? INT_MAX : INT_MIN);
    }

    template<>
    int inline saturate_cast<int>(uint64 v)
    {
        return (int)std::min(v, (uint64)INT_MAX);
    }

    template<> inline uint64 saturate_cast<uint64>(schar v)      { return (uint64)std::max(v, (schar)0); }

    template<> inline uint64 saturate_cast<uint64>(short v)      { return (uint64)std::max(v, (short)0); }

    template<> inline uint64 saturate_cast<uint64>(int v)        { return (uint64)std::max(v, (int)0); }

    template<> inline uint64 saturate_cast<uint64>(int64 v)      { return (uint64)std::max(v, (int64)0); }

    template<> inline int64 saturate_cast<int64>(uint64 v)       { return (int64)std::min(v, (uint64)LLONG_MAX); }
} // namespace lcv
#endif // LCV_CORE_SATURATE_HPP
