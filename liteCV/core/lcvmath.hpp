#pragma once
#ifndef LCV_CORE_LCVMATH_HPP
#define LCV_CORE_LCVMATH_HPP
#include <cmath>

#include "lcvdef.hpp"


namespace lcv
{
    template<typename T>
    T inline lcvModulo(T a, T b)
    {
        return (b + (a % b)) % b;
    } // lcvModulo

    template<typename Float>
    double inline lcvRound(Float v)
    {
        return round(v);
    } // lcvRound

    template<typename Float>
    int inline lcvFloor(Float v)
    {
        return (int)v - (((int)v > v) ? 1 : 0);
    } // lcvFloor

    template<typename Float>
    int inline lcvCeil(Float v)
    {
        return (int)v + (((int)v < v) ? 1 : 0);
    } // lcvCeil
} // namespace lcv
#endif // LCV_CORE_LCVMATH_HPP
