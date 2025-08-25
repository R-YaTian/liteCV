#pragma once
#ifndef LCV_CORE_LCVMATH_HPP
#define LCV_CORE_LCVMATH_HPP
#if (defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __SSE2__ && !defined __APPLE__)
#include <xmmintrin.h>
#endif
#include <cmath>

#include "lcvdef.hpp"


namespace lcv
{
    template <typename T>
    inline T lcvClamp(T v, T lo, T hi)
    {
        return std::max(lo, std::min(v, hi));
    }

    template<typename T>
    T inline lcvModulo(T a, T b)
    {
        return (b + (a % b)) % b;
    } // lcvModulo

    template<typename Float>
    int inline lcvRound(Float value)
    {
#if (defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__ && defined __SSE2__ && !defined __APPLE__)
        __m128 t = _mm_set_ss( value );
        return _mm_cvtss_si32(t);
#elif defined _MSC_VER && defined _M_IX86
        int t;
        __asm
        {
            fld value;
            fistp t;
        }
        return t;
#elif defined __GNUC__
        return (int)lrintf(value);
#else
        /* it's ok if round does not comply with IEEE754 standard;
        the tests should allow +/-1 difference when the tested functions use round */
        return (int)(value + (value >= 0 ? 0.5f : -0.5f));
#endif
    } // lcvRound

    template<typename Float>
    int inline lcvFloor(Float value)
    {
#if (defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __SSE2__ && !defined __APPLE__)
        __m128 t = _mm_set_ss( value );
        int i = _mm_cvtss_si32(t);
        return i - _mm_movemask_ps(_mm_cmplt_ss(t, _mm_cvtsi32_ss(t,i)));
#elif defined __GNUC__
        int i = (int)value;
        return i - (i > value);
#else
        int i = lcvRound(value);
        float diff = (float)(value - i);
        return i - (diff < 0);
#endif
    } // lcvFloor

    template<typename Float>
    int inline lcvCeil(Float value)
    {
#if (defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __SSE2__&& !defined __APPLE__)
        __m128 t = _mm_set_ss( value );
        int i = _mm_cvtss_si32(t);
        return i + _mm_movemask_ps(_mm_cmplt_ss(_mm_cvtsi32_ss(t,i), t));
#elif defined __GNUC__
        int i = (int)value;
        return i + (i < value);
#else
        int i = lcvRound(value);
        float diff = (float)(i - value);
        return i + (diff < 0);
#endif
    } // lcvCeil
} // namespace lcv
#endif // LCV_CORE_LCVMATH_HPP
