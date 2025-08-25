#pragma once
#ifndef LCV_IMGPROC_INTERPOLATION_HPP
#define LCV_IMGPROC_INTERPOLATION_HPP
#include "liteCV/core/lcvdef.hpp"
#include "liteCV/core/lcvmath.hpp"
#include "liteCV/core/matrix.hpp"


namespace lcv
{
    enum InterpolationFlags
    {
        INTER_NEAREST = 0,
        INTER_LINEAR = 1
    }; // enum InterpolationFlags

    class InterpolationPolicy
    {
    public:
        virtual double interpolate(const Matrix& src, int dwidth, int dheight, int dx, int dy, int ch) = 0;
    }; // class InterpolationPolicy

    class NearestInterpolationPolicy : public InterpolationPolicy
    {
    public:
        double interpolate(const Matrix& src, int dwidth, int dheight, int dx, int dy, int ch) final
        {
            const int sx = lcvRound(((float)dx / dwidth) * src.cols);
            const int sy = lcvRound(((float)dy / dheight) * src.rows);
            return (double)src.ptr<uchar>(sy, sx)[ch];
        }
    }; // class NearestInterpolationPolicy

    class LinearInterpolationPolicy : public InterpolationPolicy
    {
        double inline calc_ratio(double p)
        {
            return p - (int)p;
        }

    public:
        double interpolate(const Matrix& src, int dwidth, int dheight, int dx, int dy, int ch) final
        {
            float sx = (dx + 0.5) * src.cols / (double)dwidth - 0.5;
            float sy = (dy + 0.5) * src.rows / (double)dheight - 0.5;

            int x0 = lcvFloor(sx);
            int y0 = lcvFloor(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            x0 = lcvClamp(x0, 0, src.cols - 1);
            y0 = lcvClamp(y0, 0, src.rows - 1);
            x1 = lcvClamp(x1, 0, src.cols - 1);
            y1 = lcvClamp(y1, 0, src.rows - 1);

            float alpha = sx - x0;
            float beta  = sy - y0;

            uchar a = src.ptr<uchar>(y0, x0)[ch];
            uchar b = src.ptr<uchar>(y0, x1)[ch];
            uchar c = src.ptr<uchar>(y1, x0)[ch];
            uchar d = src.ptr<uchar>(y1, x1)[ch];

            return ((1 - alpha) * (1 - beta) * a) +
                   alpha       * (1 - beta) * b +
                   (1 - alpha) * beta       * c +
                   alpha       * beta       * d;
        }
    }; // class LinearInterpolationPolicy

    class InterpolationPolicyStorage
    {
    private:
        InterpolationPolicyStorage() = delete;

    public:
        static InterpolationPolicy* get_policy(int flag = INTER_NEAREST)
        {
            static NearestInterpolationPolicy nip;
            static LinearInterpolationPolicy lip;

            switch (flag)
            {
            case INTER_NEAREST:
                return static_cast<InterpolationPolicy*>(&nip);

            case INTER_LINEAR:
                return static_cast<InterpolationPolicy*>(&lip);
            }

            return nullptr;
        }
    }; // class InterpolationPolicyStorage
} // namespace lcv
#endif // LCV_IMGPROC_INTERPOLATION_HPP
