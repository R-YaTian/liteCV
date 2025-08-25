#pragma once
#ifndef LCV_CORE_ALGO_HPP
#define LCV_CORE_ALGO_HPP
#include <vector>
#include <cmath>
#include <cstring>
#include <cfloat>

#include "lcvdef.hpp"
#include "lcvtypes.hpp"
#include "lcvmath.hpp"

#define DIST_L2 2
#define CV_DIST_L2 DIST_L2

namespace lcv {

void fitLine(const std::vector<Point2d>& points,
             Vec4d& line,
             int distType = DIST_L2,   // Only support DIST_L2 currently
             double param = 0.0f,
             double reps = 0.01f,
             double aeps = 0.01f)
{
    if (points.size() < 2) {
        throw std::invalid_argument("At least 2 points are required for line fitting");
    }

    if (distType != DIST_L2) {
        throw std::invalid_argument("Only DIST_L2 is currently supported");
    }

    int count = static_cast<int>(points.size());

    double x = 0, y = 0, x2 = 0, y2 = 0, xy = 0, w = 0;
    double dx2, dy2, dxy;
    int i;
    float t;

    // Calculating the average of x and y...
    for( i = 0; i < count; i += 1 )
    {
        x += points[i].x;
        y += points[i].y;
        x2 += points[i].x * points[i].x;
        y2 += points[i].y * points[i].y;
        xy += points[i].x * points[i].y;
    }
    w = (float) count;

    x /= w;
    y /= w;
    x2 /= w;
    y2 /= w;
    xy /= w;

    dx2 = x2 - x * x;
    dy2 = y2 - y * y;
    dxy = xy - x * y;

    t = (float) atan2( 2 * dxy, dx2 - dy2 ) / 2;
    line[0] = (float) cos( t );
    line[1] = (float) sin( t );
    line[2] = (float) x;
    line[3] = (float) y;
}

void split(const Matrix& src, std::vector<Matrix>& mv)
{
    if (src.empty())
        throw std::runtime_error("split: source matrix is empty");

    int channels = src.channels();
    if (channels == 1) {
        mv.resize(1);
        mv[0] = src.clone();
        return;
    }

    mv.resize(channels);

    // Create single-channel matrices - use depth() to build correct single-channel type
    int depth = src.depth();
    int single_channel_type;
    if (depth == LCV_8U) single_channel_type = LCV_8UC1;
    else if (depth == LCV_8S) single_channel_type = LCV_8SC1;
    else if (depth == LCV_16U) single_channel_type = LCV_16UC1;
    else if (depth == LCV_16S) single_channel_type = LCV_16SC1;
    else if (depth == LCV_32U) single_channel_type = LCV_32UC1;
    else if (depth == LCV_32S) single_channel_type = LCV_32SC1;
    else if (depth == LCV_32F) single_channel_type = LCV_32FC1;
    else if (depth == LCV_64F) single_channel_type = LCV_64FC1;
    else throw std::runtime_error("split: unsupported matrix depth");

    for (int c = 0; c < channels; c++) {
        mv[c].create(src.rows, src.cols, single_channel_type);
    }

    // Split channel data
    size_t elem_size = src.elemSize1(); // Size of each element in bytes

    for (int y = 0; y < src.rows; y++)
    {
        const uchar* src_ptr = src.ptr(y);
        for (int x = 0; x < src.cols; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                uchar* dst_ptr = mv[c].ptr(y, x);
                const uchar* src_elem = src_ptr + x * src.elemSize() + c * elem_size;
                memcpy(dst_ptr, src_elem, elem_size);
            }
        }
    }
}

void split(const Matrix& src, Matrix* mvbegin)
{
    std::vector<Matrix> mv;
    split(src, mv);
    for (int i = 0; i < src.channels(); i++)
        mvbegin[i] = mv[i];
}

void minMaxIdx(const Matrix& src, double* minVal, double* maxVal = nullptr,
               Point* minLoc = nullptr, Point* maxLoc = nullptr,
               const Matrix& mask = Matrix())
{
    if (src.empty())
        throw std::runtime_error("minMaxIdx: source matrix is empty");

    if (src.channels() != 1)
        throw std::runtime_error("minMaxIdx: source matrix must be single channel");

    double min_val = DBL_MAX;
    double max_val = -DBL_MAX;
    Point min_loc(-1, -1);
    Point max_loc(-1, -1);

    bool has_valid_pixel = false;
    int depth = src.depth();

    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            // Check mask
            if (!mask.empty()) {
                uchar mask_val = mask.at<uchar>(y, x);
                if (mask_val == 0) continue;
            }

            double val = 0.0f;

            // Read pixel value based on depth
            if (depth == LCV_8U)
                val = src.at<uchar>(y, x);
            else if (depth == LCV_8S)
                val = src.at<schar>(y, x);
            else if (depth == LCV_16U)
                val = src.at<ushort>(y, x);
            else if (depth == LCV_16S)
                val = src.at<short>(y, x);
            else if (depth == LCV_32U)
                val = src.at<uint>(y, x);
            else if (depth == LCV_32S)
                val = src.at<int>(y, x);
            else if (depth == LCV_32F)
                val = src.at<float>(y, x);
            else if (depth == LCV_64F)
                val = src.at<double>(y, x);
            else
                throw std::runtime_error("minMaxIdx: unsupported matrix depth");

            if (!has_valid_pixel) {
                min_val = max_val = val;
                min_loc = max_loc = Point(x, y);
                has_valid_pixel = true;
            } else {
                if (val < min_val) {
                    min_val = val;
                    min_loc = Point(x, y);
                }
                if (val > max_val) {
                    max_val = val;
                    max_loc = Point(x, y);
                }
            }
        }
    }

    if (!has_valid_pixel)
        throw std::runtime_error("minMaxIdx: no valid pixels found");

    if (minVal) *minVal = min_val;
    if (maxVal) *maxVal = max_val;
    if (minLoc) *minLoc = min_loc;
    if (maxLoc) *maxLoc = max_loc;
}

Matrix RGB2YCbCr(Matrix in) {
    in.convertTo(in, LCV_32FC3);
    int h = in.rows;
    int w = in.cols;

    Matrix ycbcr(h, w, LCV_8UC3);
    float32 R, G, B;
    float64 f = pow(2, 16);

    for (int i = 0;i < h;i++)
    {
        for (int j = 0;j < w;j++)
        {
            Vec3f pixel = in.at<Vec3f>(i, j);
            R = pixel[2];
            G = pixel[1];
            B = pixel[0];

            Vec3b color;
            color[0] = lcvClamp<uchar>(round((0.2568 * R + 0.5041 * G + 0.0979 * B + 16) * f) / f, 16, 235);
            color[1] = lcvClamp<uchar>(round((-0.1482 * R - 0.2910 * G + 0.4392 * B + 128) * f) / f, 16, 240);
            color[2] = lcvClamp<uchar>(round((0.4392 * R - 0.3678 * G - 0.0714 * B + 128) * f) / f, 16, 240);

            ycbcr.at<Vec3b>(i, j) = color;
        }
    }

    return ycbcr;
}

} // namespace lcv
#endif // LCV_CORE_ALGO_HPP
