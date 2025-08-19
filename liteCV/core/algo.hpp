#pragma once
#ifndef LCV_CORE_ALGO_HPP
#define LCV_CORE_ALGO_HPP
#include <vector>
#include <cmath>
#include <cstring>
#include <cfloat>

#include "lcvdef.hpp"
#include "lcvtypes.hpp"

#define DIST_L2 2

namespace lcv {

void fitLine(const std::vector<lcv::Point2d>& points,
             lcv::Vec4d& line,
             int distType = DIST_L2,   // 暂时只需支持 DIST_L2
             double param = 0,
             double reps = 0.01,
             double aeps = 0.01)
{
    if (points.size() < 2)
        throw std::runtime_error("fitLine: at least 2 points are required");

    if (distType != DIST_L2)  // cv::DIST_L2
        throw std::runtime_error("fitLine: only DIST_L2 is supported");

    // 1. 计算均值
    double meanX = 0.0, meanY = 0.0;
    for (const auto& p : points) {
        meanX += p.x;
        meanY += p.y;
    }
    meanX /= points.size();
    meanY /= points.size();

    // 2. 计算协方差矩阵
    double Sxx = 0, Syy = 0, Sxy = 0;
    for (const auto& p : points) {
        double dx = p.x - meanX;
        double dy = p.y - meanY;
        Sxx += dx * dx;
        Syy += dy * dy;
        Sxy += dx * dy;
    }
    Sxx /= points.size();
    Syy /= points.size();
    Sxy /= points.size();

    // 3. 求解主方向（最大特征值对应的特征向量）
    double theta = 0.5 * atan2(2 * Sxy, Sxx - Syy);
    double vx = cos(theta);
    double vy = sin(theta);

    // 4. 输出结果 [vx, vy, x0, y0]
    line[0] = vx;
    line[1] = vy;
    line[2] = meanX;
    line[3] = meanY;
}

void split(const Matrix& src, std::vector<Matrix>& mv)
{
    if (src.empty())
        throw std::runtime_error("split: source matrix is empty");
    
    int channels = src.channels();
    if (channels == 1) {
        mv.resize(1);
        src.copyTo(mv[0]);
        return;
    }
    
    mv.resize(channels);
    
    // 创建单通道矩阵 - 使用depth()构建正确的单通道类型
    int depth = src.depth();
    int single_channel_type;
    if (depth == LCV_8U) single_channel_type = LCV_8UC1;
    else if (depth == LCV_8S) single_channel_type = LCV_8SC1;
    else if (depth == LCV_16U) single_channel_type = LCV_16UC1;
    else if (depth == LCV_16S) single_channel_type = LCV_16SC1;
    else if (depth == LCV_32S) single_channel_type = LCV_32SC1;
    else if (depth == LCV_32F) single_channel_type = LCV_32FC1;
    else if (depth == LCV_64F) single_channel_type = LCV_64FC1;
    else throw std::runtime_error("split: unsupported matrix depth");
    
    for (int c = 0; c < channels; c++) {
        mv[c].create(src.cols, src.rows, single_channel_type);
    }
    
    // 分离通道数据
    size_t elem_size = src.elemSize1(); // 单个元素的字节大小
    
    for (int y = 0; y < src.rows; y++) {
        const uchar* src_ptr = src.ptr(y);
        for (int x = 0; x < src.cols; x++) {
            for (int c = 0; c < channels; c++) {
                uchar* dst_ptr = mv[c].ptr(y, x);
                const uchar* src_elem = src_ptr + x * src.elemSize() + c * elem_size;
                memcpy(dst_ptr, src_elem, elem_size);
            }
        }
    }
}

void minMaxIdx(const Matrix& src, double* minVal, double* maxVal = nullptr, 
               Point* minLoc = nullptr, Point* maxLoc = nullptr, 
               const Matrix& mask = Matrix())
{
    if (src.empty())
        throw std::runtime_error("minMaxIdx: source matrix is empty");
    
    if (src.channels() != 1)
        throw std::runtime_error("minMaxIdx: source matrix must be single channel");
    
    if (!mask.empty() && (mask.rows != src.rows || mask.cols != src.cols))
        throw std::runtime_error("minMaxIdx: mask size must match source matrix size");
    
    double min_val = DBL_MAX;
    double max_val = -DBL_MAX;
    Point min_loc(-1, -1);
    Point max_loc(-1, -1);
    
    bool has_valid_pixel = false;
    int depth = src.depth();
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // 检查mask
            if (!mask.empty()) {
                uchar mask_val = mask.at<uchar>(y, x);
                if (mask_val == 0) continue;
            }
            
            double val = 0.0;
            
            // 根据深度读取像素值
            if (depth == LCV_8U)
                val = src.at<uchar>(y, x);
            else if (depth == LCV_8S)
                val = src.at<schar>(y, x);
            else if (depth == LCV_16U)
                val = src.at<ushort>(y, x);
            else if (depth == LCV_16S)
                val = src.at<short>(y, x);
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

} // namespace lcv
#endif // LCV_CORE_ALGO_HPP
