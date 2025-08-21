#pragma once
#ifndef LCV_CORE_MATRIX_HPP
#define LCV_CORE_MATRIX_HPP
#include <string>
#include <atomic>
#include <regex>

#include "lcvdef.hpp"
#include "lcvtypes.hpp"
#include "saturate.hpp"


#define LCV_8U      (lcv::ConstMatrixType<8, 0, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_8UC1    (lcv::ConstMatrixType<8, 1, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_8UC2    (lcv::ConstMatrixType<8, 2, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_8UC3    (lcv::ConstMatrixType<8, 3, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_8UC4    (lcv::ConstMatrixType<8, 4, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_8S      (lcv::ConstMatrixType<8, 0, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_8SC1    (lcv::ConstMatrixType<8, 1, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_8SC2    (lcv::ConstMatrixType<8, 2, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_8SC3    (lcv::ConstMatrixType<8, 3, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_8SC4    (lcv::ConstMatrixType<8, 4, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_16U     (lcv::ConstMatrixType<16, 0, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_16UC1   (lcv::ConstMatrixType<16, 1, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_16UC2   (lcv::ConstMatrixType<16, 2, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_16UC3   (lcv::ConstMatrixType<16, 3, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_16UC4   (lcv::ConstMatrixType<16, 4, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_16S     (lcv::ConstMatrixType<16, 0, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_16SC1   (lcv::ConstMatrixType<16, 1, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_16SC2   (lcv::ConstMatrixType<16, 2, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_16SC3   (lcv::ConstMatrixType<16, 3, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_16SC4   (lcv::ConstMatrixType<16, 4, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_32U     (lcv::ConstMatrixType<32, 0, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_32UC1   (lcv::ConstMatrixType<32, 1, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_32UC2   (lcv::ConstMatrixType<32, 2, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_32UC3   (lcv::ConstMatrixType<32, 3, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_32UC4   (lcv::ConstMatrixType<32, 4, lcv::MatrixType::UNSIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_32S     (lcv::ConstMatrixType<32, 0, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_32SC1   (lcv::ConstMatrixType<32, 1, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_32SC2   (lcv::ConstMatrixType<32, 2, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_32SC3   (lcv::ConstMatrixType<32, 3, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_32SC4   (lcv::ConstMatrixType<32, 4, lcv::MatrixType::SIGNED_INTEGER_NUMBER>().constant.packed.value)
#define LCV_32F     (lcv::ConstMatrixType<32, 0, lcv::MatrixType::REAL_NUMBER>().constant.packed.value)
#define LCV_32FC1   (lcv::ConstMatrixType<32, 1, lcv::MatrixType::REAL_NUMBER>().constant.packed.value)
#define LCV_32FC2   (lcv::ConstMatrixType<32, 2, lcv::MatrixType::REAL_NUMBER>().constant.packed.value)
#define LCV_32FC3   (lcv::ConstMatrixType<32, 3, lcv::MatrixType::REAL_NUMBER>().constant.packed.value)
#define LCV_32FC4   (lcv::ConstMatrixType<32, 4, lcv::MatrixType::REAL_NUMBER>().constant.packed.value)
#define LCV_64F     (lcv::ConstMatrixType<64, 0, lcv::MatrixType::REAL_NUMBER>().constant.packed.value)
#define LCV_64FC1   (lcv::ConstMatrixType<64, 1, lcv::MatrixType::REAL_NUMBER>().constant.packed.value)
#define LCV_64FC2   (lcv::ConstMatrixType<64, 2, lcv::MatrixType::REAL_NUMBER>().constant.packed.value)
#define LCV_64FC3   (lcv::ConstMatrixType<64, 3, lcv::MatrixType::REAL_NUMBER>().constant.packed.value)
#define LCV_64FC4   (lcv::ConstMatrixType<64, 4, lcv::MatrixType::REAL_NUMBER>().constant.packed.value)

#define CV_8U     LCV_8U
#define CV_8UC1   LCV_8UC1
#define CV_8UC2   LCV_8UC2
#define CV_8UC3   LCV_8UC3
#define CV_8UC4   LCV_8UC4
#define CV_8S     LCV_8S
#define CV_8SC1   LCV_8SC1
#define CV_8SC2   LCV_8SC2
#define CV_8SC3   LCV_8SC3
#define CV_8SC4   LCV_8SC4
#define CV_16U    LCV_16U
#define CV_16UC1  LCV_16UC1
#define CV_16UC2  LCV_16UC2
#define CV_16UC3  LCV_16UC3
#define CV_16UC4  LCV_16UC4
#define CV_16S    LCV_16S
#define CV_16SC1  LCV_16SC1
#define CV_16SC2  LCV_16SC2
#define CV_16SC3  LCV_16SC3
#define CV_16SC4  LCV_16SC4
#define CV_32U    LCV_32U
#define CV_32UC1  LCV_32UC1
#define CV_32UC2  LCV_32UC2
#define CV_32UC3  LCV_32UC3
#define CV_32UC4  LCV_32UC4
#define CV_32S    LCV_32S
#define CV_32SC1  LCV_32SC1
#define CV_32SC2  LCV_32SC2
#define CV_32SC3  LCV_32SC3
#define CV_32SC4  LCV_32SC4
#define CV_32F    LCV_32F
#define CV_32FC1  LCV_32FC1
#define CV_32FC2  LCV_32FC2
#define CV_32FC3  LCV_32FC3
#define CV_32FC4  LCV_32FC4
#define CV_64F    LCV_64F
#define CV_64FC1  LCV_64FC1
#define CV_64FC2  LCV_64FC2
#define CV_64FC3  LCV_64FC3
#define CV_64FC4  LCV_64FC4

namespace lcv
{
    struct MatrixType
    {
        const static int INVALID_NUMBER = 0b00;
        const static int SIGNED_INTEGER_NUMBER = 0b01;
        const static int UNSIGNED_INTEGER_NUMBER = 0b10;
        const static int REAL_NUMBER = 0b11;

        static int find_ntype(int bits, char typec)
        {
            if (typec == 's')
                return SIGNED_INTEGER_NUMBER;
            else if (typec == 'u')
                return UNSIGNED_INTEGER_NUMBER;
            else if (typec == 'f')
                return REAL_NUMBER;
            return INVALID_NUMBER;
        }

        static std::string find_dtype(int bits, char typec_or_ntype)
        {
            if (bits == 8 && (typec_or_ntype == 'u' || typec_or_ntype == UNSIGNED_INTEGER_NUMBER))
                return "uint8";
            else if (bits == 8 && (typec_or_ntype == 'i' || typec_or_ntype == SIGNED_INTEGER_NUMBER))
                return "int8";
            else if (bits == 16 && (typec_or_ntype == 'u' || typec_or_ntype == UNSIGNED_INTEGER_NUMBER))
                return "uint16";
            else if (bits == 16 && (typec_or_ntype == 'i' || typec_or_ntype == SIGNED_INTEGER_NUMBER))
                return "int16";
            else if (bits == 32 && (typec_or_ntype == 'u' || typec_or_ntype == UNSIGNED_INTEGER_NUMBER))
                return "uint32";
            else if (bits == 32 && (typec_or_ntype == 'i' || typec_or_ntype == SIGNED_INTEGER_NUMBER))
                return "int32";
            else if (bits == 32 && (typec_or_ntype == 'f' || typec_or_ntype == REAL_NUMBER))
                return "float32";
            else if (bits == 64 && (typec_or_ntype == 'f' || typec_or_ntype == REAL_NUMBER))
                return "float64";
            return "nat";
        }

        static void from_dtype(const std::string& dtype, int& bits, int& ntype)
        {
            if (dtype == "uint8")
            {
                bits = 8;
                ntype = UNSIGNED_INTEGER_NUMBER;
            }
            else if (dtype == "int8")
            {
                bits = 8;
                ntype = SIGNED_INTEGER_NUMBER;
            }
            else if (dtype == "uint16")
            {
                bits = 16;
                ntype = UNSIGNED_INTEGER_NUMBER;
            }
            else if (dtype == "int16")
            {
                bits = 16;
                ntype = SIGNED_INTEGER_NUMBER;
            }
            else if (dtype == "uint32")
            {
                bits = 32;
                ntype = UNSIGNED_INTEGER_NUMBER;
            }
            else if (dtype == "int32")
            {
                bits = 32;
                ntype = SIGNED_INTEGER_NUMBER;
            }
            else if (dtype == "float32")
            {
                bits = 32;
                ntype = REAL_NUMBER;
            }
            else if (dtype == "float64")
            {
                bits = 64;
                ntype = REAL_NUMBER;
            }
            else
            {
                bits = 0;
                ntype = INVALID_NUMBER;
            }

            assert(bits > 0 && (bits & (bits - 1)) == 0);
            assert(ntype != INVALID_NUMBER);
        }

        static void from_channel_string(const std::string& channel_string, int& channels, int& bits, int& ntype)
        {
            std::regex re("(\\d+)([f|i|u])c(\\d+)");
            std::smatch match;
            if (std::regex_match(channel_string, match, re))
            {
                std::string _dtype;
                int _channels, _bits, _ntype;

                _bits = std::stoi(match[1]);
                _channels = std::stoi(match[3]);
                _ntype = find_ntype(_bits, match[2].str()[0]);

                assert(_bits > 0 && !(_bits & (_bits - 1)));
                assert(_channels > 0);
                assert(_ntype != INVALID_NUMBER);

                channels = _channels;
                bits = _bits;
                ntype = _ntype;
            }
        }

        union
        {
            short int value;
            struct
            {
                int bits : 8;
                int channels : 4;
                int ntype : 3;
            } fields;
        } packed;

        MatrixType(short int value = 0)
            : packed({ value }) {}

        MatrixType(int bits, int channels, int ntype)
            : packed({ 0 })
        {
            packed.fields.bits = bits;
            packed.fields.channels = channels;
            packed.fields.ntype = ntype;
        }

        MatrixType(int channels, const std::string& dtype)
            : packed({ 0 })
        {
            int bits, ntype;
            from_dtype(dtype, bits, ntype);

            packed.fields.bits = bits;
            packed.fields.channels = channels;
            packed.fields.ntype = ntype;
        }

        explicit MatrixType(const std::string& channel_string)
            : packed({ 0 })
        {
            int channels, bits, ntype;
            from_channel_string(channel_string, channels, bits, ntype);

            packed.fields.bits = bits;
            packed.fields.channels = channels;
            packed.fields.ntype = ntype;
        }

        MatrixType(const MatrixType& another)
        {
            packed.value = another.packed.value;
        }

        MatrixType& operator=(const MatrixType& another)
        {
            packed.value = another.packed.value;
            return *this;
        }

        std::string dtype() const
        {
            return find_dtype(packed.fields.bits, packed.fields.ntype);
        }

        int channels() const // A number of channels
        {
            return packed.fields.channels;
        }

        int depth() const // Depth of element LCV_8U, LCV_8S, ...
        {
            MatrixType tmp(*this);
            tmp.packed.fields.channels = 0;
            return tmp.packed.value;
        }

        int bbp() const // Bit per pixel
        {
            return packed.fields.bits * packed.fields.channels;
        }

        bool has_sign() const // signed integer or real
        {
            return packed.fields.ntype == SIGNED_INTEGER_NUMBER || packed.fields.ntype == REAL_NUMBER;
        }

        bool is_integer() const // inter or not
        {
            return !(packed.fields.ntype == INVALID_NUMBER || packed.fields.ntype == REAL_NUMBER);
        }

        bool is_real() const // real or not
        {
            return !is_integer();
        }
    }; // struct MatrixType

    template<int Bits, int Channels, int NType>
    struct ConstMatrixType
    {
        MatrixType constant;

        constexpr ConstMatrixType()
            : constant({ Bits, Channels, NType })
        {}
    }; // struct ConstMatrixType

    class Matrix
    {
    public:
        int cols, rows;

        struct
        {
            int pixelstep;
            int linestep;
        } step_info;

        MatrixType type_info;

        uchar* data;
        uchar* datastart;
        uchar* dataend;
        uchar* datalimit;

    private:
        std::atomic<unsigned long long>* refcount;

    private:
        void incref()
        {
            if (refcount == nullptr)
                refcount = new std::atomic<unsigned long long>(0l);
            ++(*refcount);
        }

        void decref()
        {
            if (refcount != nullptr)
            {
                --(*refcount);
                if ((*refcount) == 0)
                {
                    if (data != NULL)
                        free(data);
                    delete refcount;
                }
            }
        }

    private:
        void init()
        {
            cols = rows = 0;
            step_info.linestep = step_info.pixelstep = 0;
            type_info.packed.value = 0;
            data = datastart = dataend = datalimit = NULL;
            refcount = nullptr;
        }

        void inline deep_copy(const Matrix& another)
        {
            create(another.rows, another.cols, another.type_info);

            if (!another.isSubmatrix())
            {
                // Just copying data fully
                memcpy(data, another.data, step_info.linestep * rows);
            }
            else
            {
                // Copying data line by line
                for (int y = 0;y < another.rows;++y)
                {
                    memcpy(ptr(y), another.ptr(y), step_info.linestep);
                }
            }
        }

        void inline data_copy(const Matrix& another)
        {
            if (!another.isSubmatrix())
            {
                // Just copying data fully
                memcpy(data, another.data, step_info.linestep * rows);
            }
            else
            {
                // Copying data line by line
                for (int y = 0; y < another.rows; ++y)
                {
                    memcpy(ptr(y), another.ptr(y), step_info.linestep);
                }
            }
        }

        void inline swallow_copy(const Matrix& another, bool must_be_released)
        {
            if (must_be_released)
                decref();
            this->cols = another.cols;
            this->rows = another.rows;
            this->step_info = another.step_info;
            this->type_info = another.type_info;
            this->refcount = another.refcount;
            this->data = another.data;
            this->datastart = another.datastart;
            this->dataend = another.dataend;
            this->datalimit = another.datalimit;
            incref();
        }

        void inline swallow_copy(const Matrix& another, const Rect& roi, bool must_be_released)
        {
            if (must_be_released)
                decref();
            this->cols = roi.width;
            this->rows = roi.height;
            this->step_info = another.step_info;
            this->type_info = another.type_info;
            this->refcount = another.refcount;
            this->data = (uchar*)another.ptr(roi.y, roi.x);
            this->datastart = another.datastart;
            this->dataend = another.dataend;
            this->datalimit = another.datalimit;
            incref();
        }

    public:
        Matrix() noexcept
        {
            init();
        }

        ~Matrix() noexcept
        {
            decref();
        }

        Matrix(const Matrix& another)
        {
            // Copying with initializer
            swallow_copy(another, false);
        }

        Matrix(const Matrix& another, const Rect& roi)
        {
            // Copying with ROI with initializer
            swallow_copy(another, roi, false);
        }

        explicit Matrix(int rows, int cols, int type)
        {
            init();
            create(rows, cols, type);
        }

        explicit Matrix(int rows, int cols, int type, const Scalar& s)
        {
            // We only support scalar initialization with a single value
            init();
            create(rows, cols, type);
            memset(ptr(), s[0], rows * step_info.linestep);
        }

        Matrix(int rows, int cols, const std::string& channel_string)
        {
            init();
            create(rows, cols, channel_string);
        }

        explicit Matrix(int cols, int rows, int channels, int type_wo_channels)
        {
            init();
            create(cols, rows, channels, type_wo_channels);
        }

        Matrix(int rows, int cols, int channels, const std::string& dtype)
        {
            init();
            create(rows, cols, channels, dtype);
        }

    public:
        Matrix& operator=(const Matrix& another)
        {
            // Copying by '=' operator
            swallow_copy(another, true);
            return *this;
        }

        Matrix operator()(const Rect& roi) const
        {
            // Copying with ROI by '()' operator
            Matrix sub_matrix;
            sub_matrix.swallow_copy(*this, roi, false);
            return sub_matrix;
        }

        template<typename Element>
        Matrix& operator=(const Element& value)
        {
            // setTo by '=' operator
            setTo(value);
            return *this;
        }

    private:
        // ONLY USES FOR CREATE MATRIX IN THE CLASS
        void create(int rows, int cols, const MatrixType& _type_info)
        {
            // Check type info, avoid channel 0
            MatrixType type_info = _type_info;
            if (type_info.packed.fields.channels == 0)
                type_info.packed.fields.channels = 1;

            // Calculate size
            int pixel_bytes = (int)type_info.bbp() / 8;
            int scanline_bytes = cols * pixel_bytes;

            // Allocate memory first
            uchar* datastart;
            uchar* dataend;
            if ((datastart = (uchar*)malloc(rows * scanline_bytes)) == NULL)
                throw std::bad_exception();
            dataend = datastart + (rows * scanline_bytes);

            // Update attributes
            decref();
            this->cols = cols;
            this->rows = rows;
            this->step_info.pixelstep = pixel_bytes;
            this->step_info.linestep = scanline_bytes;
            this->type_info = type_info;
            this->datastart = datastart;
            this->dataend = dataend;
            this->datalimit = datalimit;
            this->data = datastart;
            incref();
        }

    public:
        static Matrix zeros(int rows, int cols, int type)
        {
            Matrix m(rows, cols, type);
            memset(m.ptr(), 0, m.rows * m.step_info.linestep);
            return m;
        }

        static Matrix zeros(int rows, int cols, const std::string& channel_string)
        {
            Matrix m(rows, cols, channel_string);
            memset(m.ptr(), 0, m.rows * m.step_info.linestep);
            return m;
        }

        static Matrix zeros(int cols, int rows, int channels, int type_wo_channels)
        {
            Matrix m(cols, rows, channels, type_wo_channels);
            memset(m.ptr(), 0, m.rows * m.step_info.linestep);
            return m;
        }

        static Matrix zeros(int rows, int cols, int channels, const std::string& dtype)
        {
            Matrix m(rows, cols, channels, dtype);
            memset(m.ptr(), 0, m.rows * m.step_info.linestep);
            return m;
        }

    public:
        void create(int rows, int cols, int type)
        {
            // Create matrix by type
            create(rows, cols, MatrixType(type));
        }

        void create(int rows, int cols, const std::string& channel_string)
        {
            // Create matrix by channel string
            create(rows, cols, MatrixType(channel_string));
        }

        void create(int rows, int cols, int channels, int type_wo_channels)
        {
            // Create matrix by type with channels
            MatrixType mt(type_wo_channels);
            mt.packed.fields.channels = channels;
            create(rows, cols, mt);
        }

        void create(int rows, int cols, int channels, const std::string& dtype)
        {
            // Create matrix by dtype
            create(rows, cols, MatrixType(channels, dtype));
        }

        template<typename Element>
        void setTo(const Element& value)
        {
            // The size of value and matrix's element size must be same
            assert(sizeof(value) == elemSize());

            // Set all elements to single value
            for (int i = 0; i < cols * rows; ++i)
                ptr<Element>()[i] = value;
        }

        void copyTo(Matrix matrix) const
        {
            matrix.data_copy(*this);
        }

        Matrix clone() const
        {
            Matrix m;
            m.deep_copy(*this);
            return m;
        }

    public:
        bool empty() const
        {
            return data == NULL;
        }

        size_t elemSize() const
        {
            return (size_t)step_info.pixelstep;
        }

        size_t elemSize1() const
        {
            return (size_t)step_info.pixelstep / type_info.channels();
        }

        int width() const
        {
            return cols;
        }

        int height() const
        {
            return rows;
        }

        std::string dtype() const
        {
            return type_info.dtype();
        }

        int type() const
        {
            return type_info.packed.value;
        }

        int depth() const
        {
            return type_info.depth();
        }

        int channels() const
        {
            return type_info.channels();
        }

        bool isSubmatrix() const
        {
            // Is it correct?
            // Why OpenCV uses that by using (cv::Mat::flags & SUBMATRIX_FLAG)?
            return data != datastart;
        }

    public:
        uchar* ptr(int y=0)
        {
            return data + (step_info.linestep * y);
        }

        const uchar* ptr(int y=0) const
        {
            return data + (step_info.linestep * y);
        }

        uchar* ptr(int y, int x)
        {
            return data + (step_info.linestep * y) + (step_info.pixelstep * x);
        }

        const uchar* ptr(int y, int x) const
        {
            return data + (step_info.linestep * y) + (step_info.pixelstep * x);
        }

        template<typename Element>
        Element* ptr(int y=0)
        {
            return (Element*)ptr(y);
        }

        template<typename Element>
        const Element* ptr(int y=0) const
        {
            return (Element*)ptr(y);
        }

        template<typename Element>
        Element* ptr(int y, int x)
        {
            return (Element*)ptr(y, x);
        }

        template<typename Element>
        const Element* ptr(int y, int x) const
        {
            return (Element*)ptr(y, x);
        }

    public:
        template<typename Element>
        Element& at(int y)
        {
            return *ptr<Element>(y);
        }

        template<typename Element>
        const Element& at(int y) const
        {
            return *ptr<Element>(y);
        }

        template<typename Element>
        Element& at(int y, int x)
        {
            return *ptr<Element>(y, x);
        }

        template<typename Element>
        const Element& at(int y, int x) const
        {
            return *ptr<Element>(y, x);
        }

        template<typename Element>
        Element& at(Point pt)
        {
            return at<Element>(pt.y, pt.x);
        }

        template<typename Element>
        const Element& at(Point pt) const
        {
            return at<Element>(pt.y, pt.x);
        }

    public:
        // OpenCV-compatible rowRange: [startrow, endrow), share data (no copy)
        Matrix rowRange(int startrow, int endrow) const
        {
            assert(0 <= startrow && startrow <= endrow && endrow <= rows);
            Rect roi(0, startrow, cols, endrow - startrow);
            return (*this)(roi);
        }

        // OpenCV-compatible colRange: [startcol, endcol), share data (no copy)
        Matrix colRange(int startcol, int endcol) const
        {
            assert(0 <= startcol && startcol <= endcol && endcol <= cols);
            Rect roi(startcol, 0, endcol - startcol, rows);
            return (*this)(roi);
        }

        void convertTo(Matrix& dst, const MatrixType& type_info, double alpha = 1.0f, double beta = 0.0f) const
        {
            Matrix output(rows, cols, type_info.packed.value);

            int src_depth = depth();
            int dst_depth = type_info.depth();
            int src_channels = channels();
            int dst_channels = type_info.channels();

            // Ensure channel count matches
            assert(src_channels == dst_channels);

            LCV_OMP_LOOP_FOR
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    for (int c = 0; c < src_channels; c++)
                    {
                        double src_val = 0.0f;

                        // Read value from source matrix
                        if (src_depth == LCV_8U)
                            src_val = ptr<uchar>(y, x)[c];
                        else if (src_depth == LCV_8S)
                            src_val = ptr<schar>(y, x)[c];
                        else if (src_depth == LCV_16U)
                            src_val = ptr<ushort>(y, x)[c];
                        else if (src_depth == LCV_16S)
                            src_val = ptr<short>(y, x)[c];
                        else if (src_depth == LCV_32U)
                            src_val = ptr<uint>(y, x)[c];
                        else if (src_depth == LCV_32S)
                            src_val = ptr<int>(y, x)[c];
                        else if (src_depth == LCV_32F)
                            src_val = ptr<float32>(y, x)[c];
                        else if (src_depth == LCV_64F)
                            src_val = ptr<float64>(y, x)[c];

                        // Apply scale and offset
                        double converted_val = src_val * alpha + beta;

                        // Write to destination matrix, using saturate cast
                        if (dst_depth == LCV_8U)
                            output.ptr<uchar>(y, x)[c] = saturate_cast<uchar>(converted_val);
                        else if (dst_depth == LCV_8S)
                            output.ptr<schar>(y, x)[c] = saturate_cast<schar>(converted_val);
                        else if (dst_depth == LCV_16U)
                            output.ptr<ushort>(y, x)[c] = saturate_cast<ushort>(converted_val);
                        else if (dst_depth == LCV_16S)
                            output.ptr<short>(y, x)[c] = saturate_cast<short>(converted_val);
                        else if (dst_depth == LCV_32U)
                            output.ptr<uint>(y, x)[c] = saturate_cast<uint>(converted_val);
                        else if (dst_depth == LCV_32S)
                            output.ptr<int>(y, x)[c] = saturate_cast<int>(converted_val);
                        else if (dst_depth == LCV_32F)
                            output.ptr<float32>(y, x)[c] = saturate_cast<float32>(converted_val);
                        else if (dst_depth == LCV_64F)
                            output.ptr<float64>(y, x)[c] = saturate_cast<float64>(converted_val);
                    }
                }
            }

            dst = output;
        }
    }; // class Matrix

    using Mat = Matrix;
} // namespace lcv
#endif // LCV_CORE_MATRIX_HPP
