/*
  Copyright (c) 2012, Chris McClanahan
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  * Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 af_cv: Misc ArrayFire 2.0 OpenCV utility fuctions
 */
#include "af_cv.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>


// mem layout for gpu
void mat_to_array_(cv::Mat& input, array& output, bool transpose = true) {
    const unsigned w = input.cols;
    const unsigned h = input.rows;
    const unsigned channels = input.channels();
    if (channels > 3) { throw std::runtime_error(std::string("mat to array error ")); }
    if (channels == 1) {
        // bw
        if (transpose) {
            output = array(w, h, input.ptr<float>(0)).T();
        } else {
            output = array(h, w, input.ptr<float>(0));
        }
    } else {
        if (w > 2 && h > 2) {
            // 2,3 channel image
            array input_ = array(w * channels, h, input.ptr<float>(0));
            array ind = array(seq(channels - 1, channels, w * channels - 1));
            if (transpose) {
                output = array(h, w, channels);
                gfor(array k, channels) {
                    output(span, span, k) = input_(ind - k, span).T();
                }
            } else {
                output = array(w, h, channels);
                gfor(array k, channels) {
                    output(span, span, k) = input_(ind - k, span);
                }
            }
        } else {
            if (channels == 3) {
                // 3 channels
                std::vector<Mat> rgb; split(input, rgb);
                if (transpose) {
                    output = array(h, w, 3);
                    output(span, span, 0) = array(w, h, rgb[2].ptr<float>(0)).T();
                    output(span, span, 1) = array(w, h, rgb[1].ptr<float>(0)).T();
                    output(span, span, 2) = array(w, h, rgb[0].ptr<float>(0)).T();
                } else {
                    output = array(w, h, 3);
                    output(span, span, 0) = array(w, h, rgb[2].ptr<float>(0));
                    output(span, span, 1) = array(w, h, rgb[1].ptr<float>(0));
                    output(span, span, 2) = array(w, h, rgb[0].ptr<float>(0));
                }
            } else {
                // 2 channels
                std::vector<Mat> gb; split(input, gb);
                if (transpose) {
                    output = array(h, w, 2);
                    output(span, span, 0) = array(w, h, gb[1].ptr<float>(0)).T();
                    output(span, span, 1) = array(w, h, gb[0].ptr<float>(0)).T();
                } else {
                    output = array(w, h, 2);
                    output(span, span, 0) = array(w, h, gb[1].ptr<float>(0));
                    output(span, span, 1) = array(w, h, gb[0].ptr<float>(0));
                }
            }
        }
    }
}

void mat_to_array(const std::vector<Mat>& input, array& output, bool transpose) {
    try {
        int d = input[0].dims;
        int h = input[0].rows;
        int w = input[0].cols;
        if (transpose) {
            output = array(h, w, input.size());
        } else {
            output = array(w, h, input.size());
        }
        for (unsigned i = 0; i < input.size(); i++) {
            array tmp = mat_to_array(input[i], transpose);
            switch (d) {
            case 1: output(span, i) = tmp;
                break;
            case 2: output(span, span, i) = tmp;
                break;
            default: throw std::runtime_error(std::string("Only 1 and 2 dimensions supported"));
            }
        }
    } catch (const af::exception& ex) {
        throw std::runtime_error(std::string("mat to array error "));
    }
}

void mat_to_array(const cv::Mat& input, array& output, bool transpose) {
    if (input.empty()) { return; }
    cv::Mat tmp;
    if (input.channels() == 1)
        { input.convertTo(tmp, CV_32F); }
    else if (input.channels() == 2)
        { input.convertTo(tmp, CV_32FC2); }
    else if (input.channels() == 3)
        { input.convertTo(tmp, CV_32FC3); }
    mat_to_array_(tmp, output, transpose);
}

array mat_to_array(const cv::Mat& input, bool transpose) {
    array output;
    if (input.empty()) { return output; }
    cv::Mat mtmp;
    if (input.channels() == 1)
        { input.convertTo(mtmp, CV_32F); }
    else if (input.channels() == 2)
        { input.convertTo(mtmp, CV_32FC2); }
    else if (input.channels() == 3)
        { input.convertTo(mtmp, CV_32FC3); }
    mat_to_array_(mtmp, output, transpose);
    return output;
}


// mem layout for cpu
void array_to_mat(const array& input_, cv::Mat& output, int type, bool transpose) {
    const int channels = input_.dims(2);
    int ndims = input_.numdims();
    array input;
    if (transpose) {
        if (channels == 1) { input = input_.T(); }
        else {
            input = zero(channels, input_.dims(1), input_.dims(0));
            gfor(array ii, channels) {
                input(channels - ii - 1, span, span) = input_(span, span, ii).T();
            }
        }
    } else {
        input = input_;
    }
    if(ndims == 1) {
        output = cv::Mat(input.dims(ndims - 1), input.elements(), CV_MAKETYPE(type, channels));
    } else {
        output = cv::Mat(input.dims(ndims - 1), input.dims(ndims - 2), CV_MAKETYPE(type, channels));
    }
    if (type == CV_32F) {
        float* data = output.ptr<float>(0);
        input.host((void*)data);
    } else if (type == CV_32S) {
        int* data = output.ptr<int>(0);
        input.as(s32).host((void*)data);
    } else if (type == CV_64F) {
        double* data = output.ptr<double>(0);
        input.as(f64).host((void*)data);
    } else if (type == CV_8U) {
        uchar* data = output.ptr<uchar>(0);
        input.as(b8).host((void*)data);
    } else {
        throw std::runtime_error(std::string("array to mat error "));
    }
}

Mat array_to_mat(const array& input, int type, bool transpose) {
    cv::Mat output;
    array_to_mat(input, output, type, transpose);
    return output;
}


// Mat type info
std::string get_mat_type(const Mat& input) { return get_mat_type(input.type()); }
std::string get_mat_type(int input) {
    int numImgTypes = 35; // 7 base types, with five channel options each (none or C1, ..., C4)
    int enum_ints[] =       {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
                             CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
                             CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
                             CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                             CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
                             CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
                             CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4
                            };
    std::string enum_strings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
                             "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
                             "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                             "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                             "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                             "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                             "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"
                            };
    for (int i = 0; i < numImgTypes; i++) {
        if (input == enum_ints[i]) { return enum_strings[i]; }
    }
    return "unknown image type";
}


// af-cv vis (grayscale only)
void imshow(const char* name, const array& in_) {
    array in;
    if (in_.dims(2) == 3) { in = colorSpace(in_, AF_GRAY, AF_RGB); }
    else if (in.dims(2) == 1) { in = in_; }
    else { throw std::runtime_error(std::string("imshow error ")); }
    Mat tmp = array_to_mat(in);
    imshow(name, tmp); waitKey(5);
}
