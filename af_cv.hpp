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
#ifndef _af_cv_h_
#define _af_cv_h_

#include <stdio.h>
#include <arrayfire.h>
#include <af/utils.h>
#include <opencv2/opencv.hpp>

using namespace af;
using namespace cv;


// conversion for gpu
void mat_to_array(const cv::Mat& input, af::array& output, bool transpose = true) ;
void mat_to_array(const vector<Mat>& input, af::array& output, bool transpose = true);
af::array mat_to_array(const cv::Mat& input, bool transpose = true) ;


// conversion for cpu
void array_to_mat(const af::array& input, cv::Mat& output, int type = CV_32F, bool transpose = true) ;
Mat array_to_mat(const af::array& input, int type = CV_32F, bool transpose = true) ;


// get type of a Mat string
std::string get_mat_type(int input);
std::string get_mat_type(const Mat& input);


// visualize af array in opencv
void imshow(const char* name, const af::array& in) ;


// Mat stats
#define mtop(exp) _mtop(#exp, exp)
void _mtop(const char* exp, Mat X);
#define mstats(exp) _mstats(#exp, exp)
void _mstats(const char* exp, Mat X);


// af zeros
#define zero(...) af::constant(0,##__VA_ARGS__);


#endif //_af_cv_h_
