/*
Authors: Bowen Wen
Contact: wenbowenxjtu@gmail.com
Created in 2021

Copyright (c) Rutgers University, 2021 All rights reserved.

Bowen Wen and Kostas Bekris. "BundleTrack: 6D Pose Tracking for Novel Objects
 without Instance or Category-Level 3D Models."
 In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2021.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Bowen Wen, Kostas Bekris, Rutgers University,
      nor the names of its contributors may be used to
      endorse or promote products derived from this software without
      specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <random>
#include <iostream>
#include <stdint.h>
#include "Bundler.h"
#include "LossGPU.h"
#include <opencv2/opencv.hpp>
// #include <Eigen/Dense>
#include <sstream>
#include <iomanip>


typedef std::pair<int,int> IndexPair;
using namespace std;
using namespace Eigen;

Bundler::Bundler(std::shared_ptr<YAML::Node> yml1, DataLoaderBase *data_loader)
{
  _data_loader = data_loader;
  _K = data_loader->_K;

  yml = yml1;
  _max_iter = (*yml1)["bundle"]["num_iter_outter"].as<int>();

  _fm = std::make_shared<Lfnet>(yml, this);
}

Bundler::Bundler(std::shared_ptr<YAML::Node> yml1, Eigen::Matrix3f c_K)
{
  _K = c_K;

  yml = yml1;
  _max_iter = (*yml1)["bundle"]["num_iter_outter"].as<int>();

  _fm = std::make_shared<Lfnet>(yml, this);
}

static inline bool projectCamPoint(const Eigen::Vector3f& Pc,
                                   const Eigen::Matrix3f& K,
                                   cv::Point2f& uv)
{
  if (Pc.z() <= 1e-6f) return false;
  float fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
  uv.x = fx * (Pc.x() / Pc.z()) + cx;
  uv.y = fy * (Pc.y() / Pc.z()) + cy;
  return true;
}


static inline void drawPose6DOnImage(
    cv::Mat& img,
    const Eigen::Matrix4f& T_obj_in_cam,
    const Eigen::Matrix3f& K,
    float axis_len = 0.05f,
    int thickness = 2,
    cv::Point text_org = cv::Point(5, 60))
{
  Eigen::Matrix3f R = T_obj_in_cam.block<3,3>(0,0);
  Eigen::Vector3f t = T_obj_in_cam.block<3,1>(0,3);

  Eigen::Vector3f O = t;
  Eigen::Vector3f X = t + R * Eigen::Vector3f(axis_len, 0, 0);
  Eigen::Vector3f Y = t + R * Eigen::Vector3f(0, axis_len, 0);
  Eigen::Vector3f Z = t + R * Eigen::Vector3f(0, 0, axis_len);

  cv::Point2f o2, x2, y2, z2;
  if (projectCamPoint(O, K, o2))
  {
    cv::circle(img, o2, 3, cv::Scalar(0,255,255), -1, cv::LINE_AA);
    if (projectCamPoint(X, K, x2)) cv::line(img, o2, x2, cv::Scalar(0,0,255), thickness, cv::LINE_AA);
    if (projectCamPoint(Y, K, y2)) cv::line(img, o2, y2, cv::Scalar(0,255,0), thickness, cv::LINE_AA);
    if (projectCamPoint(Z, K, z2)) cv::line(img, o2, z2, cv::Scalar(255,0,0), thickness, cv::LINE_AA);
  }

  Eigen::Vector3f ypr = R.eulerAngles(2,1,0);
  float yaw   = ypr[0] * 180.0f / float(M_PI);
  float pitch = ypr[1] * 180.0f / float(M_PI);
  float roll  = ypr[2] * 180.0f / float(M_PI);

  std::ostringstream ss1, ss2;
  ss1 << std::fixed << std::setprecision(4)
      << "t = [" << t.x() << ", " << t.y() << ", " << t.z() << "]";
  ss2 << std::fixed << std::setprecision(2)
      << "rpy(deg) = [" << roll << ", " << pitch << ", " << yaw << "]";

  cv::putText(img, ss1.str(), text_org, cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0,255,255), 1, cv::LINE_AA);
  cv::putText(img, ss2.str(), text_org + cv::Point(0, 22), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0,255,255), 1, cv::LINE_AA);
}



static inline void ensure_dir(const std::string& dir)
{
  if (!boost::filesystem::exists(dir))
    boost::filesystem::create_directories(dir);
}


static inline cv::Mat normalize_mask_u8_255(const cv::Mat& in_mask)
{
  if (in_mask.empty()) return cv::Mat();

  cv::Mat m;
  if (in_mask.type() != CV_8U)
    in_mask.convertTo(m, CV_8U);
  else
    m = in_mask;

  double maxv = 0.0;
  cv::minMaxLoc(m, nullptr, &maxv);
  if (maxv <= 1.0) m *= 255;

  return m;
}


void Bundler::saveKeyframeResult(const std::shared_ptr<Frame>& frame)
{
  if (!frame) return;

  const std::string debug_dir = (*yml)["debug_dir"].as<std::string>();
  const std::string base_dir  = debug_dir + "/keyframes/";

  const std::string rgb_dir_full = base_dir + "rgb_full/";
  const std::string rgb_dir_obj  = base_dir + "rgb_obj/";
  const std::string depth_dir    = base_dir + "depth/";
  const std::string mask_dir     = base_dir + "mask/";
  const std::string pose_dir     = base_dir + "poses/";
  const std::string viz_dir      = base_dir + "viz/";

  auto ensure_dir = [](const std::string& d){
    if (!boost::filesystem::exists(d)) boost::filesystem::create_directories(d);
  };
  ensure_dir(rgb_dir_full);
  ensure_dir(rgb_dir_obj);
  ensure_dir(depth_dir);
  ensure_dir(mask_dir);
  ensure_dir(pose_dir);
  ensure_dir(viz_dir);

  // ---------------- pose ----------------
  Eigen::Matrix4f cur_in_model = frame->_pose_in_model;
  Eigen::Matrix4f ob_in_cam    = cur_in_model.inverse();

  {
    std::ofstream ff(pose_dir + frame->_id_str + ".txt");
    ff << std::setprecision(10) << ob_in_cam << std::endl;
    ff.close();
  }

  // ---------------- save rgb_full: use _vis ----------------
  if (!frame->_vis.empty())
  {
    cv::imwrite(rgb_dir_full + frame->_id_str + ".jpg",
                frame->_vis, {CV_IMWRITE_JPEG_QUALITY, 95});
  }

  // ---------------- save rgb_obj: masked object-only  ----------------
  if (!frame->_color.empty())
  {
    cv::imwrite(rgb_dir_obj + frame->_id_str + ".jpg",
                frame->_color, {CV_IMWRITE_JPEG_QUALITY, 95});
  }

  // ---------------- save mask ----------------
  if (!frame->_fg_mask.empty())
  {
    cv::Mat m = frame->_fg_mask;
    if (m.type() != CV_8U) m.convertTo(m, CV_8U);

    double maxv = 0.0;
    cv::minMaxLoc(m, nullptr, &maxv);
    if (maxv <= 1.0) m *= 255;

    cv::imwrite(mask_dir + frame->_id_str + ".png", m);
  }

  // ---------------- save depth ----------------

  if (!frame->_depth.empty())
  {
    const cv::Mat& depth = frame->_depth;
    const std::string out_path = depth_dir + frame->_id_str + ".png";

    cv::Mat d16;

    if (depth.type() == CV_16U || depth.type() == CV_16UC1)
    {
      // 原本就是毫米 uint16，直接存
      d16 = depth;
    }
    else if (depth.type() == CV_32F || depth.type() == CV_32FC1 ||
             depth.type() == CV_64F || depth.type() == CV_64FC1)
    {
      // 这里假设 depth 是“米”，恢复成“毫米”再存 PNG
      cv::Mat d32;
      depth.convertTo(d32, CV_32F);
      d32 = d32 * 1000.0f;          // m -> mm（恢复原始深度的数值范围）
      d32.convertTo(d16, CV_16U);   // 饱和转换
    }
    else
    {
      // 兜底：直接转 16U（不推荐但防崩）
      depth.convertTo(d16, CV_16U);
    }

    cv::imwrite(out_path, d16);
  }


  cv::Mat base;
  if (!frame->_vis.empty()) base = frame->_vis.clone();
  else if (!frame->_color.empty()) base = frame->_color.clone();
  else return;

  if (!frame->_fg_mask.empty() && base.rows == frame->_fg_mask.rows && base.cols == frame->_fg_mask.cols)
  {
    cv::Mat m = frame->_fg_mask;
    if (m.type() != CV_8U) m.convertTo(m, CV_8U);

    double maxv = 0.0;
    cv::minMaxLoc(m, nullptr, &maxv);
    if (maxv <= 1.0) m *= 255;

    cv::Mat dark;
    base.convertTo(dark, -1, 0.2, 0.0);

    cv::Mat inv;
    cv::bitwise_not(m, inv);
    dark.copyTo(base, inv);
  }

  drawPose6DOnImage(base, ob_in_cam, _K, 0.05f, 2, cv::Point(5, 60));
  cv::putText(base, frame->_id_str, {5,30},
              cv::FONT_HERSHEY_PLAIN, 2, {255,0,0}, 1, cv::LINE_AA);

  cv::imwrite(viz_dir + frame->_id_str + "_viz.jpg",
              base, {CV_IMWRITE_JPEG_QUALITY, 90});
}



void Bundler::processNewFrame(std::shared_ptr<Frame> frame)
{
  // std::cout<<"\n\n";
  printf("New frame %s\n",frame->_id_str.c_str());
  _newframe = frame;

  std::thread worker;

  const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+"/"+frame->_id_str+"/";
  if ((*yml)["LOG"].as<int>()>1)
  {
    if (!boost::filesystem::exists(out_dir))
    {
      system(std::string("mkdir -p "+out_dir).c_str());
    }
  }

  std::shared_ptr<Frame> last_frame;
  if (_frames.size()>0)
  {
    last_frame = _frames.back();
    assert(last_frame->_status!=Frame::FAIL);
    frame->_id = last_frame->_id+1;
    frame->_pose_in_model = last_frame->_pose_in_model;
    frame->segmentationByMaskFile();
  }
  else
  {
    frame->segmentationByMaskFile();
  }


  if (frame->_roi(1)-frame->_roi(0)<10 || frame->_roi(3)-frame->_roi(2)<10)
  {
    frame->_status = Frame::FAIL;
    printf("Frame %s cloud is empty, marked FAIL, roi WxH=%fx%f\n", frame->_id_str.c_str(),frame->_roi(1)-frame->_roi(0),frame->_roi(3)-frame->_roi(2));
    return;
  }


  if (frame->_status==Frame::FAIL)
  {
    _fm->forgetFrame(frame);
    _need_reinit = true;
    return;
  }

  try
  {
    float rot_deg = 0;
    Eigen::Matrix4f prev_in_init(Eigen::Matrix4f::Identity());

    _fm->detectFeature(frame, rot_deg);
  }
  catch (const std::exception &e)
  {
    printf("frame marked as FAIL since feature detection failed, ERROR\n");
    frame->_status = Frame::FAIL;
    _need_reinit = true;
    _fm->forgetFrame(frame);
    return;
  }

  if (_frames.size()>0)
  {
    _fm->findCorres(frame, last_frame);

    if (frame->_status==Frame::FAIL)
    {
      _need_reinit = true;
      _fm->forgetFrame(frame);
      return;
    }

    Eigen::Matrix4f model_in_cam = frame->_pose_in_model.inverse();

    Eigen::Matrix4f offset = _fm->procrustesByCorrespondence(frame, last_frame, _fm->_matches[{frame,last_frame}]);
    frame->_pose_in_model = offset * frame->_pose_in_model;
    frame->_pose_inited = true;

  }

  if (frame->_status==Frame::FAIL)
  {
    _fm->forgetFrame(frame);
    _need_reinit = true;
    return;
  }

  assert(frame->_pose_in_model!=Eigen::Matrix4f::Identity());

  const int window_size = (*yml)["bundle"]["window_size"].as<int>();
  if (_frames.size()>=window_size+3)
  {
    if (std::find(_keyframes.begin(),_keyframes.end(),_frames.front())==_keyframes.end())
    {
      _fm->forgetFrame(_frames.front());
    }
    _frames.pop_front();
  }

  _frames.push_back(frame);

  if (frame->_id==0)
  {
    checkAndAddKeyframe(frame);
    return;
  }

  if (frame->_id>=1)
  {
    selectKeyFramesForBA();
    optimizeGPU();
  }

  if (frame->_status==Frame::FAIL)
  {
    _fm->forgetFrame(frame);
    _frames.pop_back();
    _need_reinit = true;
    return;
  }

  checkAndAddKeyframe(frame);

}

void Bundler::checkAndAddKeyframe(std::shared_ptr<Frame> frame)
{
  if (frame->_id==0)
  {
    _keyframes.push_back(frame);
    saveKeyframeResult(frame);
    return;
  }
  if (frame->_status!=Frame::OTHER) return;

  const int min_interval = (*yml)["keyframe"]["min_interval"].as<int>();
  const int min_feat_num = (*yml)["keyframe"]["min_feat_num"].as<int>();
  const float min_rot = (*yml)["keyframe"]["min_rot"].as<float>();


  if (frame->_keypts.size()<min_feat_num)
  {
    return;
  }

  for (int i=0;i<_keyframes.size();i++)
  {
    const auto &k_pose = _keyframes[i]->_pose_in_model;
    const auto &cur_pose = frame->_pose_in_model;
    float rot_diff = Utils::rotationGeodesicDistance(cur_pose.block(0,0,3,3), k_pose.block(0,0,3,3));
    float trans_diff = (cur_pose.block(0,3,3,1)-k_pose.block(0,3,3,1)).norm();
    rot_diff = rot_diff*180/M_PI;
    if (rot_diff<min_rot)
    {
      return;
    }
  }


  _keyframes.push_back(frame);
  saveKeyframeResult(frame);
}


void Bundler::selectKeyFramesForBA()
{
  std::set<std::shared_ptr<Frame>> frames = {_newframe};
  const int max_BA_frames = (*yml)["bundle"]["max_BA_frames"].as<int>();
  // printf("total keyframes=%d, already chosen _local_frames=%d, want to select %d\n", _keyframes.size(), frames.size(), max_BA_frames);
  if (_keyframes.size()+frames.size()<=max_BA_frames)
  {
    for (const auto &kf:_keyframes)
    {
      frames.insert(kf);
    }
    _local_frames = std::vector<std::shared_ptr<Frame>>(frames.begin(),frames.end());
    return;
  }

  frames.insert(_keyframes[0]);

  const std::string method = (*yml)["bundle"]["subset_selection_method"].as<std::string>();

  if (method=="greedy_rot")
  {
    while (frames.size()<max_BA_frames)
    {
      float best_dist = std::numeric_limits<float>::max();
      std::shared_ptr<Frame> best_kf;
      for (int i=0;i<_keyframes.size();i++)
      {
        const auto &kf = _keyframes[i];
        if (frames.find(kf)!=frames.end()) continue;
        float cum_dist = 0;
        for (const auto &f:frames)
        {
          float rot_diff = Utils::rotationGeodesicDistance(kf->_pose_in_model.block(0,0,3,3), f->_pose_in_model.block(0,0,3,3));
          cum_dist += rot_diff;
        }
        if (cum_dist<best_dist)
        {
          best_dist = cum_dist;
          best_kf = kf;
        }
      }
      frames.insert(best_kf);
    }
  }
  else
  {
    std::cout<<"method not exist\n";
    exit(1);
  }

  _local_frames = std::vector<std::shared_ptr<Frame>>(frames.begin(),frames.end());

}




void Bundler::optimizeGPU()
{
  const int num_iter_outter = (*yml)["bundle"]["num_iter_outter"].as<int>();
  const int num_iter_inner = (*yml)["bundle"]["num_iter_inner"].as<int>();
  const int min_fm_edges_newframe = (*yml)["bundle"]["min_fm_edges_newframe"].as<int>();


  std::sort(_local_frames.begin(), _local_frames.end(), FramePtrComparator());
  // printf("#_local_frames=%d\n",_local_frames.size());
  // for (int i=0;i<_local_frames.size();i++)
  // {
  //   std::cout<<_local_frames[i]->_id_str<<" ";
  // }
  // std::cout<<std::endl;

  std::vector<EntryJ> global_corres;
  std::vector<int> n_match_per_pair;
  int n_edges_newframe = 0;

  for (int i=0;i<_local_frames.size();i++)
  {
    for (int j=i+1;j<_local_frames.size();j++)
    {
      const auto &frameA = _local_frames[j];
      const auto &frameB = _local_frames[i];
      _fm->findCorres(frameA, frameB);
      _fm->vizCorresBetween(frameA,frameB,"BA");

      const auto &matches = _fm->_matches[{frameA,frameB}];
      for (int k=0;k<matches.size();k++)
      {
        const auto &match = matches[k];
        EntryJ corres;
        corres.imgIdx_j = j;
        corres.imgIdx_i = i;
        corres.pos_j = make_float3(match._ptA_cam.x,match._ptA_cam.y,match._ptA_cam.z);
        corres.pos_i = make_float3(match._ptB_cam.x,match._ptB_cam.y,match._ptB_cam.z);
        global_corres.push_back(corres);
        if (frameA==_newframe || frameB==_newframe)
        {
          n_edges_newframe++;
        }
      }
      n_match_per_pair.push_back(matches.size());
    }
  }

  const int H = _newframe->_H;
  const int W = _newframe->_W;
  const int n_pixels = H*W;

  std::vector<float*> depths_gpu;
  std::vector<uchar4*> colors_gpu;
  std::vector<float4*> normals_gpu;
  std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> > poses;
  for (int i=0;i<_local_frames.size();i++)
  {
    const auto &f = _local_frames[i];
    depths_gpu.push_back(f->_depth_gpu);
    colors_gpu.push_back(f->_color_gpu);
    normals_gpu.push_back(f->_normal_gpu);
    poses.push_back(f->_pose_in_model);
  }

  if (n_edges_newframe<=min_fm_edges_newframe)
  {
    _newframe->_status = Frame::NO_BA;
    return;
  }

  // printf("OptimizerGPU begin, global_corres#=%d\n",global_corres.size());
  OptimizerGpu opt(yml);
  opt.optimizeFrames(global_corres, n_match_per_pair, _local_frames.size(), _newframe->_H, _newframe->_W, depths_gpu, colors_gpu, normals_gpu, poses, _newframe->_K);

  for (int i=0;i<_local_frames.size();i++)
  {
    const auto &f = _local_frames[i];
    f->_pose_in_model = poses[i];
  }

}




void Bundler::saveNewframeResult1()
{
  const std::string debug_dir = (*yml)["debug_dir"].as<std::string>();
  const std::string out_dir = debug_dir+"/"+_newframe->_id_str+"/";
  const std::string pose_out_dir = debug_dir+"/poses/";

  cout<<endl<<"debug_dir:"<<debug_dir<<endl;
  cout<<"out_dir:"<<out_dir<<endl;
  cout<<"pose_out_dir:"<<pose_out_dir<<endl;


  if (!boost::filesystem::exists(pose_out_dir))
  {
    system(std::string("mkdir -p "+pose_out_dir).c_str());
  }

  Eigen::Matrix4f cur_in_model = _newframe->_pose_in_model;
  Eigen::Matrix4f ob_in_cam = cur_in_model.inverse();

  std::ofstream ff(pose_out_dir+_newframe->_id_str+".txt");
  ff<<std::setprecision(10)<<ob_in_cam<<std::endl;
  ff.close();

  if ((*yml)["LOG"].as<int>()>0)
  {
    cv::Mat color_viz = _newframe->_vis.clone();
    for (int h=0;h<_newframe->_H;h++)
    {
      for (int w=0;w<_newframe->_W;w++)
      {
        auto &bgr = color_viz.at<cv::Vec3b>(h,w);
        if (_newframe->_fg_mask.at<uchar>(h,w)==0)
        {
          for (int i=0;i<3;i++)
          {
            bgr[i] = (uchar)bgr[i]*0.2;
          }
        }
      }
    }
    drawPose6DOnImage(color_viz, ob_in_cam, _K, 0.05f, 2, cv::Point(5, 60));
    cv::putText(color_viz,_newframe->_id_str,{5,30},cv::FONT_HERSHEY_PLAIN,2,{255,0,0},1,8,false);


    cv::imwrite(debug_dir+"/color_viz/"+_newframe->_id_str+"_color_viz.jpg",color_viz,{CV_IMWRITE_JPEG_QUALITY, 80});
    cout<<"here::::::"<<debug_dir+"/color_viz/"+_newframe->_id_str+"_color_viz.jpg"<<endl;
    cv::imwrite(out_dir+"color_viz.jpg",color_viz,{CV_IMWRITE_JPEG_QUALITY, 80});

    const std::string raw_dir = debug_dir+"/color_raw/";
    if (!boost::filesystem::exists(raw_dir))
    {
      system(std::string("mkdir -p "+raw_dir).c_str());
    }
    cv::imwrite(raw_dir+_newframe->_id_str+"_color_raw.png",_newframe->_color);
  }
}

void Bundler::saveNewframeResult()
{
  const std::string debug_dir = (*yml)["debug_dir"].as<std::string>();
  const std::string pose_out_dir = debug_dir + "/poses/";
  const std::string viz_dir = debug_dir + "/color_viz/";
  const std::string raw_dir = debug_dir + "/color_raw/";

  if (!boost::filesystem::exists(pose_out_dir))
    boost::filesystem::create_directories(pose_out_dir);
  if (!boost::filesystem::exists(viz_dir))
    boost::filesystem::create_directories(viz_dir);
  if (!boost::filesystem::exists(raw_dir))
    boost::filesystem::create_directories(raw_dir);

  Eigen::Matrix4f cur_in_model = _newframe->_pose_in_model;
  Eigen::Matrix4f ob_in_cam = cur_in_model.inverse();

  std::ofstream ff(pose_out_dir + _newframe->_id_str + ".txt");
  ff << std::setprecision(10) << ob_in_cam << std::endl;
  ff.close();

  if ((*yml)["LOG"].as<int>() > 0)
  {
    cv::Mat color_viz = _newframe->_vis.clone();


    for (int h = 0; h < _newframe->_H; h++)
      for (int w = 0; w < _newframe->_W; w++)
      {
        auto &bgr = color_viz.at<cv::Vec3b>(h,w);
        if (_newframe->_fg_mask.at<uchar>(h,w) == 0)
          for (int i = 0; i < 3; i++) bgr[i] = (uchar)(bgr[i] * 0.2);
      }


    drawPose6DOnImage(color_viz, ob_in_cam, _K, 0.05f, 2, cv::Point(5, 60));
    cv::putText(color_viz, _newframe->_id_str, {5,30},
                cv::FONT_HERSHEY_PLAIN, 2, {255,0,0}, 1, cv::LINE_AA);

    cv::imwrite(viz_dir + _newframe->_id_str + "_color_viz.jpg",
                color_viz, {CV_IMWRITE_JPEG_QUALITY, 80});
    // cv::imwrite(raw_dir + _newframe->_id_str + "_color_raw.png", _newframe->_color);
  }
}



