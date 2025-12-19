/********************************************************************************
@FileName：realtime_interface.cpp
@Description：BundleTrack realtime RPC interface via ZMQ (old cppzmq compatible)
********************************************************************************/

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include "Bundler.h"
#include "DataLoader.h"

using json = nlohmann::json;

template <typename T>
static inline T clamp_t(T v, T lo, T hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static inline std::string msg_to_string(const zmq::message_t& m)
{
    return std::string(reinterpret_cast<const char*>(m.data()), m.size());
}

static inline Eigen::Matrix4f json_to_mat4(const json& j)
{
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            T(r, c) = j[r * 4 + c].get<float>();
    return T;
}

static inline json mat4_to_json(const Eigen::Matrix4f& T)
{
    json j = json::array();
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            j.push_back((double)T(r, c));
    return j;
}

static bool recv_multipart_old(zmq::socket_t& sock, std::vector<zmq::message_t>& parts)
{
    parts.clear();
    while (true)
    {
        zmq::message_t part;
        bool ok = sock.recv(&part); // old cppzmq API
        if (!ok) return false;

        parts.emplace_back(std::move(part));

        int more = 0;
        size_t more_size = sizeof(more);
        sock.getsockopt(ZMQ_RCVMORE, &more, &more_size);
        if (!more) break;
    }
    return true;
}

static bool check_part_bytes(const zmq::message_t& m, size_t expected, const std::string& name, std::string& err)
{
    if (m.size() != expected)
    {
        err = name + " bytes mismatch: got=" + std::to_string(m.size()) +
              " expected=" + std::to_string(expected);
        return false;
    }
    return true;
}

static Eigen::Vector4f roi_from_mask_largest_cc(const cv::Mat& mask_in, int pad, int H, int W, int min_area = 50)
{
    if (mask_in.empty())
        return Eigen::Vector4f(0.f, (float)(W - 1), 0.f, (float)(H - 1));

    cv::Mat m;
    if (mask_in.type() == CV_8UC1) m = mask_in;
    else mask_in.convertTo(m, CV_8UC1);

    cv::Mat bin;
    cv::compare(m, 0, bin, cv::CMP_GT);

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
    if (n <= 1)
        return Eigen::Vector4f(0.f, (float)(W - 1), 0.f, (float)(H - 1));

    int best = -1;
    int best_area = 0;
    for (int i = 1; i < n; ++i)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > best_area) { best_area = area; best = i; }
    }

    if (best < 0 || best_area < min_area)
        return Eigen::Vector4f(0.f, (float)(W - 1), 0.f, (float)(H - 1));

    int x  = stats.at<int>(best, cv::CC_STAT_LEFT);
    int y  = stats.at<int>(best, cv::CC_STAT_TOP);
    int ww = stats.at<int>(best, cv::CC_STAT_WIDTH);
    int hh = stats.at<int>(best, cv::CC_STAT_HEIGHT);

    int x0 = clamp_t(x - pad, 0, W - 1);
    int y0 = clamp_t(y - pad, 0, H - 1);
    int x1 = clamp_t(x + ww - 1 + pad, 0, W - 1);
    int y1 = clamp_t(y + hh - 1 + pad, 0, H - 1);

    if (x1 < x0) std::swap(x0, x1);
    if (y1 < y0) std::swap(y0, y1);

    return Eigen::Vector4f((float)x0, (float)x1, (float)y0, (float)y1);
}

static void make_depth_like_readDepthImage(
    int H, int W,
    const zmq::message_t& depth_part,
    const std::string& depth_type,
    float depth_scale,
    cv::Mat& depth_raw,
    cv::Mat& depth,
    cv::Mat& depth_sim)
{
    cv::Mat depth_m;

    if (depth_type == "float32")
    {
        cv::Mat d_view(H, W, CV_32FC1, (void*)depth_part.data());
        depth_m = d_view.clone();
    }
    else
    {
        cv::Mat d_view(H, W, CV_16UC1, (void*)depth_part.data());
        cv::Mat d_u16 = d_view.clone();
        d_u16.convertTo(depth_m, CV_32FC1, 1.0f / depth_scale);
    }

    depth_raw = depth_m.clone();
    {
        cv::Mat invalid;
        cv::compare(depth_raw, 0.1f, invalid, cv::CMP_LT);
        depth_raw.setTo(0.0f, invalid);
    }

    depth_sim = depth_raw.clone();
    depth     = depth_raw.clone();
}

static inline void send_err(zmq::socket_t& sock, const std::string& msg)
{
    json r;
    r["ok"] = false;
    r["err"] = msg;
    const std::string out = r.dump();
    sock.send(out.c_str(), out.size());
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: realtime_interface path/to/config.yml\n";
        return 1;
    }

    auto yml = std::make_shared<YAML::Node>(YAML::LoadFile(argv[1]));
    const std::string data_dir = (*yml)["data_dir"].as<std::string>();
    const std::string gt_dir = data_dir + "/annotated_poses/";

    Eigen::Matrix3f cam_K;
    Utils::parseMatrixTxt(data_dir + "/cam_K.txt", cam_K);

    Bundler bundler(yml, cam_K);

    float depth_scale = 1000.f;
    if ((*yml)["depth_scale"])
    {
        depth_scale = (*yml)["depth_scale"].as<float>();
        if (depth_scale <= 1e-6f) depth_scale = 1000.f;
    }

    zmq::context_t ctx(1);
    zmq::socket_t sock(ctx, ZMQ_REP);

    int port = 5550;
    if ((*yml)["rpc_port"]) port = (*yml)["rpc_port"].as<int>();
    const std::string addr = "tcp://*:" + std::to_string(port);
    sock.bind(addr.c_str());
    std::cout << "[BundleTrack RPC] listening on " << addr << std::endl;

    Eigen::Matrix4f ob_in_cam0;
    {
        std::vector<std::string> gt_files;
        Utils::readDirectory(gt_dir, gt_files);
        assert(!gt_files.empty());
        Utils::parsePoseTxt(gt_dir + gt_files[0], ob_in_cam0);
    }
    const Eigen::Matrix4f init_pose_in_model = ob_in_cam0.inverse();

    Eigen::Matrix4f last_pose_in_model = Eigen::Matrix4f::Identity();
    bool last_pose_valid = false;

    PointCloudRGBNormal::Ptr real_model;
    int frame_id = 0;

    while (true)
    {
        std::vector<zmq::message_t> parts;
        if (!recv_multipart_old(sock, parts)) continue;

        if (parts.size() < 3)
        {
            send_err(sock, "need at least 3 parts: header,color,depth");
            continue;
        }

        json req;
        try { req = json::parse(msg_to_string(parts[0])); }
        catch (...) { send_err(sock, "json parse failed"); continue; }

        if (!req.contains("H") || !req.contains("W"))
        {
            send_err(sock, "header must contain H and W");
            continue;
        }

        const int H = req["H"].get<int>();
        const int W = req["W"].get<int>();
        if (H <= 0 || W <= 0 || H > 6000 || W > 8000)
        {
            send_err(sock, "invalid H/W");
            continue;
        }

        int color_ch = req.contains("color_ch") ? req["color_ch"].get<int>() : 3;
        if (!(color_ch == 3 || color_ch == 4))
        {
            send_err(sock, "color_ch must be 3 or 4");
            continue;
        }

        std::string depth_type = req.contains("depth_type") ? req["depth_type"].get<std::string>() : "uint16";
        if (!(depth_type == "uint16" || depth_type == "float32"))
        {
            send_err(sock, "depth_type must be uint16 or float32");
            continue;
        }

        const bool has_mask = req.contains("has_mask") ? req["has_mask"].get<bool>() : false;

        std::string err;
        const size_t color_bytes = (size_t)H * (size_t)W * (size_t)color_ch;
        if (!check_part_bytes(parts[1], color_bytes, "color", err)) { send_err(sock, err); continue; }

        const size_t depth_bytes = (size_t)H * (size_t)W * ((depth_type == "float32") ? 4u : 2u);
        if (!check_part_bytes(parts[2], depth_bytes, "depth", err)) { send_err(sock, err); continue; }

        if (has_mask)
        {
            if (parts.size() < 4) { send_err(sock, "has_mask=true but no mask part"); continue; }
            const size_t mask_bytes = (size_t)H * (size_t)W;
            if (!check_part_bytes(parts[3], mask_bytes, "mask", err)) { send_err(sock, err); continue; }
        }

        cv::Mat color;
        cv::Mat mask;

        if (color_ch == 4)
        {
            cv::Mat rgba_view(H, W, CV_8UC4, parts[1].data());
            cv::Mat rgba = rgba_view.clone();
            std::vector<cv::Mat> chs;
            cv::split(rgba, chs);
            mask = chs[3].clone();
            cv::cvtColor(rgba, color, cv::COLOR_BGRA2BGR);
        }
        else
        {
            cv::Mat bgr_view(H, W, CV_8UC3, parts[1].data());
            color = bgr_view.clone();
        }

        if (has_mask)
        {
            cv::Mat m_view(H, W, CV_8UC1, parts[3].data());
            mask = m_view.clone();
        }

        if (!mask.empty() && mask.type() != CV_8UC1)
            mask.convertTo(mask, CV_8UC1);

        cv::Mat depth_raw, depth, depth_sim;
        make_depth_like_readDepthImage(H, W, parts[2], depth_type, depth_scale, depth_raw, depth, depth_sim);

        Eigen::Vector4f roi;
        if (req.contains("roi"))
        {
            auto r = req["roi"];
            if (r.size() >= 4)
            {
                int x0 = clamp_t((int)std::round(r[0].get<float>()), 0, W - 1);
                int x1 = clamp_t((int)std::round(r[1].get<float>()), 0, W - 1);
                int y0 = clamp_t((int)std::round(r[2].get<float>()), 0, H - 1);
                int y1 = clamp_t((int)std::round(r[3].get<float>()), 0, H - 1);
                if (x1 < x0) std::swap(x0, x1);
                if (y1 < y0) std::swap(y0, y1);
                roi << (float)x0, (float)x1, (float)y0, (float)y1;
            }
            else
            {
                roi << 0.f, (float)(W - 1), 0.f, (float)(H - 1);
            }
        }
        else if (!mask.empty())
        {
            int pad = req.contains("roi_pad") ? req["roi_pad"].get<int>() : 5;
            roi = roi_from_mask_largest_cc(mask, pad, H, W);
        }
        else
        {
            roi << 0.f, (float)(W - 1), 0.f, (float)(H - 1);
        }

        Eigen::Matrix3f K = cam_K;
        if (req.contains("K"))
        {
            auto k = req["K"];
            if (k.size() == 9)
            {
                K << k[0].get<float>(), k[1].get<float>(), k[2].get<float>(),
                     k[3].get<float>(), k[4].get<float>(), k[5].get<float>(),
                     k[6].get<float>(), k[7].get<float>(), k[8].get<float>();
            }
            else if (k.size() >= 6)
            {
                K << k[0].get<float>(), 0.f, k[2].get<float>(),
                     0.f, k[4].get<float>(), k[5].get<float>(),
                     0.f, 0.f, 1.f;
            }
        }

        Eigen::Matrix4f pose_in_model = Eigen::Matrix4f::Identity();
        const bool has_init = req.contains("has_init") ? req["has_init"].get<bool>() : false;

        if (has_init && req.contains("ob_in_cam"))
        {
            const Eigen::Matrix4f ob_in_cam = json_to_mat4(req["ob_in_cam"]);
            pose_in_model = ob_in_cam.inverse();
        }
        else
        {
            pose_in_model = last_pose_valid ? last_pose_in_model : init_pose_in_model;
        }

        std::string id_str = req.contains("id_str") ? req["id_str"].get<std::string>()
                                                    : std::to_string(frame_id);

        std::shared_ptr<Frame> frame(new Frame(
            mask, color, depth, depth_raw, depth_sim,
            roi, pose_in_model,
            frame_id, id_str,
            K, yml,
            NULL,
            real_model
        ));

        try
        {
            bundler.processNewFrame(frame);
            if ((*yml)["LOG"].as<int>() > 0)
            {
                bundler.saveNewframeResult();
            }
        }
        catch (const std::exception& e)
        {
            send_err(sock, std::string("processNewFrame exception: ") + e.what());
            continue;
        }
        catch (...)
        {
            send_err(sock, "processNewFrame unknown exception");
            continue;
        }

        last_pose_in_model = frame->_pose_in_model;
        last_pose_valid = true;

        const Eigen::Matrix4f ob_in_cam = frame->_pose_in_model.inverse();

        json resp;
        resp["ok"] = true;
        resp["frame_id"] = frame_id;
        resp["id_str"] = id_str;
        resp["used_mask"] = (!mask.empty());
        resp["ob_in_cam"] = mat4_to_json(ob_in_cam);

        const std::string out = resp.dump();
        sock.send(out.c_str(), out.size());

        frame_id++;
    }

    return 0;
}
