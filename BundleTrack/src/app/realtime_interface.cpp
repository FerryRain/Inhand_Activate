/********************************************************************************
@FileName：realtime_interface.cpp
@Description：BundleTrack realtime RPC interface via ZMQ (old cppzmq compatible)
********************************************************************************/

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include "Bundler.h"
#include "DataLoader.h"

using json = nlohmann::json;

// ---------------- utils ----------------
template <typename T>
static inline T clamp_t(T v, T lo, T hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static std::string msg_to_string(const zmq::message_t& m)
{
    return std::string(reinterpret_cast<const char*>(m.data()), m.size());
}

static Eigen::Matrix4f json_to_mat4(const json& j)
{
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            T(r, c) = j[r * 4 + c].get<float>();
    return T;
}

static json mat4_to_json(const Eigen::Matrix4f& T)
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
        bool ok = sock.recv(&part);   // old cppzmq API
        if (!ok) return false;

        parts.emplace_back(std::move(part));

        int more = 0;
        size_t more_size = sizeof(more);
        sock.getsockopt(ZMQ_RCVMORE, &more, &more_size);
        if (!more) break;
    }
    return true;
}

static bool check_part_bytes(const zmq::message_t& m, size_t expected, const std::string& name,
                             std::string& err)
{
    if (m.size() != expected) {
        err = name + " bytes mismatch: got=" + std::to_string(m.size()) +
              " expected=" + std::to_string(expected);
        return false;
    }
    return true;
}

// --- ROI: use largest connected component in mask ---
static Eigen::Vector4f roi_from_mask_largest_cc(const cv::Mat& mask_in, int pad, int H, int W,
                                                int min_area = 50)
{
    // roi format: [x0, x1, y0, y1]
    if (mask_in.empty()) {
        return Eigen::Vector4f(0.f, (float)(W - 1), 0.f, (float)(H - 1));
    }

    cv::Mat m;
    if (mask_in.type() == CV_8UC1) m = mask_in;
    else mask_in.convertTo(m, CV_8UC1);

    // binary mask (0/255)
    cv::Mat bin;
    cv::compare(m, 0, bin, cv::CMP_GT);

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
    if (n <= 1) {
        return Eigen::Vector4f(0.f, (float)(W - 1), 0.f, (float)(H - 1));
    }

    int best = -1;
    int best_area = 0;
    for (int i = 1; i < n; ++i) { // 0 is background
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > best_area) {
            best_area = area;
            best = i;
        }
    }

    if (best < 0 || best_area < min_area) {
        return Eigen::Vector4f(0.f, (float)(W - 1), 0.f, (float)(H - 1));
    }

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

// --- Depth: match your Utils::readDepthImage behavior ---
// Input: uint16(mm) or float32(m)
// Output: depth_raw/depth/depth_sim all CV_32FC1 in meters, with <0.1m -> 0
static void make_depth_like_readDepthImage(
    int H, int W,
    const zmq::message_t& depth_part,
    const std::string& depth_type,
    float depth_scale,     // 1000 for mm->m
    cv::Mat& depth_raw,    // OUT CV_32FC1 (m), <0.1 -> 0
    cv::Mat& depth,        // OUT CV_32FC1 (m), <0.1 -> 0
    cv::Mat& depth_sim     // OUT CV_32FC1 (m)
)
{
    cv::Mat depth_m;

    if (depth_type == "float32") {
        cv::Mat d_view(H, W, CV_32FC1, (void*)depth_part.data());
        depth_m = d_view.clone(); // meters
    } else {
        cv::Mat d_view(H, W, CV_16UC1, (void*)depth_part.data());
        cv::Mat d_u16 = d_view.clone();
        d_u16.convertTo(depth_m, CV_32FC1, 1.0f / depth_scale); // mm -> m
    }

    // emulate: if depth < 0.1 => 0.0
    depth_raw = depth_m.clone();
    {
        cv::Mat invalid;
        cv::compare(depth_raw, 0.1f, invalid, cv::CMP_LT);
        depth_raw.setTo(0.0f, invalid);
    }

    // NOCS loader does: depth_sim = depth_raw.clone(); depth = readDepthImage(...) again
    depth_sim = depth_raw.clone();
    depth     = depth_raw.clone();
}

static void send_err(zmq::socket_t& sock, const std::string& msg)
{
    json r;
    r["ok"] = false;
    r["err"] = msg;
    std::string out = r.dump();
    sock.send(out.c_str(), out.size());
}


// ---------------- debug prints (optional) ----------------
static std::string cv_type_str(int t)
{
    switch (t) {
    case CV_8UC1:  return "CV_8UC1";
    case CV_8UC3:  return "CV_8UC3";
    case CV_8UC4:  return "CV_8UC4";
    case CV_16UC1: return "CV_16UC1";
    case CV_32FC1: return "CV_32FC1";
    default:       return "CV_TYPE_" + std::to_string(t);
    }
}

static void print_mat3(const Eigen::Matrix3f& K, const std::string& name)
{
    std::cout << name << "=\n" << K << "\n";
}

static void print_mat4(const Eigen::Matrix4f& T, const std::string& name)
{
    std::cout << name << "=\n" << T << "\n";
}

static void print_roi(const Eigen::Vector4f& roi, const std::string& name)
{
    std::cout << name << "=[x0,x1,y0,y1]=["
              << roi[0] << "," << roi[1] << "," << roi[2] << "," << roi[3] << "]\n";
}

static void print_mask_stats(const cv::Mat& mask, const std::string& name)
{
    if (mask.empty()) {
        std::cout << name << ": empty\n";
        return;
    }
    cv::Mat m;
    if (mask.type() == CV_8UC1) m = mask;
    else mask.convertTo(m, CV_8UC1);

    int nz = cv::countNonZero(m);
    double minv=0, maxv=0;
    cv::minMaxLoc(m, &minv, &maxv);
    std::cout << name << ": " << cv_type_str(mask.type())
              << " size=" << mask.cols << "x" << mask.rows
              << " nonzero=" << nz
              << " min=" << minv << " max=" << maxv << "\n";
}

static void print_depth_stats(const cv::Mat& depth_raw, const cv::Mat& depth, const std::string& name)
{
    auto stat_one = [](const cv::Mat& d, const std::string& tag){
        if (d.empty()) {
            std::cout << "  " << tag << ": empty\n";
            return;
        }
        double minv=0, maxv=0;
        cv::minMaxLoc(d, &minv, &maxv);
        std::cout << "  " << tag << ": " << cv_type_str(d.type())
                  << " size=" << d.cols << "x" << d.rows
                  << " min=" << minv << " max=" << maxv << "\n";
    };

    std::cout << name << ":\n";
    stat_one(depth_raw, "depth_raw");
    stat_one(depth,     "depth");
}

static void print_req_header(const json& req)
{
    std::cout << "header keys: ";
    for (auto it = req.begin(); it != req.end(); ++it) std::cout << it.key() << " ";
    std::cout << "\n";

    auto print_if = [&](const char* k){
        if (req.contains(k)) std::cout << "  " << k << ": " << req[k] << "\n";
    };

    print_if("id_str");
    print_if("H");
    print_if("W");
    print_if("color_ch");
    print_if("depth_type");
    print_if("has_mask");
    print_if("roi");
    print_if("roi_pad");
    print_if("has_init");
    if (req.contains("K")) std::cout << "  K: " << req["K"] << "\n";
    if (req.contains("ob_in_cam")) std::cout << "  ob_in_cam: (len=" << req["ob_in_cam"].size() << ")\n";
}


// ---------------- main ----------------
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: realtime_interface path/to/config.yml\n";
        return 1;
    }


    std::shared_ptr<YAML::Node> yml(new YAML::Node);
    *yml = YAML::LoadFile(argv[1]);

    const std::string data_dir = (*yml)["data_dir"].as<std::string>();
    std::string _gt_dir;
    _gt_dir = data_dir+"/annotated_poses/";

    Eigen::Matrix3f cam_K;
    Utils::parseMatrixTxt(data_dir+"/cam_K.txt", cam_K);

    Bundler bundler(yml, cam_K);

    // debug controls (can override in yml)
    bool rpc_debug = false;
    int  rpc_debug_first_n = 200;
    int  rpc_debug_every   = 1;
    if ((*yml)["rpc_debug"])          rpc_debug = (*yml)["rpc_debug"].as<bool>();
    if ((*yml)["rpc_debug_first_n"])  rpc_debug_first_n = (*yml)["rpc_debug_first_n"].as<int>();
    if ((*yml)["rpc_debug_every"])    rpc_debug_every = (*yml)["rpc_debug_every"].as<int>();

    std::cout << "[RPC debug] enabled=" << rpc_debug
              << " first_n=" << rpc_debug_first_n
              << " every=" << rpc_debug_every << "\n";

    // depth scale: uint16(mm) -> float32(m)
    float depth_scale = 1000.f;
    if ((*yml)["depth_scale"]) {
        depth_scale = (*yml)["depth_scale"].as<float>();
        if (depth_scale <= 1e-6f) depth_scale = 1000.f;
    }

    // ZMQ REP server
    zmq::context_t ctx(1);
    zmq::socket_t sock(ctx, ZMQ_REP);

    int port = 5550;
    if ((*yml)["rpc_port"]) port = (*yml)["rpc_port"].as<int>();
    std::string addr = "tcp://*:" + std::to_string(port);
    sock.bind(addr.c_str());
    std::cout << "[BundleTrack RPC] listening on " << addr << std::endl;

    int frame_id = 0;

    // warm start pose (match offline tracking behavior)

    Eigen::Matrix4f _ob_in_cam0;
    {
        std::vector<std::string> gt_files;
        Utils::readDirectory(_gt_dir, gt_files);
        assert(gt_files.size()>0);
        Utils::parsePoseTxt(_gt_dir + gt_files[0], _ob_in_cam0);
    };
    Eigen::Matrix4f last_pose_in_model = _ob_in_cam0.inverse();
    bool last_pose_valid = true;


    PointCloudRGBNormal::Ptr _real_model;

    while (true)
    {
        std::vector<zmq::message_t> parts;
        if (!recv_multipart_old(sock, parts)) continue;

        if (parts.size() < 3) {
            send_err(sock, "need at least 3 parts: header,color,depth");
            continue;
        }

        json req;
        try {
            req = json::parse(msg_to_string(parts[0]));
        } catch (...) {
            send_err(sock, "json parse failed");
            continue;
        }

        if (!req.contains("H") || !req.contains("W")) {
            send_err(sock, "header must contain H and W");
            continue;
        }
        int H = req["H"].get<int>();
        int W = req["W"].get<int>();
        if (H <= 0 || W <= 0 || H > 6000 || W > 8000) {
            send_err(sock, "invalid H/W");
            continue;
        }

        int color_ch = 3;
        if (req.contains("color_ch")) color_ch = req["color_ch"].get<int>();
        if (!(color_ch == 3 || color_ch == 4)) {
            send_err(sock, "color_ch must be 3 or 4");
            continue;
        }

        std::string depth_type = "uint16";
        if (req.contains("depth_type")) depth_type = req["depth_type"].get<std::string>();
        if (!(depth_type == "uint16" || depth_type == "float32")) {
            send_err(sock, "depth_type must be uint16 or float32");
            continue;
        }

        bool has_mask = false;
        if (req.contains("has_mask")) has_mask = req["has_mask"].get<bool>();

        // ---------- bytes sanity check ----------
        std::string err;
        const size_t color_bytes = (size_t)H * (size_t)W * (size_t)color_ch;
        if (!check_part_bytes(parts[1], color_bytes, "color", err)) {
            send_err(sock, err);
            continue;
        }

        const size_t depth_bytes = (size_t)H * (size_t)W * ((depth_type == "float32") ? 4 : 2);
        if (!check_part_bytes(parts[2], depth_bytes, "depth", err)) {
            send_err(sock, err);
            continue;
        }

        if (has_mask) {
            if (parts.size() < 4) {
                send_err(sock, "has_mask=true but no mask part");
                continue;
            }
            const size_t mask_bytes = (size_t)H * (size_t)W;
            if (!check_part_bytes(parts[3], mask_bytes, "mask", err)) {
                send_err(sock, err);
                continue;
            }
        }

        // ---------- decode color / mask ----------
        cv::Mat color;  // CV_8UC3
        cv::Mat mask;   // CV_8UC1

        if (color_ch == 4) {
            cv::Mat rgba_view(H, W, CV_8UC4, parts[1].data());
            cv::Mat rgba = rgba_view.clone();
            std::vector<cv::Mat> chs;
            cv::split(rgba, chs);
            mask = chs[3].clone(); // alpha
            cv::cvtColor(rgba, color, cv::COLOR_BGRA2BGR);
        } else {
            cv::Mat bgr_view(H, W, CV_8UC3, parts[1].data());
            color = bgr_view.clone();
        }

        // explicit mask overrides alpha-mask
        if (has_mask) {
            cv::Mat m_view(H, W, CV_8UC1, parts[3].data());
            mask = m_view.clone();
        }

        if (!mask.empty() && mask.type() != CV_8UC1) {
            mask.convertTo(mask, CV_8UC1);
        }

        // ---------- decode depth (match readDepthImage) ----------
        cv::Mat depth_raw; // CV_32FC1 (m), <0.1 -> 0
        cv::Mat depth;     // CV_32FC1 (m), <0.1 -> 0
        cv::Mat depth_sim; // CV_32FC1 (m)
        make_depth_like_readDepthImage(H, W, parts[2], depth_type, depth_scale, depth_raw, depth, depth_sim);

        // ---------- roi (KEEP MASK REGION) ----------
        Eigen::Vector4f roi;
        if (req.contains("roi")) {
            auto r = req["roi"];
            if (r.size() >= 4) {
                float x0 = r[0].get<float>();
                float x1 = r[1].get<float>();
                float y0 = r[2].get<float>();
                float y1 = r[3].get<float>();

                int ix0 = clamp_t((int)std::round(x0), 0, W - 1);
                int ix1 = clamp_t((int)std::round(x1), 0, W - 1);
                int iy0 = clamp_t((int)std::round(y0), 0, H - 1);
                int iy1 = clamp_t((int)std::round(y1), 0, H - 1);

                if (ix1 < ix0) std::swap(ix0, ix1);
                if (iy1 < iy0) std::swap(iy0, iy1);
                roi << (float)ix0, (float)ix1, (float)iy0, (float)iy1;
            } else {
                roi << 0.f, (float)(W - 1), 0.f, (float)(H - 1);
            }
        } else if (!mask.empty()) {
            int pad = 5;
            if (req.contains("roi_pad")) pad = req["roi_pad"].get<int>();
            roi = roi_from_mask_largest_cc(mask, pad, H, W);
        } else {
            roi << 0.f, (float)(W - 1), 0.f, (float)(H - 1);
        }

        // ---------- K ----------
        Eigen::Matrix3f K = cam_K;
        if (req.contains("K")) {
            auto k = req["K"];
            if (k.size() == 9) {
                K << k[0].get<float>(), k[1].get<float>(), k[2].get<float>(),
                     k[3].get<float>(), k[4].get<float>(), k[5].get<float>(),
                     k[6].get<float>(), k[7].get<float>(), k[8].get<float>();
            } else if (k.size() >= 6) {
                // [fx,0,cx, 0,fy,cy]
                K << k[0].get<float>(), 0.f, k[2].get<float>(),
                     0.f, k[4].get<float>(), k[5].get<float>(),
                     0.f, 0.f, 1.f;
            }
        }

        // ---------- init pose (first frame can use provided init), then warm-start ----------
        Eigen::Matrix4f pose_in_model = Eigen::Matrix4f::Identity();
        bool has_init = false;
        if (req.contains("has_init")) has_init = req["has_init"].get<bool>();

        if (has_init && req.contains("ob_in_cam")) {
            Eigen::Matrix4f ob_in_cam = json_to_mat4(req["ob_in_cam"]);
            pose_in_model = ob_in_cam.inverse();
        } else {
            // warm start from last frame (tracking mode)
            if (last_pose_valid) pose_in_model = last_pose_in_model;
        }

        std::string id_str = std::to_string(frame_id);
        if (req.contains("id_str")) id_str = req["id_str"].get<std::string>();

        // ---------- DEBUG PRINT (before processNewFrame) ----------
        if (rpc_debug && frame_id < rpc_debug_first_n && (frame_id % rpc_debug_every == 0)) {
            std::cout << "\n==================== RPC FRAME DEBUG ====================\n";
            std::cout << "frame_id=" << frame_id << " id_str=" << id_str << "\n";
            print_req_header(req);

            std::cout << "parts sizes: header=" << parts[0].size()
                      << " color=" << parts[1].size()
                      << " depth=" << parts[2].size();
            if (parts.size() >= 4) std::cout << " mask=" << parts[3].size();
            std::cout << "\n";

            std::cout << "decoded mats:\n";
            std::cout << "  color: " << cv_type_str(color.type())
                      << " size=" << color.cols << "x" << color.rows << "\n";
            print_mask_stats(mask, "  mask");
            print_depth_stats(depth_raw, depth, "  depth_stats");

            print_roi(roi, "roi");
            print_mat3(K, "K");
            print_mat4(pose_in_model, "pose_in_model");

            int sy = H / 2, sx = W / 2;
            std::cout << "samples (center): ";
            std::cout << "color[BGR]=("
                      << (int)color.at<cv::Vec3b>(sy, sx)[0] << ","
                      << (int)color.at<cv::Vec3b>(sy, sx)[1] << ","
                      << (int)color.at<cv::Vec3b>(sy, sx)[2] << ") ";
            if (!mask.empty()) std::cout << "mask=" << (int)mask.at<uint8_t>(sy, sx) << " ";
            if (!depth.empty()) std::cout << "depth(m)=" << depth.at<float>(sy, sx) << " ";
            std::cout << "\n=========================================================\n" << std::flush;
        }

        // ---------- build Frame ----------
        std::shared_ptr<Frame> frame(new Frame(
            mask, color, depth, depth_raw, depth_sim,
            roi, pose_in_model,
            frame_id, id_str,
            K, yml,
            NULL,
            _real_model
        ));

        // ---------- inference ----------
        try {
            bundler.processNewFrame(frame);
            bundler.saveNewframeResult();
        } catch (const std::exception& e) {
            send_err(sock, std::string("processNewFrame exception: ") + e.what());
            continue;
        } catch (...) {
            send_err(sock, "processNewFrame unknown exception");
            continue;
        }

        // update warm-start pose
        last_pose_in_model = frame->_pose_in_model;
        last_pose_valid = true;

        // output pose: ob_in_cam = pose_in_model^{-1}
        Eigen::Matrix4f ob_in_cam = frame->_pose_in_model.inverse();

        json resp;
        resp["ok"] = true;
        resp["frame_id"] = frame_id;
        resp["id_str"] = id_str;
        resp["used_mask"] = (!mask.empty());
        resp["ob_in_cam"] = mat4_to_json(ob_in_cam);

        std::string out = resp.dump();
        sock.send(out.c_str(), out.size());

        frame_id++;
    }

    return 0;
}
