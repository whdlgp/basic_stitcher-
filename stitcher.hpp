#ifndef STITCHER_HPP
#define STITCHER_HPP

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#define STITCHER_DEBUG_PRINT

#ifdef STITCHER_DEBUG_PRINT
#define STICHER_DBG_ERR(x) (std::cerr << x << std::endl)
#define STICHER_DBG_OUT(x) (std::cout << x << std::endl)
#else
#define STICHER_DBG_ERR(x)
#define STICHER_DBG_OUT(x)
#endif

class Basic_stitcher
{
    public:
    Basic_stitcher(bool use_cuda = false)
    {
        set_megapix(0.6, 0.1, -1);
        match_conf = 0.3;

        if(!use_cuda)
        {
            finder          = cv::makePtr<cv::detail::OrbFeaturesFinder>();;
            matcher         = cv::makePtr<cv::detail::BestOf2NearestMatcher>(false, 0.3f);;
            estimator       = cv::makePtr<cv::detail::HomographyBasedEstimator>();
            adjuster        = cv::makePtr<cv::detail::BundleAdjusterRay>();
            warper_creator  = cv::makePtr<cv::SphericalWarper>();
            seam_finder     = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
            compensator     = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN_BLOCKS);
            blender         = cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND, false);
        }
        else
        {
            finder          = cv::makePtr<cv::detail::SurfFeaturesFinderGpu>();;
            matcher         = cv::makePtr<cv::detail::BestOf2NearestMatcher>(true, 0.3f);;
            estimator       = cv::makePtr<cv::detail::HomographyBasedEstimator>();
            adjuster        = cv::makePtr<cv::detail::BundleAdjusterRay>();
#ifdef HAVE_OPENCV_CUDAWARPING
            warper_creator  = cv::makePtr<cv::SphericalWarperGpu>();
#else
            warper_creator  = cv::makePtr<cv::SphericalWarper>();
#endif
#ifdef HAVE_OPENCV_CUDALEGACY
            seam_finder     = cv::makePtr<cv::detail::GraphCutSeamFinderGpu>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
#else
            seam_finder     = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
#endif
            compensator     = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN_BLOCKS);
            blender         = cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND, true);
        }
    }

    void                                    set_megapix             (double work_megapix_ = 0.6, double seam_megapix_ = 0.1, double compose_megapix_ = -1);
    void                                    update_image_scale      (std::vector<cv::Mat> &full_img);
    std::vector<cv::detail::ImageFeatures>  finding_features        (const std::vector<cv::Mat> &imgs);
    std::vector<cv::detail::MatchesInfo>    pairwise_matching       (const std::vector<cv::detail::ImageFeatures> &features);
    std::vector<cv::detail::CameraParams>   estimate_camera_params  (const std::vector<cv::detail::ImageFeatures> &features
                                                                    , const std::vector<cv::detail::MatchesInfo> &pairwise_matches);
    void                                    warping_for_prepare_composition(const std::vector<cv::Mat> &images
                                                                    , const std::vector<cv::detail::CameraParams> &cameras
                                                                    , std::vector<cv::Point> &corners_out
                                                                    , std::vector<cv::UMat> &warped_out
                                                                    , std::vector<cv::UMat> &warped_mask_out
                                                                    , std::vector<cv::Rect> &rois_out);
    void                                    warping_for_composition (const std::vector<cv::Mat> &images
                                                                    , const std::vector<cv::detail::CameraParams> &cameras
                                                                    , std::vector<cv::Point> &corners_out
                                                                    , std::vector<cv::UMat> &warped_out
                                                                    , std::vector<cv::UMat> &warped_mask_out
                                                                    , std::vector<cv::Rect> &rois_out);
    void                                    finding_seam            (std::vector<cv::Point> &corners
                                                                    , std::vector<cv::UMat> &warped
                                                                    , std::vector<cv::UMat> &warped_mask);
    void                                    feeding_exposure_compensator(std::vector<cv::Point> &corners
                                                                    , std::vector<cv::UMat> &images_warped
                                                                    , std::vector<cv::UMat> &masks_warped);
    void                                    applying_exposure_compensator(std::vector<cv::Point> &corners
                                                                    , std::vector<cv::UMat> &images_warped
                                                                    , std::vector<cv::UMat> &masks_warped);
    cv::Mat                                 blending                (std::vector<cv::UMat> &image_warped
                                                                    , std::vector<cv::Rect> &rois
                                                                    , std::vector<cv::UMat> &seam_masks);
    
    void                                    calculate_camera_params (std::vector<cv::Mat> &full_img);
    std::vector<cv::detail::CameraParams>   get_camera_params       ();
    void                                    set_camera_params       (std::vector<cv::detail::CameraParams> &cameras_);

    void                                    prepare_compose         (std::vector<cv::Mat> &full_img);

    void                                    compose                 (std::vector<cv::Mat> &full_img);

    cv::Mat                                 stitcher_do_all         (std::vector<cv::Mat> &imgs);

    private:
    cv::Ptr<cv::detail::FeaturesFinder> finder;
    cv::Ptr<cv::detail::FeaturesMatcher> matcher;
    cv::Ptr<cv::detail::Estimator> estimator;
    cv::Ptr<cv::detail::BundleAdjusterBase> adjuster;
    cv::Ptr<cv::WarperCreator> warper_creator;
    cv::Ptr<cv::detail::SeamFinder> seam_finder;
    cv::Ptr<cv::detail::ExposureCompensator> compensator;
    cv::Ptr<cv::detail::Blender> blender;

    std::vector<cv::detail::ImageFeatures> features;
    std::vector<cv::detail::MatchesInfo> pairwise_matches;
    std::vector<cv::detail::CameraParams> cameras;
    std::vector<cv::Point> corners_prepare;
    std::vector<cv::UMat> warped_prepare;
    std::vector<cv::UMat> warped_mask_prepare;
    std::vector<cv::Rect> rois_prepare;
    std::vector<cv::Point> corners_compose;
    std::vector<cv::UMat> warped_compose;
    std::vector<cv::UMat> warped_mask_compose;
    std::vector<cv::Rect> rois_compose;
    cv::Mat result;

    double work_megapix;
    double seam_megapix;
    double compose_megapix;
    double match_conf;

    std::vector<double> work_scale;
    std::vector<double> seam_scale;
    std::vector<double> seam_work_aspect;
    std::vector<double> compose_scale;
    std::vector<double> compose_work_aspect;
};

#endif
