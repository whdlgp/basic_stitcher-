#include "stitcher.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

void Basic_stitcher::set_megapix (double work_megapix_, double seam_megapix_, double compose_megapix_)
{
    work_megapix = work_megapix_;
    seam_megapix = seam_megapix_;
    compose_megapix = compose_megapix_;
}

void Basic_stitcher::update_image_scale(vector<Mat> &full_img)
{
    for(int i = 0; i < full_img.size(); i++)
    {
        double work_scale_;
        double seam_scale_;
        double seam_work_aspect_;
        double compose_scale_;
        double compose_work_aspect_;

        if(work_megapix > 0)
            work_scale_ = min(1.0, sqrt(work_megapix * 1e6 / full_img[i].size().area()));
        else
            work_scale_ = 1;

        if(seam_megapix > 0)
        {
            seam_scale_ = min(1.0, sqrt(seam_megapix * 1e6 / full_img[i].size().area()));
            seam_work_aspect_ = seam_scale_ / work_scale_;
        }
        else
        {
            seam_scale_ = 1;
            seam_work_aspect_ = 1;
        }

        if(compose_megapix > 0)
        {
            compose_scale_ = min(1.0, sqrt(compose_megapix * 1e6 / full_img[i].size().area()));
            compose_work_aspect_ = compose_scale_ / work_scale_;
        }
        else
        {
            compose_scale_ = 1;
            compose_work_aspect_ = 1;
        }

        work_scale.push_back(work_scale_);
        seam_scale.push_back(seam_scale_);
        seam_work_aspect.push_back(seam_work_aspect_);
        compose_scale.push_back(compose_scale_);
        compose_work_aspect.push_back(compose_work_aspect_);
    }
}

vector<ImageFeatures> Basic_stitcher::finding_features(const vector<Mat> &imgs)
{
    vector<ImageFeatures> features(imgs.size());
    for(int i = 0; i < imgs.size(); i++)
    {
        (*finder)(imgs[i], features[i]);
        features[i].img_idx = i;
    }

    return features;
}

vector<MatchesInfo> Basic_stitcher::pairwise_matching(const vector<ImageFeatures> &features)
{
    vector<MatchesInfo> pairwise_matches;
    (*matcher)(features, pairwise_matches);

    return pairwise_matches;
}

vector<CameraParams> Basic_stitcher::estimate_camera_params(const vector<ImageFeatures> &features, const vector<MatchesInfo> &pairwise_matches)
{
    vector<CameraParams> cameras;
    (*estimator)(features, pairwise_matches, cameras);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    adjuster->setConfThresh(1.0f);

    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    refine_mask(0,0) = 1;
    refine_mask(0,1) = 1;
    refine_mask(0,2) = 1;
    refine_mask(1,1) = 1;
    refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);

    (*adjuster)(features, pairwise_matches, cameras);

    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R.clone());
    waveCorrect(rmats, detail::WAVE_CORRECT_HORIZ);
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];

    return cameras;
}

void Basic_stitcher::warping_for_prepare_composition(const vector<Mat> &images, const vector<CameraParams> &cameras, vector<Point> &corners_out, vector<UMat> &warped_out, vector<UMat> &warped_mask_out, vector<Rect> &rois_out)
{
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        focals.push_back(cameras[i].focal);
    }
    sort(focals.begin(), focals.end());

    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    int num_images = images.size();
    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);
    vector<Rect> rois(num_images);

    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    for (int i = 0; i < num_images; ++i)
    {
        Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale) * static_cast<float>(seam_work_aspect[i]));
        
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect[i];
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);

        rois[i] = warper->warpRoi(images[i].size(), K, cameras[i].R);
    }
    
    for (int i = 0; i < num_images; ++i)
    {
        corners_out.push_back(corners[i]);
        warped_out.push_back(images_warped[i]);
        warped_mask_out.push_back(masks_warped[i]);
        rois_out.push_back(rois[i]);
    }
}

void Basic_stitcher::warping_for_composition(const vector<Mat> &images, const vector<CameraParams> &cameras, vector<Point> &corners_out, vector<UMat> &warped_out, vector<UMat> &warped_mask_out, vector<Rect> &rois_out)
{
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        focals.push_back(cameras[i].focal);
    }
    sort(focals.begin(), focals.end());

    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    int num_images = images.size();
    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);
    vector<Rect> rois(num_images);
    vector<CameraParams> cameras_;
    cameras_ = cameras;

    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    for (int i = 0; i < num_images; ++i)
    {
        Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale) * static_cast<float>(compose_work_aspect[i]));
        
        Mat_<float> K;
        cameras_[i].K().convertTo(K, CV_32F);
        cameras_[i].focal *= compose_work_aspect[i];
        cameras_[i].ppx *= compose_work_aspect[i];
        cameras_[i].ppy *= compose_work_aspect[i];

        Size sz = images[i].size();
        if (std::abs(compose_scale[i] - 1) > 1e-1)
        {
            sz.width = cvRound(images[i].size().width * compose_scale[i]);
            sz.height = cvRound(images[i].size().height * compose_scale[i]);
        }
        rois[i] = warper->warpRoi(sz, K, cameras_[i].R);

        corners[i] = warper->warp(images[i], K, cameras_[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras_[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }
    
    for (int i = 0; i < num_images; ++i)
    {
        corners_out.push_back(corners[i]);
        warped_out.push_back(images_warped[i]);
        warped_mask_out.push_back(masks_warped[i]);
        rois_out.push_back(rois[i]);
    }
}

void Basic_stitcher::finding_seam(vector<Point> &corners, vector<UMat> &warped, vector<UMat> &warped_mask)
{
    vector<UMat> images_warped_f(warped.size());
    for (int i = 0; i < warped.size(); ++i)
        warped[i].convertTo(images_warped_f[i], CV_32F);

    seam_finder->find(images_warped_f, corners, warped_mask);
}

void Basic_stitcher::feeding_exposure_compensator(vector<Point> &corners, vector<UMat> &images_warped, vector<UMat> &masks_warped)
{
    compensator->feed(corners, images_warped, masks_warped);
}

void Basic_stitcher::applying_exposure_compensator(vector<Point> &corners, vector<UMat> &images_warped, vector<UMat> &masks_warped)
{
    for(int img_idx = 0; img_idx < images_warped.size(); ++img_idx)
        compensator->apply(img_idx, corners[img_idx], images_warped[img_idx], masks_warped[img_idx]);
}

Mat Basic_stitcher::blending(vector<UMat> &image_warped, vector<Rect> &rois, vector<UMat> &seam_masks)
{
    vector<Point> corners;
    vector<Size> sizes;
    for(int i = 0; i <image_warped.size(); i++)
    {
        corners.push_back(rois[i].tl());
        sizes.push_back(rois[i].size());
    }
    blender->prepare(corners, sizes);

    for(int i = 0; i <image_warped.size(); i++)
    {
        Mat img_warped_s, mask_warped;
        image_warped[i].convertTo(img_warped_s, CV_16S);
    
        seam_masks[i].copyTo(mask_warped);

        blender->feed(img_warped_s, seam_masks[i], rois[i].tl());
    }

    Mat result, result_mask;
    blender->blend(result, result_mask);

    return result;
}

void Basic_stitcher::calculate_camera_params(vector<Mat> &full_img)
{
    STICHER_DBG_OUT("resize full images to working size images");
    vector<Mat> imgs;
    for(int i = 0; i < full_img.size(); i++)
    {
        Mat resized;
        resize(full_img[i], resized, Size(), work_scale[i], work_scale[i], INTER_LINEAR_EXACT);
        imgs.push_back(resized);
    }

    STICHER_DBG_OUT("finding_features");
    features = finding_features(imgs);

    STICHER_DBG_OUT("pairwise_matching");
    pairwise_matches = pairwise_matching(features);

    STICHER_DBG_OUT("estimate_camera_params");
    cameras = estimate_camera_params(features, pairwise_matches);
}

vector<CameraParams> Basic_stitcher::get_camera_params()
{
    return cameras;
}

void Basic_stitcher::set_camera_params(vector<CameraParams> &cameras_)
{
    cameras = cameras_;
}

void Basic_stitcher::prepare_compose(std::vector<cv::Mat> &full_img)
{
    STICHER_DBG_OUT("resize full images to seam scale size images");
    vector<Mat> imgs;
    for(int i = 0; i < full_img.size(); i++)
    {
        Mat resized;
        resize(full_img[i], resized, Size(), seam_scale[i], seam_scale[i], INTER_LINEAR_EXACT);
        imgs.push_back(resized);
    }

    STICHER_DBG_OUT("warping for finding seam and exposure compensation");
    warping_for_prepare_composition(imgs, cameras, corners_prepare, warped_prepare, warped_mask_prepare, rois_prepare);
    
    STICHER_DBG_OUT("feeding with seam scale resized images");
    feeding_exposure_compensator(corners_prepare, warped_prepare, warped_mask_prepare);

    STICHER_DBG_OUT("finding_seam");
    finding_seam(corners_prepare, warped_prepare, warped_mask_prepare);
}

void Basic_stitcher::compose(vector<Mat> &full_img)
{
    STICHER_DBG_OUT("resize full images to compose scale size images");
    vector<Mat> imgs;

    for(int i = 0; i < full_img.size(); i++)
    {
        if (abs(compose_scale[i] - 1) > 1e-1)
        {
            Mat resized;
            resize(full_img[i], resized, Size(), compose_scale[i], compose_scale[i], INTER_LINEAR_EXACT);
            imgs.push_back(resized);
        }
        else
        {
            imgs.push_back(full_img[i]);
        }
    }

    STICHER_DBG_OUT("warping for composition");
    warping_for_composition(imgs, cameras, corners_compose, warped_compose, warped_mask_compose, rois_compose);

    STICHER_DBG_OUT("applying exposure compensation gain");
    vector<Point> corner_roi;
    for(int i = 0; i < imgs.size(); i++)
        corner_roi.push_back(rois_compose[i].tl());
    applying_exposure_compensator(corner_roi, warped_compose, warped_mask_compose);

    STICHER_DBG_OUT("blending");
    vector<UMat> mask_warped_;
    for(int i = 0; i < imgs.size(); i++)
    {
        Mat dilated_mask;
        Mat seam_mask;
        Mat warped_mask_compose_;
        warped_mask_compose[i].copyTo(warped_mask_compose_);
        Mat mask_warped_tmp;

        dilate(warped_mask_prepare[i], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, warped_mask_compose[i].size(), 0, 0, INTER_LINEAR_EXACT);

        mask_warped_tmp = seam_mask & warped_mask_compose_;
        UMat tmp;
        mask_warped_tmp.copyTo(tmp);
        mask_warped_.push_back(tmp);
    }
    result = blending(warped_compose, rois_compose, mask_warped_);
}

Mat Basic_stitcher::stitcher_do_all(vector<Mat> &imgs)
{
    STICHER_DBG_OUT("----For fast stitching, resize images with scale factors----");
    update_image_scale(imgs);

    STICHER_DBG_OUT("----Calcualating CameraParams----");
    calculate_camera_params(imgs);

    STICHER_DBG_OUT("----Prepare exposure compensate gains, find seam----");
    prepare_compose(imgs);

    STICHER_DBG_OUT("----Warping for compose, apply exposure gain, blending----");
    compose(imgs);
    
    STICHER_DBG_OUT("done");
    return result;
}