#include "stitcher.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

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

void Basic_stitcher::warping_images(const vector<Mat> &images, const vector<CameraParams> &cameras, vector<Point> &corners_out, vector<UMat> &warped_out, vector<UMat> &warped_mask_out, vector<Rect> &rois_out)
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

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);

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

void Basic_stitcher::finding_seam(vector<Point> &corners, vector<UMat> &warped, vector<UMat> &warped_mask)
{
    vector<UMat> images_warped_f(warped.size());
    for (int i = 0; i < warped.size(); ++i)
        warped[i].convertTo(images_warped_f[i], CV_32F);

    seam_finder->find(images_warped_f, corners, warped_mask);
}

void Basic_stitcher::compensating_exposure(vector<Point> &corners, vector<UMat> &images_warped, vector<UMat> &masks_warped)
{
    compensator->feed(corners, images_warped, masks_warped);
    for (int img_idx = 0; img_idx < images_warped.size(); ++img_idx)
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

Mat Basic_stitcher::stitcher_do_all(vector<Mat> &imgs)
{
    STICHER_DBG_OUT("finding_features");
    features = finding_features(imgs);

    STICHER_DBG_OUT("pairwise_matching");
    pairwise_matches = pairwise_matching(features);

    STICHER_DBG_OUT("estimate_camera_params");
    cameras = estimate_camera_params(features, pairwise_matches);
    
    STICHER_DBG_OUT("warping_images");
    warping_images(imgs, cameras, corners, warped, warped_mask, rois);

    for(int i = 0; i < imgs.size(); i++)
    {
        warped_mask_seam.push_back(warped_mask[i].clone());
        warped_mask_expo.push_back(warped_mask[i].clone());
    }

    STICHER_DBG_OUT("finding_seam");
    finding_seam(corners, warped, warped_mask_seam);

    STICHER_DBG_OUT("compensating_exposure");
    vector<Point> corner_roi;
    for(int i = 0; i < imgs.size(); i++)
        corner_roi.push_back(rois[i].tl());
    compensating_exposure(corner_roi, warped, warped_mask_expo);

    STICHER_DBG_OUT("blending");
    result = blending(warped, rois, warped_mask_seam);
    
    STICHER_DBG_OUT("done");
    return result;
}