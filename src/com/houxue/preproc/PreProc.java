package com.houxue.preproc;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

/**
 * 图片预处理，包括：转化为灰度图，直方图均衡化，对齐（归一化剪切）
 * @author houxue
 * @version 1.1
 * @date 2015.5.11
 */
public class PreProc {
	
	private static final String TAG = "PreProc.class";
	
	private static final String CLASSIFIER_HAAR_FACE1 = "classifier/haarcascade_frontalface_alt2.xml";
	//private static final String CLASSIFIER_HAAR_FACE2 = "";
	//private static final String CLASSIFIER_HAAR_FACE3 = "";
	private static final String CLASSIFIER_HAAR_EYE1 = "classifier/haarcascade_eye_tree_eyeglasses.xml";
	//private static final String CLASSIFIER_HAAR_EYE2 = "";
	private static final String CLASSIFIER_HAAR_NOSE = "classifier/haarcascade_mcs_nose.xml";
	private static final String CLASSIFIER_HAAR_MOUTH = "classifier/haarcascade_mcs_mouth.xml";
	
	/**
	 * 图像灰度化，直方图均衡化
	 * 
	 * @param srcMat 
	 * 			原始格式的图像
	 * 
	 * @return dstMat
	 * 			处理后的图像 
	 */
	public Mat cvtColHist(Mat srcMat) {
		// Mat tmpMat = new Mat();
		Mat dstMat = new Mat();
		
		// Imgproc.cvtColor(srcMat, tmpMat, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(srcMat, dstMat);
		
		return dstMat;
	}
	
	/**
	 * 特征位置检测
	 * 
	 * @param srcMat 灰度化和均衡化后的图像
	 * 
	 * @return feature 人脸特征
	 */
	public FeaturePosi FeatureDetect(Mat srcMat) {
		FeaturePosi feature = new FeaturePosi();
		
		CascadeClassifier faceClassifier = new CascadeClassifier(CLASSIFIER_HAAR_FACE1);
		CascadeClassifier eyeClassifier = new CascadeClassifier(CLASSIFIER_HAAR_EYE1);
		CascadeClassifier noseClassifier = new CascadeClassifier(CLASSIFIER_HAAR_NOSE);
		CascadeClassifier mouthClassifier = new CascadeClassifier(CLASSIFIER_HAAR_MOUTH);
		
		MatOfRect faceRect = new MatOfRect();
		faceClassifier.detectMultiScale(srcMat, faceRect);
		if(faceRect.toArray().length != 0) {
			feature.setFace(faceRect.toArray()[0]);
		} else {
			System.err.println(TAG + ": No face.");
		}
		
		MatOfRect eyeRect = new MatOfRect();
		eyeClassifier.detectMultiScale(srcMat, eyeRect);
		if(eyeRect.toArray().length > 1) {
			feature.setLeftEye(eyeRect.toArray()[0]);
			feature.setRightEye(eyeRect.toArray()[1]);
		}
		
		MatOfRect noseRect = new MatOfRect();
		noseClassifier.detectMultiScale(srcMat, noseRect);
		if(noseRect.toArray().length != 0) {
			feature.setNose(noseRect.toArray()[0]);
		}
		
		MatOfRect mouthRect = new MatOfRect();
		mouthClassifier.detectMultiScale(srcMat, mouthRect);
		if(mouthRect.toArray().length != 0) {
			feature.setMouth(mouthRect.toArray()[0]);
		}
		
		return feature;
	}
	
	/**
	 * 图片剪切
	 * 
	 * @param srcMat 灰度化和均衡化后的人脸图像
	 * @param fPosi 人脸特征，包括人脸，眼睛，鼻子，嘴巴的位置
	 * 
	 * @return dstMat 剪切后的人脸图像
	 */
	public Mat CropImg(Mat srcMat, FeaturePosi fPosi) {
		Mat dstMat = new Mat();
		Mat tmpMat = new Mat();
		
		if (fPosi.getFace() == null) {
			tmpMat = srcMat;
		} else {
			tmpMat = srcMat.submat(fPosi.getFace());
		}
		
		Imgproc.resize(tmpMat, dstMat, new Size(200, 200));
		
		return dstMat; 
	}
	
	/**
	 * 图像标准化
	 * 
	 * @param srcMat 灰度化和均衡化后的人脸图像
	 * 
	 * @return dstMat 标准化后的图像
	 */
	public Mat NormalImg(Mat srcMat) {
		Mat dstMat = new Mat();
		
		Mat tmpMat = cvtColHist(srcMat);
		FeaturePosi fp = FeatureDetect(tmpMat);
		
		//fp.print();
		
		dstMat = CropImg(tmpMat, fp);
		
		return dstMat;
	}

}
