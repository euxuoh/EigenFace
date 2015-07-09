package com.houxue.localdetect;

import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import com.houxue.preproc.EigenFace;
import com.houxue.preproc.EigenFaceRe;
import com.houxue.preproc.FaceDetect;

public class OpenCVTest {

	private final static String IMAGE_TEST = "img/hx2_30.jpg";

	public static void main(String[] args) {
		System.loadLibrary("opencv_java2410");
		
		// ========================FaceDetect=======================
		// facedetect();
		
		// ========================EigenFaceRe=======================
		// eigenFaceReTest();
		
		// =========================EigenFace========================
		// eigenFaceTest();
		
		// =========================un PCA===========================
		unPCATest();
	}
	
	/**
	 * 
	 */
	public static void facedetect() {
		// haar feature aimed at face detect
		String classifier_haar_alt = "classifier/haarcascade_frontalface_alt.xml";
		String classifier_haar_alt2 = "classifier/haarcascade_frontalface_alt2.xml";
		
		// hog feature aimed at pedestrian detect
		String classifier_hog = "classifier/hogcascade_pedestrians.xml";
		
		// lbp feature aimed at what?
		String classifier_lbp_frontal = "classifier/lbpcascade_frontalface.xml";
		String classifier_lbp_profile = "classifier/lbpcascade_profileface.xml";
		
		Vector<Mat> image = new Vector<Mat>();
		for (int i = 1; i < 51; i++) {
			String filename = String.format("detect/input/in (%d).jpg", i);
			
			// preprocess
			//Mat im = new Mat();
			//Imgproc.equalizeHist(Highgui.imread(filename, 0), im);
			
			// unpreprocess
			Mat im = Highgui.imread(filename, 0);
			
			image.add( im );
		}
		
		FaceDetect fd = new FaceDetect(classifier_haar_alt2);
		
		long endtime = 0L;
		long starttime = System.currentTimeMillis();
		int i = 1;
		for (Mat mat : image) {
			String name = String.format("detect/output/out_%d.jpg", i++);
			
			Mat dst = fd.detect(mat);
			
			Highgui.imwrite(name, dst);
		}
		endtime = System.currentTimeMillis();
		System.out.println("Time: " + (endtime - starttime) );
	}
	
	/**
	 * 
	 */
	public static void eigenFaceReTest() {
		
		// PreProc pp = new PreProc();
		Vector<Mat> image = new Vector<Mat>();
		for (int i = 1; i < 6; i++) {
			String filename = String.format("eigenface/hx%d_30.jpg", i);
			
			Mat im = new Mat();
			Imgproc.equalizeHist(Highgui.imread(filename, 0), im);
			
			image.add( im );
		}
		
		// Mat testSample = Highgui.imread(IMAGE_TEST, 0);//pp.NormalImg( Highgui.imread(IMAGE_TEST) ).reshape(1);
		long start = System.currentTimeMillis();
		EigenFaceRe efr = new EigenFaceRe();
		efr.train(image);
		long time = System.currentTimeMillis() - start;
		System.out.println("Time = " + time);
		
		Vector<Mat> testimage = new Vector<Mat>();
		for (int i = 1; i < 19; i++) {
			String filename = String.format("eigenface/test (%d).jpg", i);
			
			Mat im = new Mat();
			Imgproc.equalizeHist(Highgui.imread(filename, 0), im);
			
			testimage.add(im);
		}
		
		int count = 0;
		
		for (Mat mat : testimage) {
			System.out.println( "==========test" + (++count) );
			Vector<Double> confidence = efr.predict(mat);
			
			for (int i = 0; i < confidence.size(); i++) {
				System.out.println(confidence.get(i));
			}
		}
		
		// ============================mean face===============================
		Mat mean = efr.getMean().reshape(1, 200);
		Highgui.imwrite("out/mean.jpg", mean);
		
		// ============================eigen face==============================
		Mat eigenvectors = efr.getEigenVectors();
		System.out.println("eigenvectors type : " + eigenvectors.rows() + "-" + eigenvectors.cols());
		for (int i = 0; i < eigenvectors.cols(); i++) {
			String filename = String.format("out/eigen%d.png", i);
			Mat ev = eigenvectors.col(i).clone().reshape(1, 200);
			System.out.println(ev.type() + " " + ev.rows() + " " + ev.cols());
			Mat grayscale = new Mat();
			Core.normalize(ev, grayscale, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
			
			Highgui.imwrite(filename, grayscale);
		}
	}
	
	/**
	 * 
	 */
	public static void unPCATest() {
		Mat mat1 = Highgui.imread("eigenface/hx1_30.jpg", 0);
		Mat mat2 = Highgui.imread("eigenface/hx2_30.jpg", 0);
		
		long start = System.currentTimeMillis();
		double distance = Core.norm(mat1, mat2, Core.NORM_L2);
		long time = System.currentTimeMillis() - start;
		System.out.println("time = " + time);
	}
	
	
	/**
	 * 
	 */
	public static void eigenFaceTest() {
		String filename = String.format("img/hx%d_20.jpg", 1);
		Mat image = Highgui.imread(filename, 0);
		// Mat testImage = Highgui.imread("img/hx_20.jpg", 0);
		EigenFace ef = new EigenFace();
		ef.train(image);
		
	}

}
