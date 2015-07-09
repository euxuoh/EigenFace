package com.houxue.localdetect;

import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class GetPerson {

	public static void main(String[] args) {
		System.loadLibrary("opencv_java2410");
		
		Mat src = Highgui.imread("img/hx_30.jpg", 0);
		Mat dst = new Mat();
		
		Imgproc.equalizeHist(src, dst);
		
		Highgui.imwrite("out/hx_30_src.jpg", src);
		Highgui.imwrite("out/hx_30.jpg", dst);
	}

}
