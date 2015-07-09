package com.houxue.preproc;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

/**
 * ÈËÁ³¼ì²âÀà
 */
public class FaceDetect {
	private static final String TAG = "FaceDetect";
	
	private CascadeClassifier cascadeClassifier = null;
	
	public FaceDetect(String classifier) {
		cascadeClassifier = new CascadeClassifier(classifier);
	}
	
	public Mat detect(Mat srcMat) {
		Mat dst = new Mat();
		Mat dstMat = srcMat.clone();
		MatOfRect faceRect = new MatOfRect();
		
		cascadeClassifier.detectMultiScale(srcMat, faceRect);
		if (faceRect.toArray().length > 0) {
			for (Rect rect : faceRect.toArray()) {
				Core.rectangle(dstMat, new Point(rect.x, rect.y), new Point( (rect.x + rect.width), (rect.y + rect.height) ), new Scalar(0, 255, 0));
			}
			
			Mat crop = dstMat.submat(faceRect.toArray()[0]);
			Imgproc.resize(crop, dst, new Size(200, 200));
		} else {
			System.err.println(TAG + ".detect(): No face detect");
		}
		
		return dst;
	}
}
