package com.houxue.preproc;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

public class EigenFace {

	// private final String TAG = "EigenFace.class";
	
	private Mat _eigenVectors = new Mat();
	private Mat _mean = new Mat();
	private Mat _projections = new Mat();

	public EigenFace() {
	}
	
	public void train(Mat mat) {
		Mat srcT = new Mat();
		mat.convertTo(srcT, CvType.CV_64FC1);
		
		Mat mean = new Mat();
		Mat eigenvectors = new Mat();
		Core.PCACompute(srcT, mean, eigenvectors);
		Mat mean1 = new Mat();
		Core.transpose(mean, mean1);
		
		Mat X = new Mat();
		mat.convertTo(X, mean1.type());
		for (int i = 0; i < mat.cols(); i++) {
			Mat tmp = X.col(i);
			Core.subtract(tmp, mean1, X.col(i));
		}
		
		Highgui.imwrite("out/meanMat.jpg", X);
		
		Mat Xt = new Mat();
		Mat Y = new Mat();
		Core.transpose(X, Xt);
		Core.gemm(Xt, X, 1.0, new Mat(), 0.0, Y);
		
		Mat mean2 = new Mat();
		Mat eigenvector1 = new Mat();
		Core.PCACompute(Y, mean2, eigenvector1);
	}
	
	public double predict(Mat mat) {
		Mat srcMat = new Mat();
		mat.convertTo(srcMat, CvType.CV_64FC1);
		
		Mat projection = subspaceProject(this._eigenVectors, this._mean, srcMat);
		
		double confidence = Core.norm(_projections, projection, Core.NORM_L2);

		return confidence;
	}

	public void train0(Mat mat) {
		Mat srcMatT = new Mat();
		Mat eigenVectors = new Mat();
		Mat mean = new Mat();
		Mat mean1 = new Mat();
		Mat covMat = new Mat();
		
		Core.transpose(mat, srcMatT);
		Core.PCACompute(srcMatT, mean, eigenVectors);

		mean.convertTo(mean1, CvType.CV_64FC1);
		Core.transpose(mean1, mean1);

		// 将矩阵的每一行进行零均值化，即减去每一个feature的均值
		for (int i = 0; i < mat.cols(); i++) {
			Mat tmp = mat.col(i);
			Core.subtract(tmp, mean1, mat.col(i));
		}
				
		// 求出协方差矩阵
		Core.gemm(mat, srcMatT, (double)1/mat.cols(), new Mat(), 0.0, covMat);
		
		// 求出协方差矩阵的特征向量
		Core.PCACompute(covMat, this._mean, this._eigenVectors);
		
		_projections.release();
		
		_projections = subspaceProject(this._eigenVectors, this._mean, mat);
	}

	public Mat subspaceProject(Mat eigenvector, Mat mean, Mat mat) {
		Mat pMat = new Mat();
		Mat yMat = new Mat();
		
		eigenvector.convertTo(pMat, CvType.CV_64FC1);
		Core.gemm(pMat, mat, 1.0, new Mat(), 0.0, yMat);
				
		Highgui.imwrite("out/face.png", yMat);
		
		return yMat;
	}

}
