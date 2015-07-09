package com.houxue.preproc;

import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class EigenFaceRe {
	
	private static final String TAG = "EigenFaceRe";
	
	private Mat _eigenvalue = new Mat();
	private Mat _mean = new Mat();
	private Mat _eigenvectors = new Mat();
	private Vector<Mat> _projection = new Vector<Mat>();
	
	/**
	 * 数据训练
	 * 
	 * @param mat 存储训练图像的数组
	 * 
	 * @return void
	 */
	public void train(Vector<Mat> mat) {
		if (mat.size() == 0) {
			String error_message = String.format("Empty training data was given. You will need more than one sample to learn a model.");
			System.err.println(TAG + ": " + error_message);
		}
		
		int rows = mat.get(0).rows();
		int cols = mat.get(0).cols();
		int total = rows * cols;
		
		Mat mean = new Mat();
		Mat eigenvectors = new Mat();
		
		// 将训练数据组成一个矩阵，每行存储一个图像
		Mat data = new Mat(mat.size(), total, CvType.CV_64FC1);
		for (int i = 0; i < mat.size(); i++) {
			mat.get(i).reshape(1, 1).convertTo(data.row(i), CvType.CV_64FC1);
		}
		
		// clear existing model data
		_projection.clear();
		
		// 计算训练数据每个feature的均值，和特征向量
		Core.PCACompute(data, mean, eigenvectors);

		// copy the PCA result
		_mean = mean.reshape(1, 1);
		// 将特征向量进行转置
		Core.transpose(eigenvectors, _eigenvectors);
		
		// 将数据映射到低维空间
		for (int i = 0; i < data.rows(); i++) {
			_projection.add( subspaceProject(_eigenvectors, _mean, data.row(i)) );
		}
	}
	
	/**
	 * 图像预测
	 * 
	 * @param mat 待识别的图像
	 * 
	 * @return confidence 待识别的图像与已知图像之间的欧氏距离
	 */
	public Vector<Double> predict(Mat mat) {
		Vector<Double> confidences = new Vector<Double>();
		// 首先将待识别的图像映射到低维空间
		Mat q = subspaceProject(_eigenvectors, _mean, mat.reshape(1, 1));
		
		for (int i = 0; i < _projection.size(); i++) {
			// 计算两矩阵的L2范数，即欧拉距离
			double confidence = Core.norm(_projection.get(i), q, Core.NORM_L2);
			confidences.add(confidence);
		}
		
		return confidences;
	}
	
	/**
	 * 将图像映射到低维空间
	 * 
	 * @param eigenvectors 低维空间的特征向量
	 * @param mean 每一维度的均值
	 * @param mat 待映射的图像
	 * 
	 * @return Y 映射到低维空间的图像
	 */
	public Mat subspaceProject(Mat eigenvectors, Mat mean, Mat mat) {
		int n = mat.rows();
		// int d = mat.cols();
				
		Mat X = new Mat();
		Mat Y = new Mat();
		
		mat.convertTo(X, eigenvectors.type());
		
		for (int i = 0; i < n; i++) {
			Mat r_i = X.row(i);
			Core.subtract(r_i, mean.reshape(1, 1), X.row(i));
		}
		
		Core.gemm(X, eigenvectors, 1.0, new Mat(), 0.0, Y);
		// System.out.println("Y = " + Y.rows() + "-" + Y.cols());
		
		return Y;
	}
	
	public Mat getEigenValues() {
		return this._eigenvalue;
	}
	
	public Mat getEigenVectors() {
		return this._eigenvectors;
	}
	
	public Mat getMean() {
		return this._mean;
	}
	
	public Vector<Mat> getProjection() {
		return this._projection;
	}

}
