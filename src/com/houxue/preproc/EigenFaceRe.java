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
	 * ����ѵ��
	 * 
	 * @param mat �洢ѵ��ͼ�������
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
		
		// ��ѵ���������һ������ÿ�д洢һ��ͼ��
		Mat data = new Mat(mat.size(), total, CvType.CV_64FC1);
		for (int i = 0; i < mat.size(); i++) {
			mat.get(i).reshape(1, 1).convertTo(data.row(i), CvType.CV_64FC1);
		}
		
		// clear existing model data
		_projection.clear();
		
		// ����ѵ������ÿ��feature�ľ�ֵ������������
		Core.PCACompute(data, mean, eigenvectors);

		// copy the PCA result
		_mean = mean.reshape(1, 1);
		// ��������������ת��
		Core.transpose(eigenvectors, _eigenvectors);
		
		// ������ӳ�䵽��ά�ռ�
		for (int i = 0; i < data.rows(); i++) {
			_projection.add( subspaceProject(_eigenvectors, _mean, data.row(i)) );
		}
	}
	
	/**
	 * ͼ��Ԥ��
	 * 
	 * @param mat ��ʶ���ͼ��
	 * 
	 * @return confidence ��ʶ���ͼ������֪ͼ��֮���ŷ�Ͼ���
	 */
	public Vector<Double> predict(Mat mat) {
		Vector<Double> confidences = new Vector<Double>();
		// ���Ƚ���ʶ���ͼ��ӳ�䵽��ά�ռ�
		Mat q = subspaceProject(_eigenvectors, _mean, mat.reshape(1, 1));
		
		for (int i = 0; i < _projection.size(); i++) {
			// �����������L2��������ŷ������
			double confidence = Core.norm(_projection.get(i), q, Core.NORM_L2);
			confidences.add(confidence);
		}
		
		return confidences;
	}
	
	/**
	 * ��ͼ��ӳ�䵽��ά�ռ�
	 * 
	 * @param eigenvectors ��ά�ռ����������
	 * @param mean ÿһά�ȵľ�ֵ
	 * @param mat ��ӳ���ͼ��
	 * 
	 * @return Y ӳ�䵽��ά�ռ��ͼ��
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
