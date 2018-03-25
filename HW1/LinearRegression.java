package HW1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class LinearRegression implements Classifier {

    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
        //TODO: complete this method
        m_coefficients = gradientDescent(trainingData);
	}

	private void findAlpha(Instances data) throws Exception {
        double[] errors = new double[17];
        for (int i = -17; i < 0; i++) {
            m_alpha = Math.pow(3, i);
            gradientDescent(data);
            double error = calculateMSE(data), currentError = 0;
            for (int j = 0; j < 20000; j++) {
                gradientDescent(data);
                if(j%100==1){
                    currentError = calculateMSE(data);
                    if(currentError > error){
                        currentError = error;
                        break;
                    }
                    error = currentError;
                }
            }
            errors[i+17] = currentError;
        }
        m_alpha = Math.pow(3, findMinIndex(errors) - 17);
        System.out.println(m_alpha);
    }

	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     *
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {

        int numAttributes = trainingData.numAttributes();
        int m = trainingData.numInstances();

        if(m_coefficients == null) {
            m_coefficients = new double[numAttributes];
            for (int i = 0; i < m_coefficients.length; i++) {
                m_coefficients[i] = 1;
            }
        }

        double[] temp = new double[numAttributes];

        while(calculateMSE(trainingData) > 0.003) {

            // For all thetas
            for (int k = 0; k < m_coefficients.length; k++) {

                double partDerivative = 0;
                for (int i = 0; i < m; i++) {
                    double xik = 1;
                    if (k != 0) {
                        xik = trainingData.instance(i).value(k);
                    }
                    Instance instance = trainingData.instance(i);
                    if (k == 0) {
                        partDerivative += (regressionPrediction(instance) - instance.value(numAttributes - 1));
                    } else {
                        partDerivative += (regressionPrediction(instance) - instance.value(numAttributes - 1)) * xik;
                    }
                }
                partDerivative = (1.0/m) * partDerivative;

                // Update temp
                temp[k] = m_coefficients[k] - m_alpha * partDerivative;

            }

            // Update the actual thetas
            for (int i = 0; i < m_coefficients.length; i++) {
                m_coefficients[i] = temp[i];
            }
        }
        return m_coefficients;
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
        double result = m_coefficients[0];
        for (int i = 1; i < instance.numAttributes(); i++) {
            result += m_coefficients[i] * instance.value(i-1);
        }
        return result;
	}

	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
        double sum = 0;
        double m = data.numInstances();
        for (int i = 0; i < m; i++) {
            sum += Math.pow(regressionPrediction(data.instance(i)) -
                    data.instance(i).value(data.numAttributes()-1), 2);
        }
        return (1.0/(2.0*m)) * sum;
	}

    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}

    private static int findMinIndex(double[] arr){
        double min = arr[0];
        int j = 0;
        for (int i = 1; i < arr.length; i++) {
            if (min > arr[i]) {
                min = arr[i];
                j = i;
            }
        }
        return j;
    }

    // !!!!!!!!!! TESTING. TO REMOVE BEFORE SUBMITTING !!!!!!!!!!!!!!!!!!
    public static void main(String[] args) throws Exception {

        // CREATES INSTANCE OF OUR DATA
        Instances data = null;
        try {
            data = new Instances(new BufferedReader(new FileReader("wind_training.txt")));
        } catch (IOException e) {
            e.printStackTrace();
        }

        LinearRegression linear = new LinearRegression();
        linear.m_alpha = 0.003;
        linear.gradientDescent(data);
        for (int i = 0; i < linear.m_coefficients.length; i++) {
            System.out.println(linear.m_coefficients[i]);
        }

    }
}
