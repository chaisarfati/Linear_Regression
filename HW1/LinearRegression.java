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

        double error = 100;
        double currentError = calculateMSE(data);
        while(currentError > 0.003){
            error = currentError;
            for (int i = 0; i < 100; i++) {
                gradientDescent(data);
            }
            currentError = calculateMSE(data);
            System.out.println(currentError);
        }
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


        int m = trainingData.numInstances();

        if(m_coefficients == null) {
            m_coefficients = new double[m_truNumAttributes];
            for (int i = 0; i < m_coefficients.length; i++) {
                m_coefficients[i] = 1;
            }
        }
        double[] temp = new double[m_truNumAttributes];

        // For all thetas
        for (int k = 0; k < m_coefficients.length; k++) {


            double partDerivative = 0;
            for (int i = 0; i < m; i++) {
                double xik = 1;
                if(k!=0) {
                    xik = trainingData.instance(i).value(k);
                }
                Instance instance = trainingData.instance(i);
                if(k==0){
                    partDerivative += (regressionPrediction(instance) - instance.value(m_ClassIndex));
                }else{
                    partDerivative += (regressionPrediction(instance) - instance.value(m_ClassIndex))*xik;
                }
            }
            partDerivative = 1/m * partDerivative;

            // Update temp
            temp[k] = m_coefficients[k] - m_alpha * partDerivative;

        }

        // Update the actual thetas
        for (int i = 0; i < m_coefficients.length; i++) {
            m_coefficients[i] = temp[i];
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
            result += m_coefficients[i] * (instance.value(i-1));
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
        for (int i = 0; i < data.numInstances(); i++) {
            sum += (regressionPrediction(data.instance(i)) -
                    data.instance(i).value(data.numAttributes()-1))*(regressionPrediction(data.instance(i)) -
                    data.instance(i).value(data.numAttributes()-1));
        }
        return  (1/(2.0*m)) * sum;
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

    /*private static double partialDerivative(int k, Instances data){
        int m = data.numInstances();
        double result = 0;
        for (int i = 0; i < m; i++) {
            double xik = data.instance(i).value(k);
            Instance instance = data.instance(i);
            result += (regressionPrediction(instance) - instance.value(data.numAttributes()-1))*xik;
        }
        return 1/m * result;
    }*/

    public static void main(String[] args) throws Exception {

        // CREATES INSTANCE OF OUR DATA
        Instances data = null;
        try {
            data = new Instances(new BufferedReader(new FileReader("wind_training.txt")));
        } catch (IOException e) {
            e.printStackTrace();
        }

        data.setClass(data.attribute(data.numAttributes()-1));
        System.out.println(data.classIndex());
        System.out.println(data.classAttribute());
        LinearRegression linear = new LinearRegression();
        linear.findAlpha(data);
    }
}
