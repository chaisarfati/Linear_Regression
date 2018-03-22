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
            double error = calculateMSE(data), currentError = 0;
            for (int j = 0; j < 20000; j++) {
                m_coefficients = gradientDescent(data);
                if(i%100==0){
                    //calculate error and compare to the previous one
                    currentError = calculateMSE(data);
                    if(currentError > error){
                        errors[i+17] = error;
                        break;
                    }
                }
                errors[i+17] = currentError;
            }
        }
        m_alpha = findMin(errors);
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
        m_alpha = 0.03;
        // weights array is thetas guess = 1 and temp store temporary thetas
        int numAttributes = trainingData.numAttributes();
        int numInstances = trainingData.numInstances();

        m_coefficients = new double[numAttributes];
        double[] temp = new double[numAttributes];

        for (int i = 0; i < m_coefficients.length; i++) {
            m_coefficients[i] = 1;
            temp[i] = 0;
        }

        double sum;
        double[] xi;


        // For all thetas
        for (int k = 0; k < temp.length; k++) {
            sum = 0;

            // Sum on all instances
            for (int i = 0; i < trainingData.numInstances(); i++) {

                xi = trainingData.instance(i).toDoubleArray();
                if(k==0){
                    sum += scalarProduct(xi, m_coefficients);
                    sum -= xi[xi.length - 1];
                }else{
                    sum += scalarProduct(xi, m_coefficients);
                    sum -= xi[xi.length - 1];
                    sum *= xi[k - 1];
                }
                sum += m_coefficients[0];

            }

            // Update temp
            temp[k] = m_coefficients[k] - m_alpha * ((1/((double)numInstances)) * sum );
        }

        // Update the actual thetas
        for (int i = 0; i < m_coefficients.length - 1; i++) {
            m_coefficients[i] = temp[i];
        }

        // !! FOR TESTING !!
        String s = "";
        for (int i = 0; i < m_coefficients.length; i++) {
            s += " Theta"+i+" = " + m_coefficients[i];
        }
        System.out.println(s);

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
        double result = 0;
        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            result += instance.value(i) * m_coefficients[i];
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
        for (int i = 0; i < data.numInstances(); i++) {
            sum += Math.pow(m_coefficients[i] -
                    data.instance(i).value(data.numAttributes()-1), 2);
        }
        return  (1/(2.0*(double)data.numInstances())) * sum;
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

	private static double scalarProduct(double[] v1, double[] v2) {
        double result = 0;
        for (int i = 0; i <v1.length-1; i++) {
            result += v1[i] * v2[i];
        }
        return result;
    }

    private static int findMinIndex(double[] arr){
        double min = arr[0];
        int j = 0;
        for (int i = 1; i < arr.length; i++) {
            if (min > arr[i]) min = arr[i];
            j= i;
        }
        return j;
    }

    public static void main(String[] args) throws Exception {

        // CREATES INSTANCE OF OUR DATA
        Instances data = null;
        try {
            data = new Instances(new BufferedReader(new FileReader("wind_training.txt")));
        } catch (IOException e) {
            e.printStackTrace();
        }

        LinearRegression linear = new LinearRegression();
        linear.gradientDescent(data);

    }
}
