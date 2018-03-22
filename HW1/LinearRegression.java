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
        findAlpha(trainingData);
		m_coefficients = gradientDescent(trainingData);
	}
	
	private void findAlpha(Instances data) throws Exception {
        double[] errors = new double[17];
        for (int i = -17; i < 0; i++) {
            m_alpha = Math.pow(3, i);
            double error = Double.POSITIVE_INFINITY, currentError = 0;
            for (int j = 0; j < 20000; j++) {
                m_coefficients = gradientDescent(data);
                if(j%100==0){
                    //calculate error and compare to the previous one
                    currentError = calculateMSE(data);
                    if(currentError > error){
                        errors[i+17] = error;
                        break;
                    }
                    error = currentError;
                }
                errors[i+17] = currentError;
            }
        }
        m_alpha = Math.pow(3, findMinIndex(errors) - 17);
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
        int numInstances = trainingData.numInstances();

        if(m_coefficients == null) {
            m_coefficients = new double[numAttributes];
            for (int i = 0; i < m_coefficients.length; i++) {
                m_coefficients[i] = 1;
            }
        }

        double[] temp = new double[numAttributes];
        double partDerivative;
        double sum, inSum;

        // For all thetas
        for (int k = 0; k < m_coefficients.length; k++) {
            sum = 0;

            // Sum on all instances
            for (int i = 0; i < trainingData.numInstances(); i++) {

                inSum = 0;
                /*if(k==0){
                    sum += m_coefficients[0] + scalarProduct(m_coefficients, xi) - xi[xi.length - 1];
                }else{
                    sum += m_coefficients[0];
                    sum += scalarProduct(m_coefficients, xi);
                    sum -= xi[xi.length - 1];
                    sum *= xi[k - 1];
                }*/

                double innerProduct = 0;
                for (int l = 1; i < numAttributes; i++){
                    innerProduct +=
                            trainingData.instance(i).value(l-1) * m_coefficients[l];
                }
                innerProduct += m_coefficients[0];

                inSum += innerProduct - trainingData.instance(i).value(numAttributes-1);
                if(k!=0) {
                    inSum *= trainingData.instance(i).value(k);
                }
                sum += inSum;
            }

            // Update temp
            temp[k] = m_coefficients[k] - m_alpha * ((1/((double)numInstances)) * sum );
        }

        // Update the actual thetas
        for (int i = 0; i < m_coefficients.length; i++) {
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
        for (int i = 0; i < m_coefficients.length; i++) {
            sum += Math.pow(scalarProduct(m_coefficients, data.instance(i).toDoubleArray()) -
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

	private static double scalarProduct(double[] coeff, double[] instance) {
        double result = 0;
        for (int i = 1; i <coeff.length; i++) {
            result += coeff[i] * instance[i-1];
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
