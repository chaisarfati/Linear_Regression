package HW1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {

    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;


    public void setM_truNumAttributes(int m_truNumAttributes) {
        this.m_truNumAttributes = m_truNumAttributes;
    }

    public double getM_alpha() {
        return m_alpha;
    }

    //the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
        m_truNumAttributes = trainingData.numAttributes();
        if(m_coefficients == null) {
            m_coefficients = new double[m_truNumAttributes];
            for (int i = 0; i < m_coefficients.length; i++) {
                m_coefficients[i] = 1;
            }
        }
        findAlpha(trainingData);
	}

    private void findAlpha(Instances data) throws Exception {
        double[] errors = new double[17];
        double[][] coeff = new double[17][m_truNumAttributes];

        for (int i = -17; i < 0; i++) {
            m_alpha = Math.pow(3, i);
            gradientDescent(data);
            double prevError = calculateMSE(data), currentError = 0;

            for (int j = 0; j < 20000; j++) {
                gradientDescent(data);

                if(j%100==0){

                    for (int k = 0; k < m_coefficients.length; k++) {
                        coeff[i+17][k] = m_coefficients[k];
                    }

                    System.out.println(prevError);
                    currentError = calculateMSE(data);
                    if(currentError > prevError){
                        prevError = currentError;
                        break;
                    }else{
                        prevError = currentError;
                    }
                }

            }
            errors[i+17] = prevError;
        }
        m_alpha = Math.pow(3, findMinIndex(errors) - 17);
        m_coefficients = coeff[findMinIndex(errors)];

        System.out.println("!!!!!!!!!!! " + m_alpha);

        //gradientDescentAfterAlpha(data);
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

        double[] temp = new double[m_coefficients.length];

            // For all thetas
            for (int k = 0; k < m_truNumAttributes; k++) {
                temp[k] = m_coefficients[k] - m_alpha * adjustDirection(trainingData, k);
            }

            // Update the actual thetas
            for (int i = 0; i < m_coefficients.length; i++) {
                m_coefficients[i] = temp[i];
            }

        return m_coefficients;
	}

    public double[] gradientDescentAfterAlpha(Instances trainingData) throws Exception {
        double[] temp = new double[m_coefficients.length];
        double prevError = calculateMSE(trainingData);
        double error = 0;
        int counter = 1;
        boolean dontstop = true;

        while (dontstop) {
            // For all thetas
            for (int k = 0; k < m_truNumAttributes; k++) {
                temp[k] = m_coefficients[k] - m_alpha * adjustDirection(trainingData, k);
            }

            // Update the actual thetas
            for (int i = 0; i < m_coefficients.length; i++) {
                m_coefficients[i] = temp[i];
            }

            if(counter==100){
                prevError = calculateMSE(trainingData);
            }
            if(counter==200){
                counter = 1;
                error = calculateMSE(trainingData);
                if(Math.abs(error - prevError) > 0.003){
                    dontstop = false;
                }
            }
            counter++;

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
        for (int i = 1; i < m_truNumAttributes; i++) {
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
                    data.instance(i).value(m_ClassIndex), 2);
        }
        return sum / (2.0 * m);
	}

	public double adjustDirection(Instances instances, int j) throws Exception {
        double sum = 0;
        if(j==0){
            for (int i = 0; i < instances.numInstances(); i++) {
                sum += (regressionPrediction(instances.instance(i)) - instances.instance(i).value(m_ClassIndex));
            }
        }else {
            for (int i = 0; i < instances.numInstances(); i++) {
                sum += (regressionPrediction(instances.instance(i)) - instances.instance(i).value(m_ClassIndex))
                        * instances.instance(i).value(j-1);
            }
        }
        return  sum/((double)instances.numInstances());
    }

    public void resetCoefficients(){
        for (int i = 0; i < m_coefficients.length; i++) {
            m_coefficients[i] = 1;
        }
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

}
