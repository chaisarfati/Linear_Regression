package HW1;

import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class MainHW1 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		return inputReader;
	}
		
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		//load data
        Instances training_data = loadData("wind_training.txt"),
                testing_data = loadData("wind_testing.txt");
        LinearRegression model = new LinearRegression();
        model.buildClassifier(training_data);

        double testError = model.calculateMSE(testing_data),
                trainingError = model.calculateMSE(training_data);


		//find best alpha and build classifier with all attributes


   		//build classifiers with all 3 attributes combinations
		
	}

}
