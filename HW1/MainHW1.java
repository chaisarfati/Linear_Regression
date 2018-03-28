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


		//find best alpha and build classifier with all attributes
        LinearRegression model = new LinearRegression();
        model.buildClassifier(training_data);

        double testErrorAll = model.calculateMSE(testing_data),
                trainingErrorAll = model.calculateMSE(training_data);

        System.out.println("Training error : " + trainingErrorAll);
        System.out.println("Testing error : " + testErrorAll);
        System.out.println("Best alpha : " + model.getM_alpha());

        /*
   		//build classifiers with all 3 attributes combinations
        for (int i = 0; i < 13; i++) {
            for (int j = i + 1; j < 14; j++) {
                for (int k = j + 1; k < 15; k++) {

                }
            }
        }
        */

	}

}
