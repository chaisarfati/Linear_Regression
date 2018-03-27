package HW1;

import weka.core.Attribute;
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


   		//build classifiers with all 3 attributes combinations
        for (int i = 0; i < training_data.numAttributes(); i++) {
            for (int j = i+1; j < training_data.numAttributes(); j++) {
                for (int k = i+2; k < training_data.numAttributes(); k++) {
                    Attribute att1 = training_data.attribute(i);
                    Attribute att2 = training_data.attribute(j);
                    Attribute att3 = training_data.attribute(k);


                }
            }
        }
        //training_data.insertAttributeAt();
		
	}

}
