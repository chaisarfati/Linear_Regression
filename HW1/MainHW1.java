package HW1;

import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

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

        System.out.println("The chosen alpha is: " + model.getM_alpha() + "\n" +
        "Training error with all features is: " + model.calculateMSE(training_data) + "\n" +
                        "Test error with all features is: " + model.calculateMSE(testing_data) + "\n");

        model.setM_truNumAttributes(4);
        model.resetCoefficients();
        Remove remove = new Remove();
        remove.setInvertSelection(true);
        int[] attributes = new int[3];

   		//build classifiers with all 3 attributes combinations
        for (int i = 0; i < 12; i++) {
            attributes[0] = i;
            for (int j = i + 1; j < 13; j++) {
                attributes[1] = j;
                for (int k = j + 1; k < 14; k++) {
                    attributes[2] = k;

                    remove.setAttributeIndicesArray(attributes);
                    remove.setInputFormat(testing_data);
                    remove.setInputFormat(training_data);
                    model.gradientDescentAfterAlpha(training_data);

                    System.out.println(training_data.attribute(i).name() + " " + training_data.attribute(j).name() + " " + training_data.attribute(k).name() +
                            " " + "Training error : " + model.calculateMSE(training_data));
                }
            }
        }


	}

}
