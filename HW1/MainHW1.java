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
        int[] toConsider = new int[training_data.numAttributes() - 1];
        for (int i = 0; i < toConsider.length; i++) {
            toConsider[i] = i;
        }
        model.setToConsider(toConsider);
        model.buildClassifier(training_data);

        System.out.println("The chosen alpha is: " + model.getM_alpha() + "\n" +
        "Training error with all features is: " + model.calculateMSE(training_data) + "\n" +
                        "Test error with all features is: " + model.calculateMSE(testing_data));

        model.setM_truNumAttributes(4);
        int[] attributes = new int[3];
        double minTrainingErr =  Double.POSITIVE_INFINITY;
        double minTestingErr = 0;
        int[] ds = new int[3];

        System.out.println("List of all combination of 3 features and the training error");
        //build classifiers with all 3 attributes combinations
        for (int i = 0; i < 12; i++) {
            attributes[0] = i;
            for (int j = i + 1; j < 13; j++) {
                attributes[1] = j;
                for (int k = j + 1; k < 14; k++) {
                    attributes[2] = k;
                    model.resetCoefficients();
                    model.setToConsider(attributes);
                    model.gradientDescentAfterAlpha(training_data);

                    System.out.println(training_data.attribute(i).name() + ", " + training_data.attribute(j).name()
                            + ", " + training_data.attribute(k).name() +
                            " " + model.calculateMSE(training_data));

                    if(minTrainingErr > model.calculateMSE(training_data)){
                        minTrainingErr = model.calculateMSE(training_data);
                        minTestingErr = model.calculateMSE(testing_data);
                        ds[0] = i;
                        ds[1] = j;
                        ds[2] = k;
                    }
                }
            }
        }


        System.out.println("Training error the features " + training_data.attribute(ds[0]).name() + ", " +
                training_data.attribute(ds[1]).name() + ", " +
                training_data.attribute(ds[2]).name() + ", " +
                minTrainingErr);


        System.out.println("Test error the features " + training_data.attribute(ds[0]).name() + ", " +
                training_data.attribute(ds[1]).name() + ", " +
                training_data.attribute(ds[2]).name() + ", " +
                minTestingErr);


	}

}
