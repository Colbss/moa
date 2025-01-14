/*
 *    kNN.java
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.lazy;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.Measurement;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * k Nearest Neighbor.<p>
 *
 * Valid options are:<p>
 *
 * -k number of neighbours <br> -m max instances <br> 
 *
 * @author Jesse Read (jesse@tsc.uc3m.es)
 * @version 03.2012
 */
public class kNN_Centroid2 extends AbstractClassifier implements MultiClassClassifier, Regressor {

    private static final long serialVersionUID = 1L;

	public IntOption kOption = new IntOption( "k", 'k', "The number of neighbors", 10, 1, Integer.MAX_VALUE);

	// For checking regression with mean value or median value
	public FlagOption medianOption = new FlagOption("median",'m',"median or mean");

	public IntOption limitOption = new IntOption( "limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

        public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption(
            "nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{
                "LinearNN", "KDTree"},
            new String[]{"Brute force search algorithm for nearest neighbour search. ",
                "KDTree search algorithm for nearest neighbour search"
            }, 0);


	int C = 0;

    @Override
    public String getPurposeString() {
        return "kNN: special.";
    }

    protected Instances window;

	protected Instances centroids;

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			this.window = new Instances(context,0); //new StringReader(context.toString())
			this.window.setClassIndex(context.classIndex());
		} catch(Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

    @Override
    public void resetLearningImpl() {
		this.window = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
		if (inst.classValue() > C)
			C = (int)inst.classValue();
		if (this.window == null) {
			this.window = new Instances(inst.dataset());
		}
		if (this.limitOption.getValue() <= this.window.numInstances()) {
			this.window.delete(0);
		}
		this.window.add(inst);
		addToCentroids(inst);


    }

	public void addToCentroids(Instance inst){

		if(this.centroids == null){
			this.centroids = new Instances(inst.dataset());
			this.centroids.add(inst);
		}
		else{

			for(int i = 0; i < this.centroids.size(); i++){

				Instance cent = this.centroids.get(i);

				if(cent.classValue() == inst.classValue()) {

					double[] cent_attrs = cent.toDoubleArray();

				}
				else{
					if(i != this.centroids.size()-1) {
						this.centroids.add(inst);
						break;
					}
				}
			}

		}

	}

	public void printInstance(Instance inst){
		StringBuilder sb = new StringBuilder();
		double classValue = inst.classValue();
		double[] attrs = inst.toDoubleArray();
		sb.append("Class: ").append(classValue).append(", Attributes: [");
		for( int i = 0; i < inst.numAttributes(); i++){
			sb.append(attrs[i]);
			if (i < inst.numAttributes() - 1) {
				sb.append(", ");
			}
		}
		sb.append("]\n");
		// Print the map contents using a single println statement
		System.out.println("Map: ");
		System.out.println(sb.toString());
	}

	@Override
    public double[] getVotesForInstance(Instance inst) {
		double v[] = new double[C+1];
		try {
			NearestNeighbourSearch search;
			if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
				search = new LinearNNSearch(this.window);  
			} else {
				search = new KDTree();
				search.setInstances(this.window);
			}	
			if (this.window.numInstances()>0) {	
				Instances neighbours = search.kNearestNeighbours(inst,Math.min(kOption.getValue(),this.window.numInstances()));
				//================== Regression ====================
				if(inst.classAttribute().isNumeric()){
					double[] result = new double[1];
					// For storing the sum of class values of all the k nearest neighbours
					double sum = 0;
					// For storing the number of the nearest neighbours
					int num = neighbours.numInstances();
					//================== Median ====================
					if(medianOption.isSet()){
						// For storing every neighbour's class value
						double[] classValues = new double[num];

						for(int i=0;i<num;i++){
							classValues[i] = neighbours.instance(i).classValue();
						}
						// Sort the class values
						Arrays.sort(classValues);
						// Assign the median value into result
						if(classValues.length%2==1){
							result[0] = classValues[num/2];
						}else{
							result[0] = (classValues[num/2 - 1] + classValues[num/2]) / 2;
						}
						return result;
						//=============== End of Median ============
					}else{
						//================== Mean ==================
						for(int i=0;i<num;i++){
							sum += neighbours.instance(i).classValue();
						}
						// Calculate the mean of all k nearest neighbours' class values
						result[0] = sum / num;
						return result;
						//=============== End of Mean ==============
					}
					//============= End of Regression ==============
				}else{
					for (int i = 0; i < neighbours.numInstances(); i++) {
						v[(int) neighbours.instance(i).classValue()]++;
					}
				}
			}
		} catch(Exception e) {
			//System.err.println("Error: kNN search failed.");
			//e.printStackTrace();
			//System.exit(1);
			return new double[inst.numClasses()];
		}
		return v;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    public boolean isRandomizable() {
        return false;
    }
}