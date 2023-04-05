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
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.lazy.neighboursearch.*;
import moa.core.Measurement;
import org.apache.commons.collections.map.HashedMap;
import org.apache.commons.math3.geometry.spherical.twod.Edge;
import scala.Int;

import java.util.ArrayList;
import java.util.HashMap;
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
public class kNN_Centroid extends AbstractClassifier implements MultiClassClassifier, Regressor {

    private static final long serialVersionUID = 1L;

	public IntOption kOption = new IntOption( "k", 'k', "The number of neighbors", 1, 1, 1);

	public IntOption limitOption = new IntOption( "WindowSize", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

	public FlagOption debugOption = new FlagOption("debug", 'd',"Debug");

	int C = 0;

    @Override
    public String getPurposeString() {
        return "kNN: special.";
    }

    protected Instances window;

	protected Map<Double, List<Double>> attr_sums;
	protected Map<Double, Integer> class_sums;

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			this.window = new Instances(context,0); //new StringReader(context.toString())
			this.window.setClassIndex(context.classIndex());
			this.attr_sums = new HashMap<Double, List<Double>>();
			this.class_sums = new HashMap<Double, Integer>();
		} catch(Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

    @Override
    public void resetLearningImpl() {
		if(this.debugOption.isSet()) System.out.println("kNN_Centroid.resetLearningImpl()");
		this.window = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {

		//if(this.debugOption.isSet()) System.out.println("kNN_Centroid.trainOnInstanceImpl()");

		if (inst.classValue() > C)
			C = (int)inst.classValue();
		if (this.window == null) {
			this.window = new Instances(inst.dataset());
		}
		if (this.limitOption.getValue() <= this.window.numInstances()) {
			removeFromSum(this.window.get(0));
			this.window.delete(0);
		}
		this.window.add(inst);
		addToSum(inst);

		if(this.debugOption.isSet()){
			System.out.println("Weight: " + inst.weight());
			// Print map sum of all attributes for each class
			StringBuilder sb = new StringBuilder();
			for (Map.Entry<Double, List<Double>> entry : this.attr_sums.entrySet()) {
				double key = entry.getKey();
				List<Double> values = entry.getValue();
				sb.append("Class: ").append(key).append(", Summed Attributes: [");
				for (int i = 0; i < values.size(); i++) {
					sb.append(values.get(i));
					if (i < values.size() - 1) {
						sb.append(", ");
					}
				}
				sb.append("]\n");
			}
			// Print the map contents using a single println statement
			System.out.println("Map: ");
			System.out.println(sb.toString());

			for (Map.Entry<Double, Integer> entry : this.class_sums.entrySet()) {
				double key = entry.getKey();
				int sum = entry.getValue();
				System.out.println("Class: " + key + ", Summed Instances: " + sum);
			}
			System.out.println("--------------------------");
		}

    }

	public void addToSum(Instance inst) {

		// Create new array of attributes if this class key is null
		this.attr_sums.computeIfAbsent(inst.classValue(), k -> new ArrayList<Double>());
		// Set sum of classes to 0 if this class key is null
		this.class_sums.putIfAbsent(inst.classValue(), 0);
		// Get attributes for current class
		List<Double> sum_attrs = this.attr_sums.get(inst.classValue());
		// Increment class sum for current class
		this.class_sums.merge(inst.classValue(), 1, Integer::sum);
		// Loop through all attributes
		for(int i = 0; i < inst.numAttributes(); i++){
			// If not attribute exists, add current instance attribute val
			if(sum_attrs.size() == i){
				sum_attrs.add(inst.valueSparse(i));
			}
			// Otherwise add current instance to attribute sum
			else{
				sum_attrs.set(i, sum_attrs.get(i) + inst.valueSparse(i));
			}
		}

		if(debugOption.isSet()) System.out.println("Attributes ADDED " + (this.window.size() - 1) + " -> " + this.window.size() );

	}

	public void removeFromSum(Instance inst) {

		this.class_sums.merge(inst.classValue(), -1, Integer::sum);

		List<Double> sum_attrs = this.attr_sums.get(inst.classValue());

		if( sum_attrs == null) return;

		for(int i = 0; i < inst.numAttributes(); i++){

			sum_attrs.set(i, sum_attrs.get(i) - inst.valueSparse(i));

		}

		if(debugOption.isSet()) System.out.println("Attributes REMOVED " + this.window.size() + " -> " + (this.window.size() - 1) );
	}

	@Override
    public double[] getVotesForInstance(Instance inst) {

		//if(this.debugOption.isSet()) System.out.println("kNN_Centroid.getVotesForInstance()");

		DistanceFunction m_DistanceFunction = new EuclideanDistance();
		m_DistanceFunction.setInstances(this.window);
		double v[] = new double[C+1];
		Instance temp;
		double min_distance = 0.0;
		double min_class = 0.0;
		try {

			System.out.println("Num vals : " + inst.numValues());

//			for (double key : this.attr_sums.keySet()) {
//				double[] attrs = averageAttributes(this.attr_sums.get(inst.classValue()), this.class_sums.get(inst.classValue()));
//				temp = new InstanceImpl(0.0, attrs, new int[] {}, attrs.length);
//				//distance = this.m_EuclideanDistance.distance(target, m_Instances.instance(m_InstList[idx]), Double.POSITIVE_INFINITY);
//				double d = m_DistanceFunction.distance(inst, temp);
//				if( d < min_distance ) min_class = key;
//			}
//
//			v[(int)min_class]++;


		} catch(Exception e) {
			System.err.println("Error: kNN search failed.");
			e.printStackTrace();
			//System.exit(1);
			return new double[inst.numClasses()];
		}
		return v;
    }

	public double[] averageAttributes(List<Double> summed_attributes, int summed_class){

		double[] avg_atr = new double[summed_attributes.size()];

		for( int i = 0; i < summed_attributes.size(); i++ ){
			avg_atr[i] = summed_attributes.get(i) / summed_class;
		}

		return avg_atr;
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