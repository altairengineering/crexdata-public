import TapasDataReader.Flight;
import TapasDataReader.Record;

import java.util.Hashtable;
import java.util.TreeSet;
import java.util.Vector;

public class Main {

  public static void main(String[] args) {
    //System.out.println("Hello World!");
    String path="C:\\CommonGISprojects\\tracks-avia\\TAPAS\\ATFCM-20210331\\0_delays\\",
           fnCapacities=path+"scenario_20190801_capacities",
           fnDecisions=path+"scenario_20190801_exp0_decisions",
           fnFlightPlans=path+"scenario_20190801_exp0_baseline_flight_plans";
    TreeSet<Integer> steps=TapasDataReader.Readers.readStepsFromDecisions(fnDecisions);
    System.out.println(steps);
    Hashtable<String,Flight> flights=TapasDataReader.Readers.readFlightDelaysFromDecisions(fnDecisions,steps);
    System.out.println(flights.get("EDDK-LEPA-EWG598-20190801083100").delays[2]);
    Hashtable<String,Vector<Record>> records=TapasDataReader.Readers.readFlightPlans(fnFlightPlans,flights);
    Hashtable<String,float[]> attrs=new Hashtable<String, float[]>();
    TapasDataReader.Readers.readExplanations(path,steps,flights,attrs);
    for (String s:attrs.keySet())
      System.out.println(s+": ["+attrs.get(s)[0]+".."+attrs.get(s)[1]+"]");
  }
}
