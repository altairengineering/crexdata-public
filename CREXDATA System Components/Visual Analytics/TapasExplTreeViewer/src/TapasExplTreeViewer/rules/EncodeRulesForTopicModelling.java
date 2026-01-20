package TapasExplTreeViewer.rules;

import TapasDataReader.CommonExplanation;
import TapasDataReader.ExplanationItem;
import TapasExplTreeViewer.ui.TMrulesDefineSettings;

import java.util.*;

public class EncodeRulesForTopicModelling {
  public static void encode (RuleSet ruleSet, String dataFolder) {
    // TreeMap to maintain unique sorted values for each key
    Map<String, TreeSet<Float>> uniqueSortedValues = new TreeMap<>();
    // TreeMap to maintain all values for each key (for sorting later)
    Map<String, List<Float>> allValues = new TreeMap<>();
    // processing sttrMinMax
    for (String key: ruleSet.attrMinMax.keySet()) {
      //float minMax[]=attrMinMax.get(key);
      // Initialize the data structures for each key if not already initialized
      uniqueSortedValues.putIfAbsent(key, new TreeSet<>());
      allValues.putIfAbsent(key, new ArrayList<>());
    }
    for (CommonExplanation rule: ruleSet.rules)
      for (ExplanationItem cond: rule.eItems)
        for (int i=0; i<cond.interval.length; i++)
          if (Double.isFinite(cond.interval[i])) {
            uniqueSortedValues.get(cond.attr).add((float)cond.interval[i]);
            allValues.get(cond.attr).add((float)cond.interval[i]);
          }

    // Sort all values lists for each key
    for (List<Float> valueList: allValues.values())
      Collections.sort(valueList);
    // Output the results
    System.out.println("Unique Sorted Values by Key:");
    for (Map.Entry<String, TreeSet<Float>> entry : uniqueSortedValues.entrySet()) {
      System.out.println("Key: " + entry.getKey() +
          ", NValues: " + entry.getValue().size() +
          ", Values: " + entry.getValue());
    }
    System.out.println("All Sorted Values by Key:");
    for (Map.Entry<String, List<Float>> entry : allValues.entrySet()) {
      System.out.println("Key: " + entry.getKey() +
          ", NValues: " + entry.getValue().size() +
          ", Values: " + entry.getValue());
    }
    // Defining intervals for feature discretization
    TMrulesDefineSettings tmRulesSettings=new TMrulesDefineSettings(ruleSet.rules,
        ruleSet.listOfFeatures,ruleSet.attrMinMax,uniqueSortedValues,allValues,
        dataFolder);
  }

}
