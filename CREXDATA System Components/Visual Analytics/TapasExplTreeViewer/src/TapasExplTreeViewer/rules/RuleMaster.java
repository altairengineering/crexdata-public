package TapasExplTreeViewer.rules;

import TapasDataReader.CommonExplanation;
import TapasDataReader.Explanation;
import TapasDataReader.ExplanationItem;
import TapasExplTreeViewer.clustering.ObjectWithMeasure;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;

/**
 * Intended to contain methods for minimisation and aggregation of a set of rules (or explanations)
 */
public class RuleMaster {
  /**
   * Checks if one rule (explanation) subsumes another. If so, returns the more general rule;
   * otherwise, returns null.
   * If two rules differ in the action (decision, prediction), returns null without checking the conditions.
   */
  public static UnitedRule selectMoreGeneral(CommonExplanation ex1, CommonExplanation ex2) {
    if (ex1==null || ex2==null)
      return null;
    if (ex1.action!=ex2.action)
      return null;
    if (ex1.subsumes(ex2))
      return putTogether(ex1,ex2);
    if (ex2.subsumes(ex1))
      return putTogether(ex2,ex1);
    return null;
  }
  
  /**
   * Creates a new explanation with the same conditions as in ex1 (which is treated as more general)
   * summarizing information about the uses of ex1 and ex2.
   * @param ex1 - more general explanation
   * @param ex2 - less general information
   * @return more general explanation with summarized uses
   */
  public static UnitedRule putTogether(CommonExplanation ex1, CommonExplanation ex2) {
    if (ex1==null || ex2==null)
      return null;
    UnitedRule rule= new UnitedRule();
    rule.action=ex1.action;
    rule.nUses=ex1.nUses+ex2.nUses;
    rule.weight=ex1.weight+ex2.weight;
    rule.treeId=ex1.treeId;
    rule.treeCluster=ex1.treeCluster;
    if (ex1.uses!=null || ex2.uses!=null) {
      int n1=(ex1.uses!=null)?ex1.uses.size():0, n2=(ex2.uses!=null)?ex2.uses.size():0;
      rule.uses = new Hashtable<String, ArrayList<Explanation>>(n1+n2);
      if (n1>0)
        rule.uses.putAll(ex1.uses);
      if (n2>0)
        rule.uses.putAll(ex2.uses);
      //if (rule.uses.size()<n1+n2)  //there are common uses of the original rules
        //rule.nUses-=n1+n2-rule.uses.size();
    }
    rule.eItems=CommonExplanation.makeCopy(ex1.eItems);
    /*
    rule.fromRules=new ArrayList<UnitedRule>(10);
    rule.fromRules.add(UnitedRule.getRule(ex1));
    rule.fromRules.add(UnitedRule.getRule(ex2));
    for (int i=0; i<rule.fromRules.size(); i++) {
      rule.nOrigRight+=rule.fromRules.get(i).nOrigRight;
      rule.nOrigWrong+=rule.fromRules.get(i).nOrigWrong;
    }
    */
  
    rule.minQ=Math.min(ex1.minQ,ex2.minQ);
    rule.maxQ=Math.max(ex1.maxQ,ex2.maxQ);
    rule.sumQ=ex1.sumQ+ex2.sumQ;
    rule.meanQ=(float)(rule.sumQ/rule.nUses);

    return rule;
  }

  /**
   * Removes contradictory rules, i.e., rules whose conditions cover conditions of at least one rule
   * with a different outcome. That is, the more general of the two rules is removed.
   * If two rules have coinciding conditions but different outcomes, both are removed.
   */
  public static ArrayList<CommonExplanation> removeContradictory(ArrayList<CommonExplanation> origRules,
                                                                 boolean useQ, double maxQDiff) {
    if (origRules==null || origRules.size()<2)
      return origRules;
    ArrayList<CommonExplanation> rules=new ArrayList<CommonExplanation>(origRules.size());
    rules.addAll(origRules);
    boolean removed;
    do {
      removed=false;
      int prevSize=rules.size();
      for (int i=0; i<rules.size()-1 && !removed; i++) {
        CommonExplanation r1=rules.get(i);
        for (int j = rules.size()-1; j>i && !removed; j--) {
          CommonExplanation r2=rules.get(j);
          boolean sameResult=(useQ)?Math.abs(r1.minQ-r2.minQ)<maxQDiff && Math.abs(r1.maxQ-r2.maxQ)<maxQDiff:
              r1.action==r2.action;
          if (sameResult) {
            if (CommonExplanation.sameExplanations(r1,r2))
              rules.remove(j);
          }
          else
          if (CommonExplanation.sameExplanations(r1,r2)) {
            rules.remove(j); rules.remove(i); removed=true;
          }
          else
          if (r1.subsumes(r2,false)) {
            rules.remove(i); removed=true;
          }
          else
          if (r2.subsumes(r1,false)) {
            rules.remove(j); removed=true;
          }
        }
      }
      if (removed) {
        System.out.println("Removed "+(prevSize-rules.size())+" rules in one step, "+
            (origRules.size()-rules.size())+" rules removed in total.");
      }
    } while (removed);
    if (rules.size()==origRules.size())
      return origRules;
    for (int i=0; i<rules.size(); i++) {
      UnitedRule r=UnitedRule.getRule(rules.get(i));
      rules.set(i,r);
      if (useQ)
        r.countRightAndWrongCoveragesByQ(rules);
      else
        r.countRightAndWrongCoverages(rules);
    }
    return rules;
  }

  /**
   * Removes explanations (or rules) covered by other rules with the same actions
   * whose conditions are more general.
   * @return reduced set of explanations (rules).
   */
  public static ArrayList<CommonExplanation> removeLessGeneral(ArrayList<CommonExplanation> rules,
                                                               ArrayList<CommonExplanation> origRules,
                                                               Hashtable<String,float[]> attrMinMax,
                                                               boolean useQ, double maxQDiff) {
    if (rules==null || rules.size()<2)
      return rules;
    if (attrMinMax!=null) {
      ArrayList<CommonExplanation> rules2=null;
      for (int i=0; i<rules.size(); i++) {
        CommonExplanation ex= rules.get(i), ex2=UnitedRule.adjustToFeatureRanges(ex,attrMinMax);
        if (!ex2.equals(ex)) {
          if (rules2==null) {
            rules2=new ArrayList<CommonExplanation>(rules.size());
            for (int j=0; j<i; j++)
              rules2.add(rules.get(j));
          }
          rules2.add(ex2);
        }
        else
          if (rules2!=null)
            rules2.add(ex);
      }
      if (rules2!=null)
        rules=rules2;
    }
    for (int i=0; i<rules.size(); i++) {
      CommonExplanation ex= rules.get(i);
      UnitedRule r=UnitedRule.getRule(ex);
      if (useQ)
        r.countRightAndWrongCoveragesByQ(origRules);
      else
        r.countRightAndWrongCoverages(origRules);
      if (!(ex instanceof UnitedRule)) {
        rules.remove(i);
        rules.add(i,r);
      }
    }
    
    ArrayList<CommonExplanation> moreGeneral=new ArrayList<CommonExplanation>(rules.size());
    boolean removed[]=new boolean[rules.size()];
    for (int i=0; i<removed.length; i++)
      removed[i]=false;
    for (int i=0; i<rules.size()-1; i++)
      if (!removed[i]) {
        CommonExplanation ex= rules.get(i);
        for (int j = i + 1; j < rules.size(); j++)
          if (!removed[j]) {
            UnitedRule gEx = selectMoreGeneral(ex, rules.get(j));
            if (gEx != null) {
              if (useQ && gEx.maxQ-gEx.minQ>maxQDiff);
              else {
                removed[i] = removed[j] = true;
                ex = gEx;
              }
            }
          }
        if (removed[i]) {
          for (int j=moreGeneral.size()-1; j>=0; j--) {
            UnitedRule gEx = selectMoreGeneral(ex, moreGeneral.get(j));
            if (gEx!=null) {
              if (useQ && gEx.maxQ-gEx.minQ>maxQDiff);
              else {
                moreGeneral.remove(j);
                ex = gEx;
              }
            }
          }
          moreGeneral.add(UnitedRule.getRule(ex));
        }
      }
    if (moreGeneral.isEmpty())
      return rules; //nothing reduced
    boolean changed;
    do {
      changed=false;
      for (int i = 0; i < moreGeneral.size(); i++)
        for (int j = 0; j < rules.size(); j++)
          if (!removed[j]){
            UnitedRule gEx=selectMoreGeneral(moreGeneral.get(i),rules.get(j));
            if (gEx!=null) {
              if (useQ && gEx.maxQ-gEx.minQ>maxQDiff);
              else {
                moreGeneral.add(i, gEx);
                moreGeneral.remove(i + 1);
                removed[j] = true;
                changed = true;
              }
            }
          }
    } while (changed);
    for (int i=0; i<rules.size(); i++)
      if (!removed[i])
        moreGeneral.add(UnitedRule.getRule(rules.get(i)));
      else
        for (CommonExplanation gen:moreGeneral)
          if (gen.subsumes(rules.get(i))) {
            UnitedRule genR=(UnitedRule.getRule(gen));
            if (genR.equals(rules.get(i)))
              break;
            if (genR.fromRules==null)
              genR.fromRules=new ArrayList<UnitedRule>(50);
            genR.fromRules.add(UnitedRule.getRule(rules.get(i)));
            break;
          }
  
    boolean noActions=noActionDifference(rules);
    for (int i=0; i<moreGeneral.size(); i++)
      if (noActions)
        ((UnitedRule)moreGeneral.get(i)).countRightAndWrongCoveragesByQ((origRules!=null)?origRules:rules);
      else
        ((UnitedRule)moreGeneral.get(i)).countRightAndWrongCoverages((origRules!=null)?origRules:rules);
    return  moreGeneral;
  }
  
  public static ArrayList<String> getListOfFeatures (ArrayList<CommonExplanation> rules) {
    Hashtable<String,Integer> attrUses=new Hashtable<String,Integer>(50);
    for (int i=0; i<rules.size(); i++) {
      CommonExplanation cEx=rules.get(i);
      for (int j = 0; j < cEx.eItems.length; j++) {
        Integer count=attrUses.get(cEx.eItems[j].attr);
        if (count==null)
          attrUses.put(cEx.eItems[j].attr,1);
        else
          attrUses.put(cEx.eItems[j].attr,count+1);
      }
    }
    if (attrUses.isEmpty())
      return null;
    ArrayList<String> listOfFeatures=new ArrayList<String>(attrUses.size());
    for (Map.Entry<String,Integer> entry:attrUses.entrySet()) {
      String aName=entry.getKey();
      if (listOfFeatures.isEmpty())
        listOfFeatures.add(aName);
      else {
        int count=entry.getValue(), idx=-1;
        for (int i=0; i<listOfFeatures.size() && idx<0; i++)
          if (count>attrUses.get(listOfFeatures.get(i)))
            idx=i;
        if (idx<0)
          listOfFeatures.add(aName);
        else
          listOfFeatures.add(idx,aName);
      }
    }
    return listOfFeatures;
  }

  /**
   * Selects the rules satisfying query conditions. For integers, -1 means that the limit is not set.
   * For doubles, Double.NaN means that the condition is not set.
   */
  public static ArrayList<CommonExplanation> selectByQuery(ArrayList<CommonExplanation> rules,
                                                            int minAction, int maxAction,
                                                            double minValue, double maxValue,
                                                            int minTreeId, int maxTreeId,
                                                            int minTreeCluster, int maxTreeCluster,
                                                            Vector<String> categories) {
    if (rules==null || rules.isEmpty())
      return rules;
    if (categories==null && minAction<0 && maxAction<0 &&
        minTreeId<0 && maxTreeId<0 && minTreeCluster<0 && maxTreeCluster<0 &&
        Double.isNaN(minValue) && Double.isNaN(maxValue))
      return rules; //no query conditions
    ArrayList<CommonExplanation> subset=null;
    for (CommonExplanation r:rules) {
      if (categories!=null && !categories.contains(r.category))
        continue;
      if (minAction>=0 && r.action<minAction)
          continue;
      if (maxAction>=0 && r.action>maxAction)
        continue;
      if (!Double.isNaN(minValue) && r.maxQ<minValue)
        continue;
      if (!Double.isNaN(maxValue) && r.minQ>maxValue)
        continue;
      if (minTreeId>=0 && r.treeId<minTreeId)
        continue;
      if (maxTreeId>=0 && r.treeId>maxTreeId)
        continue;
      if (minTreeCluster>=0 && r.treeCluster<minTreeCluster)
        continue;
      if (maxTreeCluster>=0 && r.treeCluster>maxTreeCluster)
        continue;
      if (subset==null)
        subset=new ArrayList<CommonExplanation>(rules.size()/10);
      subset.add(r);
    }
    return subset;
  }

  public static boolean noActionDifference(ArrayList rules) {
    if (rules==null || rules.size()<2 || !(rules.get(0) instanceof CommonExplanation))
      return true;
    CommonExplanation ex=(CommonExplanation)rules.get(0);
    for (int i=1; i<rules.size(); i++)
      if (ex.action!=((CommonExplanation)rules.get(i)).action)
        return false;
    return true;
  }
  
  public static double suggestMaxQDiff(ArrayList rules) {
    if (rules==null || rules.size()<3 || !(rules.get(0) instanceof CommonExplanation))
      return Double.NaN;
    ArrayList<Double> qList=new ArrayList<Double>(rules.size());
    for (int i=0; i<rules.size(); i++) {
      CommonExplanation ex=(CommonExplanation)rules.get(i);
      if (!Double.isNaN(ex.meanQ))
        qList.add(new Double(ex.meanQ));
    }
    if (qList.size()<3)
      return Double.NaN;
    Collections.sort(qList);
    ArrayList<Double> qDiffList=new ArrayList<Double>(qList.size()-1);
    for (int i=0; i<qList.size()-1; i++)
      qDiffList.add(qList.get(i+1)-qList.get(i));
    Collections.sort(qDiffList);
    int idx=Math.round(0.025f*qDiffList.size());
    if (idx<5)
      idx=Math.round(0.05f*qDiffList.size());
    if (idx<5)
      idx=Math.max(Math.round(0.1f*qDiffList.size()),3);
    double maxQDiff=qDiffList.get(idx);
    if (maxQDiff==0)
      maxQDiff=qDiffList.get(qDiffList.size()-1)/10;
    return maxQDiff;
  }
  
  /**
   * Aggregates rules with coinciding actions by bottom-up
   * hierarchical joining of the closest rules. The accuracy of the united rules
   * is checked against the set of original explanations (rules). If the accuracy
   * (a number from 0 to 1) is less than minAccuracy, the united rule is discarded
   * and further aggregation is stopped.
   */
  public static ArrayList<UnitedRule> aggregate(ArrayList<UnitedRule> rules,
                                                ArrayList<CommonExplanation> origRules,
                                                double minAccuracy,
                                                Hashtable<String,float[]> attrMinMax) {
    return aggregate(rules,origRules,null,minAccuracy,attrMinMax,null);
  }
  
  /**
   * Aggregates rules with coinciding actions by bottom-up
   * hierarchical joining of the closest rules.
   * The accuracy of the united rules is checked against the set of original explanations (rules).
   * If the set of data instances exData is not null, the accuracy is also checked against the data.
   * From two accuracy estimates, the smaller one is taken.
   * If the accuracy (a number from 0 to 1) is less than minAccuracy, the united rule is discarded
   * and further aggregation is stopped.
   */
  public static ArrayList<UnitedRule> aggregate(ArrayList<UnitedRule> rules,
                                                ArrayList<CommonExplanation> origRules,
                                                AbstractList<Explanation> exData,
                                                double minAccuracy,
                                                Hashtable<String,float[]> attrMinMax,
                                                ChangeListener listener) {
    if (rules==null || rules.size()<2)
      return rules;
    ArrayList<ArrayList<UnitedRule>> ruleGroups=new ArrayList<ArrayList<UnitedRule>>(rules.size()/2);
    ArrayList<UnitedRule> agRules=new ArrayList<UnitedRule>(ruleGroups.size()*2);
    for (int i=0; i<rules.size(); i++) {
      UnitedRule rule=rules.get(i);
      rule.countRightAndWrongCoverages(origRules);
      if (minAccuracy>0 && exData != null && rule.nCasesRight + rule.nCasesWrong < 1)
        rule.countRightAndWrongApplications(exData, true);
      
      boolean added=false;
      for (int j=0; j<ruleGroups.size() && !added; j++) {
        UnitedRule rule2=ruleGroups.get(j).get(0);
        if (rule.action==rule2.action) {
          ruleGroups.get(j).add(rule);
          added=true;
        }
      }
      if (!added) {
        ArrayList<UnitedRule> group=new ArrayList<UnitedRule>(100);
        group.add(rule);
        ruleGroups.add(group);
      }
    }
    if (ruleGroups.size()==rules.size())
      return rules;
    
    if (ruleGroups.size()<2)  //handle the case of no actions
      agRules=aggregateByQ(ruleGroups.get(0), suggestMaxQDiff(rules),origRules,exData,minAccuracy,attrMinMax);
    else {
      SwingWorker workers[]=(listener==null)?null:new SwingWorker[ruleGroups.size()];
      for (int ig=0; ig<ruleGroups.size(); ig++) {
        if (workers!=null)
          workers[ig]=null;
        ArrayList<UnitedRule> group=ruleGroups.get(ig);
        if (group.size()==1) {
          agRules.add(group.get(0));
          continue;
        }
        if (workers!=null) {
          final ArrayList<UnitedRule> finAgRules = agRules;
          workers[ig] = new SwingWorker() {
            @Override
            public Boolean doInBackground() {
              aggregateGroup(group, origRules, exData, minAccuracy, attrMinMax);
              return true;
            }

            @Override
            protected void done() {
              finAgRules.addAll(group);
              if (listener != null)
                listener.stateChanged(new ChangeEvent(finAgRules));
            }
          };
          workers[ig].execute();
        }
        else {
          aggregateGroup(group, origRules, exData, minAccuracy, attrMinMax);
          agRules.addAll(group);
          if (listener != null)
            listener.stateChanged(new ChangeEvent(agRules));
        }
      }
      if (workers!=null) {
        // Wait for all workers to finish
        SwingWorker resultAggregator = new SwingWorker() {
          @Override
          protected Void doInBackground() throws Exception {
            for (SwingWorker worker : workers) {
              if (worker!=null)
                worker.get(); // Wait for each worker to finish
            }
            return null;
          }

          @Override
          protected void done() {
            if (listener != null)
              listener.stateChanged(new ChangeEvent("aggregation_finished"));
          }
        };
        resultAggregator.execute();
      }
    }
    return (agRules.size()<rules.size())?agRules:rules;
  }
  
  public static void aggregateGroup(ArrayList<UnitedRule> group,
                                    ArrayList<CommonExplanation> origRules,
                                    AbstractList<Explanation> exData,
                                    double minAccuracy,
                                    Hashtable<String,float[]> attrMinMax) {
    if (group==null || group.size()<2)
      return;
    int origSize=group.size();
    System.out.println("Aggregating group with "+origSize+" rules; action or class = "+group.get(0).action);
    ArrayList<UnitedRule> notAccurate=(minAccuracy>0)?new ArrayList<UnitedRule>(group.size()):null;
    if (minAccuracy>0)
      for (int i=group.size()-1; i>=0; i--) {
        UnitedRule r=group.get(i);
        double dataAcc=(exData==null)?1:(1.0*r.nCasesRight/(r.nCasesRight+r.nCasesWrong));
        if (dataAcc<minAccuracy || getAccuracy(r,origRules,false)<minAccuracy) {
          group.remove(i);
          notAccurate.add(r);
        }
      }
    boolean united;
    ArrayList<ObjectWithMeasure> pairs=new ArrayList<ObjectWithMeasure>(group.size()*2);
    System.out.println("Computing pairwise distances");
    for (int i=0; i<group.size()-1; i++)
      for (int j=i+1; j<group.size(); j++)
        if (/*UnitedRule.sameFeatures(group.get(i),group.get(j))*/true) {
          double d=UnitedRule.distance(group.get(i),group.get(j),attrMinMax);
          int pair[]={i,j};
          ObjectWithMeasure om=new ObjectWithMeasure(pair,d,false);
          pairs.add(om);
        }
    System.out.println("Sorting "+pairs.size()+" pairs by distances");
    Collections.sort(pairs);
    System.out.println("Sorted "+pairs.size()+" pairs!");

    HashSet<Integer> excluded=new HashSet<Integer>(group.size()*10);

    int nUnions=0, nExcluded=0;
    Hashtable<Integer,HashSet<Integer>> failedPairs=new Hashtable<Integer,HashSet<Integer>>(group.size()*5);
    do {
      united=false;
      UnitedRule union=null;
      //pairs.clear();
      for (int i=0; i<pairs.size() && !united; i++) {
        ObjectWithMeasure om=pairs.get(i);
        int pair[]=(int[])om.obj;
        int i1=pair[0], i2=pair[1];
        if (excluded.contains(i1) || excluded.contains(i2))
          continue;
        HashSet<Integer> failed=failedPairs.get(i1);
        if (failed!=null && failed.contains(i2))
          continue;
        union=UnitedRule.unite(group.get(i1),group.get(i2),attrMinMax);
        boolean success=union!=null;
        if (success) {
          union.countRightAndWrongCoverages(origRules);
          if (union.nOrigRight<1)
            System.out.println("Zero coverage!");
          if (minAccuracy>0) {
            if (getAccuracy(union, origRules, false)<minAccuracy)
              success=false;
            else
              if (exData!=null) { //check the accuracy based on the data
                union.countRightAndWrongApplications(exData, true);
                if (1.0*union.nCasesRight/(union.nCasesRight+union.nCasesWrong)<minAccuracy)
                  success=false;
              }
          }
          if (!success) {
            if (failed==null) {
              failed=new HashSet<Integer>(origSize*5);
              failedPairs.put(i1,failed);
            }
            failed.add(i2);
            continue;
          }
          //group.remove(i2);
          //group.remove(i1);
          excluded.add(i1);
          excluded.add(i2);
          nExcluded+=2;
          if (failedPairs!=null) {
            failedPairs.remove(i1); //no more needed
            failedPairs.remove(i2); //no more needed
          }
          for (int j=group.size()-1; j>=0; j--)
            if (!excluded.contains(j) && union.subsumes(group.get(j),true)) {
              union.attachAsFromRule(group.get(j));
              if (minAccuracy>0 && exData!=null)  //check the accuracy based on the data
                union.countRightAndWrongApplications(exData,true);
              //group.remove(j);
              excluded.add(j);
              ++nExcluded;
              if (failedPairs!=null)
                failedPairs.remove(j);
            }
          group.add(union);
          united=true;
          ++nUnions;
        }
        else {
          if (failed==null) {
            failed=new HashSet<Integer>(origSize*5);
            failedPairs.put(i1,failed);
          }
          failed.add(i2);
        }
      }
      if (united) {
        for (int i = pairs.size() - 1; i > 0; i--) {
          ObjectWithMeasure om = pairs.get(i);
          int pair[] = (int[]) om.obj;
          int i1 = pair[0], i2 = pair[1];
          if (excluded.contains(i1) || excluded.contains(i2))
            pairs.remove(i);
        }
        int nRemain=1;
        for (int i=0; i<group.size()-1; i++)
          if (!excluded.contains(i)) {
            ++nRemain;
            double d=UnitedRule.distance(group.get(i),union,attrMinMax);
            int pair[]={i,group.size()-1};
            ObjectWithMeasure om=new ObjectWithMeasure(pair,d,false);
            pairs.add(om);
          }
        //System.out.println("Sorting "+pairs.size()+" pairs by distances");
        Collections.sort(pairs);
        //System.out.println("Sorted "+pairs.size()+" pairs!");
        if (nUnions%10==0) {
          System.out.println("Aggregation: made "+nUnions+" unions; excluded "+nExcluded+" rules; "+
              nRemain+" rules remain; group size = "+group.size());
        }
      }
    } while (united /*&& group.size()>1*/ && pairs.size()>0);

    for (int i=group.size()-1; i>=0; i--)
      if (excluded.contains(i))
        group.remove(i);

    if (notAccurate!=null && !notAccurate.isEmpty())
      group.addAll(notAccurate);

    System.out.println("Finished group aggregation attempt; action or class = "+group.get(0).action+
        "; original size = "+origSize+"; final size = "+group.size());
  }
  /**
   * Aggregates rules with close real-valued outcomes (represented by variable Q) by bottom-up
   * hierarchical joining of the closest rules. Rule results are considered close if their difference
   * does not exceed maxQDiff.
   * The accuracy of the united rules is checked against the set of original explanations (rules).
   * If the accuracy (a number from 0 to 1) is less than minAccuracy, the united rule is discarded
   * and further aggregation is stopped.
   */
  public static ArrayList<UnitedRule> aggregateByQ(ArrayList<UnitedRule> rules,
                                                   double maxQDiff,
                                                   ArrayList<CommonExplanation> origRules,
                                                   double minAccuracy,
                                                   Hashtable<String,float[]> attrMinMax) {
    return aggregateByQ(rules,maxQDiff,origRules,null,minAccuracy,attrMinMax);
  }
  
  /**
   * Aggregates rules with close real-valued outcomes (represented by variable Q) by bottom-up
   * hierarchical joining of the closest rules. Rule results are considered close if their difference
   * does not exceed maxQDiff.
   * The accuracy of the united rules is checked against the set of original explanations (rules).
   * If the set of data instances exData is not null, the accuracy is also checked against the data.
   * From two accuracy estimates, the smaller one is taken.
   * If the accuracy (a number from 0 to 1) is less than minAccuracy, the united rule is discarded
   * and further aggregation is stopped.
   */
  public static ArrayList<UnitedRule> aggregateByQ(ArrayList<UnitedRule> rules,
                                            double maxQDiff,
                                            ArrayList<CommonExplanation> origRules,
                                            AbstractList<Explanation> exData,
                                            double minAccuracy,
                                            Hashtable<String,float[]> attrMinMax) {
    if (rules==null || rules.size()<2)
      return rules;

    System.out.println("Aggregating rules by Q; max difference = "+maxQDiff);
    
    ArrayList<UnitedRule> result=new ArrayList<UnitedRule>(rules.size());
    result.addAll(rules);
    ArrayList<UnitedRule> notAccurate=(minAccuracy>0)?new ArrayList<UnitedRule>(result.size()):null;
    if (minAccuracy>0)
      for (int i=result.size()-1; i>=0; i--) {
        UnitedRule r=result.get(i);
        r.countRightAndWrongCoveragesByQ(origRules);
        if (minAccuracy>0 && exData!=null && r.nCasesRight+r.nCasesWrong<1)
          r.countRightAndWrongApplications(exData,false);
        double dataAcc=(exData==null)?1:(1.0*r.nCasesRight/(r.nCasesRight+r.nCasesWrong));
        if (dataAcc<minAccuracy || getAccuracy(r,origRules,true)<minAccuracy) {
          result.remove(i);
          notAccurate.add(r);
        }
      }

    ArrayList<ObjectWithMeasure> pairs=new ArrayList<ObjectWithMeasure>(result.size());
    System.out.println("Computing pairwise distances");
    for (int i=0; i<result.size()-1; i++) {
      UnitedRule r1=result.get(i);
      for (int j = i + 1; j < result.size(); j++) {
        UnitedRule r2=result.get(j);
        if (Math.max(r1.maxQ,r2.maxQ)-Math.min(r1.minQ,r2.minQ)>maxQDiff)
          continue;
        if (/*UnitedRule.sameFeatures(r1, r2)*/true) {
          double d = UnitedRule.distance(r1, r2, attrMinMax);
          int pair[] = {i, j};
          ObjectWithMeasure om = new ObjectWithMeasure(pair, d, false);
          pairs.add(om);
        }
      }
    }
    System.out.println("Sorting "+pairs.size()+" pairs by distances");
    Collections.sort(pairs);
    System.out.println("Sorted "+pairs.size()+" pairs!");

    HashSet<Integer> excluded=new HashSet<Integer>(result.size()*10);

    int nUnions=0, nExcluded=0;
    Hashtable<Integer,HashSet<Integer>> failedPairs=new Hashtable<Integer,HashSet<Integer>>(result.size()*5);

    boolean united;
    int origSize=result.size();
    do {
      united=false;
      UnitedRule union=null;
      for (int i=0; i<pairs.size() && !united; i++) {
        ObjectWithMeasure om=pairs.get(i);
        int pair[]=(int[])om.obj;
        int i1=pair[0], i2=pair[1];
        if (excluded.contains(i1) || excluded.contains(i2))
          continue;
        HashSet<Integer> failed=failedPairs.get(i1);
        if (failed!=null && failed.contains(i2))
          continue;
        //System.out.println("Trying to unite rules "+i1+" and "+i2);
        union=UnitedRule.unite(result.get(i1),result.get(i2),attrMinMax);
        boolean success=union!=null;
        if (success) {
          if (union.maxQ-union.minQ>maxQDiff)
            continue;
          union.countRightAndWrongCoveragesByQ(origRules);
          if (union.nOrigRight<1)
            System.out.println("Zero coverage!");
          if (minAccuracy>0) {
            if (getAccuracy(union, origRules, false)<minAccuracy)
              success=false;
            else
            if (exData!=null) { //check the accuracy based on the data
              union.countRightAndWrongApplications(exData, true);
              if (1.0*union.nCasesRight/(union.nCasesRight+union.nCasesWrong)<minAccuracy)
                success=false;
            }
          }
          if (!success) {
            //System.out.println("The union of "+i1+" and "+i2+" discarded due to low accuracy.");
            if (failed==null) {
              failed=new HashSet<Integer>(origSize*5);
              failedPairs.put(i1,failed);
            }
            failed.add(i2);
            continue;
          }
          //System.out.println("Successfully united rules "+i1+" and "+i2+"!");
          excluded.add(i1);
          excluded.add(i2);
          nExcluded+=2;
          if (failedPairs!=null) {
            failedPairs.remove(i1); //no more needed
            failedPairs.remove(i2); //no more needed
          }
          for (int j=result.size()-1; j>=0; j--)
            if (!excluded.contains(j)) {
              UnitedRule r2 = result.get(j);
              if (r2.minQ >= union.minQ && r2.maxQ <= union.maxQ &&
                      union.subsumes(r2)) {
                union.attachAsFromRule(r2);
                if (minAccuracy>0 && exData!=null)  //check the accuracy based on the data
                  union.countRightAndWrongApplications(exData,false);
                excluded.add(j);
                ++nExcluded;
                if (failedPairs!=null)
                  failedPairs.remove(j);
              }
            }
          result.add(union);
          united=true;
          ++nUnions;
        }
        else {
          //System.out.println("Failed to unite rules "+i1+" and "+i2+"...");
          if (failed==null) {
            failed=new HashSet<Integer>(origSize*5);
            failedPairs.put(i1,failed);
          }
          failed.add(i2);
          continue;
        }
      }
      if (united) {
        for (int i = pairs.size() - 1; i > 0; i--) {
          ObjectWithMeasure om = pairs.get(i);
          int pair[] = (int[]) om.obj;
          int i1 = pair[0], i2 = pair[1];
          if (excluded.contains(i1) || excluded.contains(i2))
            pairs.remove(i);
        }
        int nRemain=1;
        for (int i=0; i<result.size()-1; i++)
          if (!excluded.contains(i)) {
            ++nRemain;
            double d=UnitedRule.distance(result.get(i),union,attrMinMax);
            int pair[]={i,result.size()-1};
            ObjectWithMeasure om=new ObjectWithMeasure(pair,d,false);
            pairs.add(om);
          }
        //System.out.println("Sorting "+pairs.size()+" pairs by distances");
        Collections.sort(pairs);
        //System.out.println("Sorted "+pairs.size()+" pairs!");
        if (nUnions%5==0) {
          System.out.println("Aggregation: made "+nUnions+" unions; excluded "+nExcluded+" rules; "+
              nRemain+" rules remain; current total number of rules = "+result.size());
        }
      }
    } while (united && pairs.size()>1);

    for (int i=result.size()-1; i>=0; i--)
      if (excluded.contains(i))
        result.remove(i);

    int nResult=result.size();
    if (notAccurate!=null)
      nResult+=notAccurate.size();
    
    if (nResult>=rules.size()) //no aggregation was done
      return rules;
    ArrayList<CommonExplanation> aEx=new ArrayList<CommonExplanation>(result.size());
    aEx.addAll(result);
    aEx = removeLessGeneral(aEx, origRules, attrMinMax, true, maxQDiff);
    if (aEx.size()<result.size()) {
      result.clear();
      for (int i=0; i<aEx.size(); i++)
        result.add((UnitedRule)aEx.get(i));
    }
    if (notAccurate!=null && !notAccurate.isEmpty())
      for (int i=0; i<notAccurate.size(); i++)
        result.add(notAccurate.get(i));
    return result;
  }
  
  public static double getAccuracy(UnitedRule rule, ArrayList<CommonExplanation> origRules, boolean byQ) {
    if (rule==null || origRules==null)
      return Double.NaN;
    if (rule.nOrigRight<1)
      if (byQ)
        rule.countRightAndWrongCoveragesByQ(origRules);
      else
        rule.countRightAndWrongCoverages(origRules);
    if (rule.nOrigRight<1) {
      System.out.println("Zero coverage!");
      return Double.NaN; //must not happen; it means that this rule does not cover any of origRules!
    }
    return 1.0*rule.nOrigRight/(rule.nOrigRight+rule.nOrigWrong);
  }

  public static boolean hasRuleHierarchies(ArrayList<UnitedRule> rules) {
    if (rules==null || rules.isEmpty())
      return false;
    for (int i=0; i<rules.size(); i++) {
      ArrayList<UnitedRule> from=rules.get(i).fromRules;
      if (from!=null && !from.isEmpty())
        return true;
    }
    return false;
  }
  
  public static ArrayList<UnitedRule> expandRuleHierarchies(ArrayList<CommonExplanation> rules) {
    if (rules==null || rules.isEmpty() || !(rules.get(0) instanceof UnitedRule))
      return null;
    int lastUpId=rules.get(0).upperId;
    boolean wasExpanded=false;
    for (int i=1; i<rules.size() && !wasExpanded; i++)
      wasExpanded=lastUpId!=rules.get(i).upperId;
    if (wasExpanded) {
      ArrayList<CommonExplanation> rCopy = new ArrayList<CommonExplanation>(rules.size());
      for (int i=0; i<rules.size(); i++)
        rCopy.add(((UnitedRule)rules.get(i)).makeRuleCopy(false,false));
    }
    
    ArrayList<UnitedRule> result=new ArrayList<UnitedRule>(rules.size()*5);
    for (int i=0; i<rules.size(); i++)
      expandOneRuleHierarchy((UnitedRule)rules.get(i),result);
    return result;
  }
  
  public static void expandOneRuleHierarchy(UnitedRule rule, ArrayList<UnitedRule> result) {
    if (rule==null || result==null)
      return;
    rule.numId=result.size();
    result.add(rule);
    if (rule.nOrigRight<1)
      System.out.println("Zero coverage!");
    if (rule.fromRules!=null)
      for (int i=0; i<rule.fromRules.size(); i++) {
        rule.fromRules.get(i).upperId=rule.numId;
        expandOneRuleHierarchy(rule.fromRules.get(i),result);
      }
  }
  
  public static int findRuleInList(ArrayList rules, int ruleId){
    if (rules==null || rules.isEmpty())
      return -1;
    if (!(rules.get(0) instanceof CommonExplanation))
      return -1;
    for (int i=0; i<rules.size(); i++)
      if (((CommonExplanation)rules.get(i)).numId==ruleId)
        return i;
    return -1;
  }

  public static boolean exportRulesToFile(File file,
                                          ArrayList<CommonExplanation> rules) {
    if (file==null) {
      System.out.println("Rule export: no file!");
      return false;
    }
    if (rules==null || rules.isEmpty()) {
      System.out.println("Rule export: no rules!");
      return false;
    }
    boolean hasTreeId=rules.get((int)Math.floor(Math.random()*rules.size())).treeId>=0;
    boolean isRegression=!Float.isNaN(rules.get((int)Math.floor(Math.random()*rules.size())).minQ);
    boolean areUnited=rules.get(0) instanceof UnitedRule;

    boolean ok=false;
    String header="RuleID"+((hasTreeId)?",TreeID":"")+",Weight"+",Rule,"+
        ((isRegression)?((areUnited)?"Mean value,Min value,Max value":"Value"):"Class");

    try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
      writer.write(header);
      writer.newLine();
      for (CommonExplanation ce:rules) {
        writer.write(ce.numId+",");
        if (hasTreeId)
          writer.write(ce.treeId+",");
        writer.write(ce.weight+",");
        boolean first=true;
        for (ExplanationItem e:ce.eItems) {
          if (!first)
            writer.write("; ");
          else
            first=false;
          String left=(Double.isInfinite(e.interval[0])) ? "-inf" :
                          (e.isInteger)?String.format("%d",Math.round(e.interval[0])):Double.toString(e.interval[0]);
          String right=(Double.isInfinite(e.interval[1])) ? "inf" :
                          (e.isInteger)?String.format("%d",Math.round(e.interval[1])):Double.toString(e.interval[1]);
          writer.write(e.attr + ":{" + left + ":" + right + "}");
        }
        if (isRegression) {
          writer.write("," + ce.meanQ);
          if (areUnited)
            writer.write("," + ce.minQ + "," + ce.maxQ);
        }
        else
          writer.write(","+ce.action);
        writer.newLine();
      }
      ok=true;
    } catch (Exception ex) {
      ex.printStackTrace();
    }
    return ok;
  }

  
  public static boolean exportRulesToTable(File file,
                                           ArrayList<CommonExplanation> rules,
                                           Hashtable<String,float[]> attrMinMax) {
    if (file==null) {
      System.out.println("Rule export: no file!");
      return false;
    }
    if (rules==null || rules.isEmpty()) {
      System.out.println("Rule export: no rules!");
      return false;
    }
    Hashtable<String, Integer> attrUses=new Hashtable<String, Integer>(100);
    ArrayList<String> intFeatures=new ArrayList<String>(100);

    boolean hasUnitedRules=false;
    int minClass=-1, maxClass=-1;
    int maxRight=0, maxWrong=0;
    boolean rulesHaveQIntervals=false, rulesHaveUses=false, rulesHaveTreeIds=false;
    double qMin=Double.NaN, qMax=Double.NaN;
  
    for (int i=0; i<rules.size(); i++) {
      CommonExplanation cEx=rules.get(i);
      hasUnitedRules=hasUnitedRules || (cEx instanceof UnitedRule);
      rulesHaveUses=rulesHaveUses || (cEx.uses!=null && !cEx.uses.isEmpty());
      rulesHaveTreeIds = rulesHaveTreeIds || cEx.treeId>=0;

      if (cEx instanceof UnitedRule) {
        UnitedRule r=(UnitedRule)cEx;
        if (maxRight<r.nOrigRight) maxRight=r.nOrigRight;
        if (maxWrong<r.nOrigWrong) maxWrong=r.nOrigWrong;
      }

      for (int j=0; j<cEx.eItems.length; j++) {
        Integer count=attrUses.get(cEx.eItems[j].attr);
        if (count==null) {
          attrUses.put(cEx.eItems[j].attr, 1);
          if (cEx.eItems[j].isInteger)
            intFeatures.add(cEx.eItems[j].attr);
        }
        else
          attrUses.put(cEx.eItems[j].attr, count+1);
      }
      boolean qEqualsAction=false;
      boolean hasMinQ=!Float.isNaN(cEx.minQ), hasMaxQ=!Float.isNaN(cEx.maxQ);
      if (cEx.action >= 0) {
        if (minClass<0 || minClass>cEx.action)
          minClass=cEx.action;
        if (maxClass<cEx.action)
          maxClass=cEx.action;
        qEqualsAction=hasMinQ && hasMaxQ && cEx.minQ==cEx.action && cEx.maxQ==cEx.action;
      }
      if (hasMinQ && !qEqualsAction)
        if (Double.isNaN(qMin) || qMin>cEx.minQ)
          qMin=cEx.minQ;
      if (hasMaxQ && !qEqualsAction)
        if (Double.isNaN(qMax) || qMax<cEx.maxQ)
          qMax=cEx.maxQ;
      if (hasMinQ && hasMaxQ && Float.isNaN(cEx.meanQ)) {
        if (!Double.isNaN(cEx.sumQ)) {
          int count=(cEx instanceof UnitedRule) ? ((UnitedRule) cEx).countFromRules() : cEx.getUsesCount();
          if (count<1) count=1;
          cEx.meanQ=(float) (cEx.sumQ/count);
        }
        else
          cEx.meanQ=(cEx.minQ+cEx.maxQ)/2;
      }
      rulesHaveQIntervals=rulesHaveQIntervals || (!qEqualsAction && hasMinQ && hasMaxQ && cEx.maxQ>cEx.minQ);
    }
    ArrayList<String> usedFeatures=new ArrayList<String>(attrUses.size());
    ArrayList<String> binaryFeatures=(intFeatures.isEmpty())?null:new ArrayList<String>(intFeatures.size());

    for (Map.Entry<String,Integer> entry:attrUses.entrySet()) {
      String aName=entry.getKey();
      if (usedFeatures.isEmpty())
        usedFeatures.add(aName);
      else {
        int count = entry.getValue(), idx = -1;
        for (int i = 0; i < usedFeatures.size() && idx < 0; i++)
          if (count > attrUses.get(usedFeatures.get(i)))
            idx = i;
        if (idx < 0)
          usedFeatures.add(aName);
        else
          usedFeatures.add(idx, aName);
      }
      if (intFeatures.contains(aName) && attrMinMax!=null) {
        float minmax[]=attrMinMax.get(aName);
        if (Math.round(minmax[1]-minmax[0])==1)
          binaryFeatures.add(aName);
        else
          System.out.println("Feature "+aName+": ["+minmax[0]+".."+minmax[1]+"]");
      }
    }

    ArrayList<String> fieldNames=new ArrayList<String>(25+usedFeatures.size());
    fieldNames.add("RuleID");
    if (rulesHaveTreeIds)
      fieldNames.add("TreeID");
    if (maxClass>minClass || minClass>=0)
      fieldNames.add("Class");
    if (rulesHaveQIntervals) {
      fieldNames.add("Mean value");
      fieldNames.add("Min value");
      fieldNames.add("Max value");
      fieldNames.add("Range width");
    }
    else
      if (!Double.isNaN(qMin) && qMin<qMax)
        fieldNames.add("Value");
    fieldNames.add("Weight");
    if (rulesHaveUses)
      fieldNames.add("N uses");
    if (maxRight>0)
      fieldNames.add("N right covered");
    if (maxWrong>0) {
      fieldNames.add("N wrong covered");
      fieldNames.add("% coherence");
    }
    if (hasUnitedRules) {
      fieldNames.add("N united");
      fieldNames.add("Depth of hierarchy");
    }
    fieldNames.add("N conditions");

    int fIdx0=fieldNames.size();

    if (binaryFeatures!=null && !binaryFeatures.isEmpty())
      for (int i=0; i<usedFeatures.size(); i++)
        if (binaryFeatures.contains(usedFeatures.get(i)))
          fieldNames.add(usedFeatures.get(i));

    int nBinary=fieldNames.size()-fIdx0;
    int paramIdx0=-1, paramIdxLast=-1;
    if (usedFeatures.size()>nBinary) {
      paramIdx0=fieldNames.size();
      for (int i = 0; i < usedFeatures.size(); i++)
        if (nBinary < 1 || !binaryFeatures.contains(usedFeatures.get(i))) {
          fieldNames.add(usedFeatures.get(i) + ": min");
          fieldNames.add(usedFeatures.get(i) + ": max");
        }
      paramIdxLast=fieldNames.size();
    }

    boolean ok=false;
    try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
      writer.write(fieldNames.get(0));
      for (int i=1; i<fieldNames.size(); i++)
        writer.write(","+fieldNames.get(i));
      writer.newLine();
      for (int rIdx=0; rIdx<rules.size(); rIdx++) {
        CommonExplanation cEx=rules.get(rIdx);
        UnitedRule rule=(hasUnitedRules)?(UnitedRule)cEx:null;
        writer.write(Integer.toString(cEx.numId));
        for (int i=1; i<fIdx0; i++) {
          writer.write(",");
          String fName=fieldNames.get(i);
          if (fName.equals("TreeID"))
            writer.write(Integer.toString(cEx.treeId));
          else
          if (fName.equals("Class"))
            writer.write(Integer.toString(cEx.action));
          else
          if (fName.equals("Weight"))
            writer.write(Integer.toString(cEx.weight));
          else
          if (fName.equals("Value") || fName.equals("Mean value"))
            writer.write(String.format("%.6f",cEx.meanQ));
          else
          if (fName.equals("Min value"))
            writer.write(String.format("%.6f",cEx.minQ));
          else
          if (fName.equals("Max value"))
            writer.write(String.format("%.6f",cEx.maxQ));
          else
          if (fName.equals("Range width"))
            writer.write(String.format("%.6f",cEx.maxQ-cEx.minQ));
          else
          if (fName.equals("N uses"))
            writer.write((cEx.uses==null)?"0":Integer.toString(cEx.uses.size()));
          else
          if (fName.equals("N right covered"))
            writer.write(Integer.toString(rule.nOrigRight));
          else
          if (fName.equals("N wrong covered"))
            writer.write(Integer.toString(rule.nOrigWrong));
          else
          if (fName.equals("% coherence"))
            writer.write(String.format("%.2f", 100.0*rule.nOrigRight/(rule.nOrigWrong+rule.nOrigRight)));
          else
          if (fName.equals("N united"))
            writer.write((rule.fromRules==null)?"0":Integer.toString(rule.fromRules.size()));
          else
          if (fName.equals("Depth of hierarchy"))
            writer.write(Integer.toString(rule.getHierarchyDepth()));
          else
          if (fName.equals("N conditions"))
            writer.write(Integer.toString(cEx.eItems.length));
          else
            writer.write(""); //unrecognised field name (???)
        }
        for (int i=fIdx0; i<fIdx0+nBinary; i++) {
          writer.write(",");
          double minmax[] = cEx.getFeatureInterval(fieldNames.get(i));
          if (minmax == null)
            continue;
          if (Double.isInfinite(minmax[0]))
            if (Double.isInfinite(minmax[1]))
              continue;
            else
              writer.write((minmax[1]<1)?"0":"1");
          else
            if (Double.isInfinite(minmax[1]))
              writer.write((minmax[0]>0)?"1":"0");
            else
              if (Math.round(minmax[1]-minmax[0])>=1)
                continue;
              else
                writer.write((minmax[1]<1)?"0":"1");
        }
        for (int i=fIdx0+nBinary; i<fieldNames.size(); i+=2) {
          writer.write(",");
          String fName=fieldNames.get(i);
          int idx=fName.lastIndexOf(": min");
          if (idx>0)
            fName=fName.substring(0,idx);
          double minmax[] = cEx.getFeatureInterval(fName);
          if (minmax == null || (Double.isInfinite(minmax[0]) && Double.isInfinite(minmax[1]))) {
            writer.write(",");
            continue;
          }
          if (attrMinMax!=null) {
            float absMM[]=attrMinMax.get(fName);
            if (Double.isInfinite(minmax[0]))
              minmax[0]=absMM[0];
            if (Double.isInfinite(minmax[1]))
              minmax[1]=absMM[1];
            if (minmax[0]<=absMM[0] && minmax[1]>=absMM[1]) { //whole range
              writer.write(",");
              continue;
            }
          }
          if (intFeatures.contains(fName))
            writer.write(Math.round(minmax[0])+","+Math.round(minmax[1]));
          else
            writer.write(String.format("%.6f,%.6f",minmax[0],minmax[1]));
        }
        writer.newLine();
      }
      ok=true;
    } catch (Exception ex) {
      ex.printStackTrace();
    }
    if (ok && paramIdx0>0 && paramIdxLast>paramIdx0) { //generate parameter descriptor
      String descrFName=file.getAbsolutePath()+".descr";
      File descrFile=new File(descrFName);
      try (BufferedWriter writer = new BufferedWriter(new FileWriter(descrFile))) {
        writer.write("<CaptionParameter>");
        writer.newLine();
        writer.write("param_name=\"bound\"");
        writer.newLine();
        writer.write("values=\"min\",\"max\"");
        writer.newLine();
        for (int i=paramIdx0; i<paramIdxLast; i+=2) {
          String fName=fieldNames.get(i);
          int idx=fName.lastIndexOf(": min");
          if (idx>0)
            fName=fName.substring(0,idx);
          writer.write("\""+fName+"\"="+(i+1)+","+(i+2));
          writer.newLine();
        }
        writer.write("</CaptionParameter>");
        writer.newLine();
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
    return ok;
  }
  
  public static boolean hasDistinctTreeIds(ArrayList<CommonExplanation> rules) {
    if (rules==null || rules.isEmpty())
      return false;
    int id1=-1;
    for (CommonExplanation rule:rules)
      if (rule.treeId>=0)
        if (id1<0) id1=rule.treeId;
        else
          if (id1!=rule.treeId)
            return true;
    return false;
  }

  public static int[] getAllTreeIds(ArrayList<CommonExplanation> rules) {
    if (rules==null || rules.isEmpty())
      return null;
    HashSet<Integer> treeIds=new HashSet<Integer>(250);
    for (CommonExplanation rule:rules)
      if (rule.treeId>=0 && !treeIds.contains(rule.treeId))
        treeIds.add(rule.treeId);
    if (treeIds.isEmpty())
      return null;
    int idArray[]=new int[treeIds.size()];
    int k=0;
    for (Integer n:treeIds)
      idArray[k++]=n;
    Arrays.sort(idArray);
    return idArray;
  }

  public static HashMap<Integer,Integer> getTreeClusterIds(ArrayList<CommonExplanation> rules) {
    if (rules==null || rules.isEmpty())
      return null;
    HashMap<Integer,Integer> treeClusters=new HashMap<Integer,Integer>(250);
    HashSet<Integer> clusterIds=new HashSet<Integer>(20);
    for (CommonExplanation rule:rules)
      if (rule.treeId>=0 && rule.treeCluster>=0 && !treeClusters.containsKey(rule.treeId)) {
        treeClusters.put(rule.treeId, rule.treeCluster);
        if (!clusterIds.contains(rule.treeCluster))
          clusterIds.add(rule.treeCluster);
      }
    if (clusterIds.size()<2)
      return null;
    return treeClusters;
  }

  public static double[][] computeDistancesBetweenTrees (ArrayList<CommonExplanation> rules,
                                                         int treeIds[],
                                                         HashSet<String> featuresToUse,
                                                         Hashtable<String,float[]> attrMinMaxValues) {
    if (rules==null || rules.isEmpty() || treeIds==null || treeIds.length<2)
      return null;
    int nTrees=treeIds.length;
    ArrayList<CommonExplanation> treeRules[]=new ArrayList[nTrees];
    for (int i=0; i<nTrees; i++)
      treeRules[i]=new ArrayList<CommonExplanation>(rules.size()/nTrees*2);
    for (CommonExplanation rule:rules)
      if (rule.treeId>=0)
        for (int i=0; i<nTrees; i++)
          if (rule.treeId==treeIds[i]) {
            treeRules[i].add(rule);
            break;
          }
    double d[][]=new double[nTrees][nTrees];
    for (int i=0; i<nTrees; i++) {
      d[i][i]=0;
      for (int j=i+1; j<nTrees; j++) {
        int nPairs=0;
        double sumDist=0;
        for (CommonExplanation rule1:treeRules[i]) { //for each rule of the first tree find the closest rule of the second tree
          double minDist=Double.NaN;
          for (CommonExplanation rule2:treeRules[j]) {
            double dist=CommonExplanation.distance(rule1.eItems,rule2.eItems,featuresToUse,attrMinMaxValues);
            if (!Double.isNaN(dist))
              if (Double.isNaN(minDist) || minDist>dist)
                minDist=dist;
          }
          if (!Double.isNaN(minDist)) {
            sumDist+=minDist; ++nPairs;
          }
        }
        for (CommonExplanation rule2:treeRules[j]) { //for each rule of the second tree find the closest rule of the first tree
          double minDist=Double.NaN;
          for (CommonExplanation rule1:treeRules[i]) {
            double dist=CommonExplanation.distance(rule1.eItems,rule2.eItems,featuresToUse,attrMinMaxValues);
            if (!Double.isNaN(dist))
              if (Double.isNaN(minDist) || minDist>dist)
                minDist=dist;
          }
          if (!Double.isNaN(minDist)) {
            sumDist+=minDist; ++nPairs;
          }
        }
        if (nPairs>0)
          d[i][j]=d[j][i]=sumDist/nPairs;
      }
    }
    return d;
  }

  public static ArrayList<CommonExplanation> selectRulesOfTrees(ArrayList<CommonExplanation> rules,
                                                                HashSet<Integer> treeIds) {
    if (rules==null || rules.isEmpty() || treeIds==null || treeIds.isEmpty())
      return null;
    ArrayList<CommonExplanation> selectedRules=null;
    for (CommonExplanation rule:rules)
      if (treeIds.contains(rule.treeId)) {
        if (selectedRules==null)
          selectedRules=new ArrayList<CommonExplanation>(2000);
        selectedRules.add(rule);
      }
    return selectedRules;
  }

  public static boolean applyRulesToData(ArrayList<CommonExplanation> rules,
                                         ArrayList<DataRecord> data) {
    if (rules==null || rules.isEmpty() || data==null || data.isEmpty())
      return false;
    int predictionCount=0, applicationCount=0;
    for (DataRecord record:data) {
      record.erasePrediction();
      for (CommonExplanation rule:rules)
        if (ruleAppliesToDataRecord(rule,record)) {
          ++applicationCount;
          if (record.predictions==null) {
            record.predictions = new HashMap<Integer, Integer>(10);
            ++predictionCount;
          }
          if (record.predictions.get(rule.action)==null)
            record.predictions.put(rule.action,rule.weight);
          else
            record.predictions.put(rule.action,record.predictions.get(rule.action)+rule.weight);
        }
      if (record.predictions==null)
        continue;
      int maxClass=-1, maxRulesForClass=0;
      for (Integer classN:record.predictions.keySet())
        if (record.predictions.get(classN)>maxRulesForClass) {
          maxClass=classN; maxRulesForClass=record.predictions.get(classN);
        }
      record.predictedClassIdx=maxClass;
    }
    System.out.println("Rule application to data: made "+applicationCount+
        " successful applications of "+rules.size()+" rules to "+predictionCount+
        " data records out of "+data.size()+" available records");
    return predictionCount>0;
  }

  public static boolean ruleAppliesToDataRecord(CommonExplanation rule, DataRecord data) {
    if (rule==null || rule.eItems==null || rule.eItems.length==0)
      return false;
    if (data==null || data.items==null || data.items.isEmpty())
      return false;
    //String str=null;
    for (ExplanationItem ei:rule.eItems) {
      if (ei.interval==null || (Double.isInfinite(ei.interval[0]) && Double.isInfinite(ei.interval[1])))
        continue;
      DataElement item=data.getDataElement(ei.attr);
      if (item==null || !item.hasAnyValue())
        return false;
      double value=(item.dataType==DataElement.INTEGER)?item.getIntValue():item.getDoubleValue();
      if (!Double.isInfinite(ei.interval[0]) && value<ei.interval[0])
        return false;
      if (!Double.isInfinite(ei.interval[1]) && value>ei.interval[1])
        return false;
      /*
      String s=ei.attr+" = "+value+" in ["+ei.interval[0]+","+ei.interval[1]+"]";
      if (str==null)
        str=s;
      else
        str+="\t"+s;
      */
    }
    /*
    str+="\t>>> class = "+rule.action;
    System.out.println(str);
    */
    return true;
  }

  public static boolean testRulesOnData(ArrayList<CommonExplanation> rules,
                                        DataSet data) {
    if (rules==null || rules.isEmpty() || data==null || data.determineTargetType()==DataRecord.NO_TARGET)
      return false;

    for (CommonExplanation rule:rules)
      rule.nUses=rule.nCasesRight=rule.nCasesWrong=0;

    int applicationCount=0, nRight=0, nWrong=0;
    for (DataRecord record:data.records) {
      if (record.getTargetType()==DataRecord.NO_TARGET)
        continue;
      for (CommonExplanation rule:rules)
        if (ruleAppliesToDataRecord(rule,record)) {
          ++applicationCount;
          ++rule.nUses;
          boolean ok=false;
          switch (record.getTargetType()) {
            case DataRecord.CLASS_TARGET:
              ok= record.origClassIdx==rule.action;
              break;
            case DataRecord.VALUE_TARGET:
              ok= record.origValue>=rule.minQ && record.origValue<=rule.maxQ;
              break;
          }
          if (ok) {
            ++rule.nCasesRight; ++nRight;
          }
          else {
            ++rule.nCasesWrong; ++nWrong;
          }
        }
    }
    int nRulesApplied=0;
    for (CommonExplanation rule:rules)
      if (rule.nUses>0) ++nRulesApplied;

    System.out.println("Rule testing on data: made "+applicationCount+
        " applications of "+nRulesApplied+" out of "+rules.size()+" rules " +
        String.format("%.2f",100.0*nRulesApplied/rules.size())+
        "%) to "+data.records.size()+" data records.\n"+
        "N correct predictions = "+nRight+", N wrong predictions = "+nWrong);
    return nRight+nWrong>0;
  }

  public static ArrayList<CommonExplanation> selectRulesApplicableToData(
      ArrayList<CommonExplanation> rules,
      ArrayList<DataRecord> data)
  {
    if (rules==null || rules.isEmpty() || data==null || data.isEmpty())
      return null;
    ArrayList<CommonExplanation> applicable=null;
    for (CommonExplanation rule:rules)
      for (DataRecord record:data)
        if (ruleAppliesToDataRecord(rule,record)) {
          if (applicable==null)
            applicable=new ArrayList<CommonExplanation>(100);
          applicable.add(rule);
          break;
        }
    return applicable;
  }

  public static ArrayList<DataRecord> selectDataForRule(CommonExplanation rule, DataSet data) {
    if (rule==null || data==null)
      return null;
    ArrayList<DataRecord> selectedData=null;
    byte targetType=data.records.get(0).getTargetType();
    int nWrong=0;
    for (DataRecord rec:data.records)
      if (ruleAppliesToDataRecord(rule,rec)) {
        if (selectedData==null)
          selectedData=new ArrayList<DataRecord>(data.records.size()/5);
        DataRecord rec2=rec.makeNewVersion();
        boolean ok=false;
        switch (targetType) {
          case DataRecord.CLASS_TARGET:
            ok= rec.origClassIdx==rule.action;
            rec2.predictedClassIdx=rule.action;
            break;
          case DataRecord.VALUE_TARGET:
            ok= rec.origValue>=rule.minQ && rec.origValue<=rule.maxQ;
            rec2.predictedValue=rule.meanQ;
            if (rule.minQ<rule.maxQ) {
              double range[] = {rule.minQ, rule.maxQ};
              rec2.predictedValueRange = range;
            }
            break;
        }
        if (ok)
          selectedData.add(rec2);
        else
          selectedData.add(nWrong++,rec2);
      }
    return selectedData;
  }
}
