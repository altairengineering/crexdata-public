package TapasDataReader;

import java.util.*;

public class ExTreeReconstructor {
  /**
   * Minimums and maximums for the attributes
   */
  public Hashtable<String,float[]> attrMinMaxValues =null;

  public Hashtable<Integer,ExTreeNode> topNodes=null, topNodesInt=null;
  public Hashtable<Integer,ExTreeNode> topNodesExCombined =null, topNodesIntExCombined=null;
  public Hashtable<String,Integer> attributes=null;
  public Hashtable<String,Integer> sectors=null;
  public Hashtable<String,Integer> attributesWithSectors=null;
  
  public void setAttrMinMaxValues(Hashtable<String,float[]> attrMinMaxValues) {
    this.attrMinMaxValues=attrMinMaxValues;
  }
  
  /**
   * Based on the explanations of the given set of flights, reconstructs 4 variants of the decision tree
   * (explanation tree):
   *  1) using the full sequences of the original explanation conditions;
   *     the results are stored in topNodes;
   *  2) using the full sequences of the conditions with the breaks converted to integer values;
   *     the results are stored in topNodesInt;
   *  3) using compressed explanations, in which multiple conditions referring to the same attribute
   *     are combined into a single condition; the results are stored in topNodesExCombined;
   *  4) using compressed explanations after converting the breaks to integer values;
   *     the results are stored in topNodesIntExCombined.
   *  The top noded correspond to different decisions (actions), e.g., 0, 1, ..., 10.
   *
   * @param flights - the set of flight plan variants with explanations corresponding to different
   *                  iteration steps of the solution development
   * @return true if successful
   */
  public boolean reconstructExTree (Hashtable<String, Flight> flights) {
    return reconstructExTree(flights,null,null);
  }
  
  /**
   * Based on the explanations of the given set of flights, reconstructs 4 variants of the decision tree
   * (explanation tree):
   *  1) using the full sequences of the original explanation conditions;
   *     the results are stored in topNodes;
   *  2) using the full sequences of the conditions with the breaks converted to integer values;
   *     the results are stored in topNodesInt;
   *  3) using compressed explanations, in which multiple conditions referring to the same attribute
   *     are combined into a single condition; the results are stored in topNodesExCombined;
   *  4) using compressed explanations after converting the breaks to integer values;
   *     the results are stored in topNodesIntExCombined.
   *  The top noded correspond to different decisions (actions), e.g., 0, 1, ..., 10.
   *
   * @param flights - the set of flight plan variants with explanations corresponding to different
   *                  iteration steps of the solution development
   * @param actionMinMax - an array of length 2 specifying the range of actions to include in the results.
   *                     If null, all actions are taken.
   * @param stepMinMax - an array of length 2 specifying the range of the solution steps to include
   *                   in the results. If null, all steps are taken.
   * @return true if successful
   */
  public boolean reconstructExTree (Hashtable<String, Flight> flights,
                                    int actionMinMax[], int stepMinMax[]) {
    if (flights==null || flights.isEmpty())
      return false;
    for (Map.Entry<String,Flight> e:flights.entrySet()) {
      Flight f=e.getValue();
      if (f.expl==null || f.expl.length<1)
        continue;
      for (int i=0; i<f.expl.length; i++)
        if (f.expl[i] != null && f.expl[i].eItems != null) {
          if (actionMinMax!=null &&
                  (f.expl[i].action<actionMinMax[0] || f.expl[i].action>actionMinMax[1]))
            continue;
          if (stepMinMax!=null &&
                  (f.expl[i].step<stepMinMax[0] || f.expl[i].step>stepMinMax[1]))
            continue;
          if (topNodes==null)
            topNodes=new Hashtable<Integer,ExTreeNode>(20);
          ExTreeNode currNode=topNodes.get(f.expl[i].action);
          if (currNode==null) {
            currNode=new ExTreeNode();
            topNodes.put(f.expl[i].action,currNode);
            currNode.attrName="Action = "+f.expl[i].action;
            currNode.level=-1;
          }
          currNode.addUse();
          for (int j = 0; j < f.expl[i].eItems.length; j++) {
            ExplanationItem eIt = f.expl[i].eItems[j];
            if (eIt == null)
              continue;
            ExTreeNode child = currNode.findChild(eIt.attr, eIt.interval);
            if (child == null) {
              child = new ExTreeNode();
              child.attrName = eIt.attr;
              child.level = currNode.level+1;
              child.condition = eIt.interval.clone();
              child.isInteger=eIt.isInteger;
              currNode.addChild(child);
            }
            child.addUse();
            currNode = child;
          
            if (attributes==null)
              attributes=new Hashtable<String,Integer>(100);
            Integer n=attributes.get(child.attrName);
            if (n==null) n=0;
            attributes.put(child.attrName,n+1);
          
            if (eIt.sector!=null && !eIt.sector.equalsIgnoreCase("null")) {
              child.addSectorUse(eIt.sector);
              if (sectors==null)
                sectors=new Hashtable<String,Integer>(100);
              n=sectors.get(eIt.sector);
              if (n==null) n=0;
              sectors.put(eIt.sector,n+1);
              if (attributesWithSectors==null)
                attributesWithSectors=new Hashtable<String,Integer>(100);
              n=attributesWithSectors.get(child.attrName);
              if (n==null) n=0;
              attributesWithSectors.put(child.attrName,n+1);
            }
          }
        }
      for (int i=0; i<f.expl.length; i++)
        if (f.expl[i] != null && f.expl[i].eItems != null) {
          if (actionMinMax!=null &&
                  (f.expl[i].action<actionMinMax[0] || f.expl[i].action>actionMinMax[1]))
            continue;
          if (stepMinMax!=null &&
                  (f.expl[i].step<stepMinMax[0] || f.expl[i].step>stepMinMax[1]))
            continue;
          ExplanationItem combItems[] = Explanation.getExplItemsCombined(f.expl[i].eItems);
          if (combItems != null) {
            if (topNodesExCombined == null)
              topNodesExCombined = new Hashtable<Integer, ExTreeNode>(20);
            ExTreeNode currNode = topNodesExCombined.get(f.expl[i].action);
            if (currNode == null) {
              currNode = new ExTreeNode();
              topNodesExCombined.put(f.expl[i].action, currNode);
              currNode.attrName = "Action = " + f.expl[i].action;
              currNode.level=-1;
            }
            currNode.addUse();
            for (int j = 0; j < combItems.length; j++) {
              ExplanationItem eIt = combItems[j];
              if (eIt == null)
                continue;
              ExTreeNode child = currNode.findChild(eIt.attr, eIt.interval);
              if (child == null) {
                child = new ExTreeNode();
                child.attrName = eIt.attr;
                child.level = currNode.level+1;
                child.condition = eIt.interval.clone();
                child.isInteger=eIt.isInteger;
                currNode.addChild(child);
              }
              child.addUse();
              currNode = child;
            }
          }
          if (attrMinMaxValues!=null) {
            ExplanationItem intItems[] = f.expl[i].getExplItemsAsIntegeres(f.expl[i].eItems, attrMinMaxValues);
            if (intItems!=null) {
              if (topNodesInt==null)
                topNodesInt = new Hashtable<Integer, ExTreeNode>(20);
              ExTreeNode currNode = topNodesInt.get(f.expl[i].action);
              if (currNode == null) {
                currNode = new ExTreeNode();
                topNodesInt.put(f.expl[i].action, currNode);
                currNode.attrName = "Action = " + f.expl[i].action;
                currNode.level=-1;
              }
              currNode.addUse();
              for (int j = 0; j < intItems.length; j++) {
                ExplanationItem eIt = intItems[j];
                if (eIt == null)
                  continue;
                ExTreeNode child = currNode.findChild(eIt.attr, eIt.interval);
                if (child == null) {
                  child = new ExTreeNode();
                  child.attrName = eIt.attr;
                  child.level = currNode.level+1;
                  child.condition = eIt.interval.clone();
                  child.isInteger=eIt.isInteger;
                  currNode.addChild(child);
                }
                child.addUse();
                currNode = child;
              }
            }
            ExplanationItem combIntItems[]=(intItems==null)?null:Explanation.getExplItemsCombined(intItems);
            if (combIntItems!=null) {
              if (topNodesIntExCombined==null)
                topNodesIntExCombined = new Hashtable<Integer, ExTreeNode>(20);
              ExTreeNode currNode = topNodesIntExCombined.get(f.expl[i].action);
              if (currNode == null) {
                currNode = new ExTreeNode();
                topNodesIntExCombined.put(f.expl[i].action, currNode);
                currNode.attrName = "Action = " + f.expl[i].action;
                currNode.level=-1;
              }
              currNode.addUse();
              for (int j = 0; j < combIntItems.length; j++) {
                ExplanationItem eIt = combIntItems[j];
                if (eIt == null)
                  continue;
                ExTreeNode child = currNode.findChild(eIt.attr, eIt.interval);
                if (child == null) {
                  child = new ExTreeNode();
                  child.attrName = eIt.attr;
                  child.level = currNode.level+1;
                  child.condition = eIt.interval.clone();
                  child.isInteger=eIt.isInteger;
                  currNode.addChild(child);
                }
                child.addUse();
                currNode = child;
              }
            }
          }
        }
    }
    countLeaves();
    return topNodes!=null && !topNodes.isEmpty();
  }
  
  public void countLeaves() {
    if (topNodes!=null)
      for (Map.Entry<Integer,ExTreeNode> entry:topNodes.entrySet())
        entry.getValue().countLeavesBelow();
    if (topNodesInt!=null)
      for (Map.Entry<Integer,ExTreeNode> entry:topNodesInt.entrySet())
        entry.getValue().countLeavesBelow();
    if (topNodesExCombined!=null)
      for (Map.Entry<Integer,ExTreeNode> entry:topNodesExCombined.entrySet())
        entry.getValue().countLeavesBelow();
    if (topNodesIntExCombined!=null)
      for (Map.Entry<Integer,ExTreeNode> entry:topNodesIntExCombined.entrySet())
        entry.getValue().countLeavesBelow();
  }
  
  /**
   * Traverses the trees corresponding to each action and counts for each action
   * how many times each attribute occurs
   * @return a matrix with the rows corresponding to the attributes and columns to the actions.
   * The first column contains the total counts of the actions
   */
  public CountMatrix countActionsPerAttributes() {
    if (topNodes==null || topNodes.isEmpty() || attributes==null || attributes.isEmpty())
      return null;
    CountMatrix matrix=new CountMatrix();
    matrix.rowNames=new String[attributes.size()];
    int idx=0;
    for (String aName:attributes.keySet())
      matrix.rowNames[idx++]=aName;

    matrix.colNames=new String[topNodes.size()+2];
    matrix.colNames[0]="Attribute";
    matrix.colNames[1]="Total";
    ArrayList<Integer> actions=new ArrayList<Integer>(topNodes.size());
    for (Integer action:topNodes.keySet())
      actions.add(action);
    Collections.sort(actions);
    idx=2;
    for (int i=actions.size()-1; i>=0; i--)
      matrix.colNames[idx++]=actions.get(i).toString();
    matrix.cellValues=new Integer[matrix.rowNames.length][matrix.colNames.length-1];
    for (int r=0; r<matrix.cellValues.length; r++)
      for (int c=0; c<matrix.cellValues[r].length; c++)
        matrix.cellValues[r][c]=0;
    for (Map.Entry<Integer,ExTreeNode> e:topNodes.entrySet()) {
      countAttributeUsesInNode(e.getValue(),e.getKey(),matrix);
    }
    return matrix;
  }
  
  protected void countAttributeUsesInNode(ExTreeNode node, Integer action, CountMatrix matrix) {
    if (node==null || matrix==null || matrix.cellValues==null || matrix.rowNames==null)
      return;
    if (node.attrName!=null) {
      int rIdx = -1;
      for (int i = 0; i < matrix.rowNames.length && rIdx < 0; i++)
        if (node.attrName.equals(matrix.rowNames[i]))
          rIdx = i;
      if (rIdx >= 0) {
        matrix.cellValues[rIdx][0] = matrix.cellValues[rIdx][0] + 1; //total
        if (action >= 0) {
          int cIdx = matrix.colNames.length - 2 - action; //the actions are in descending order
          if (cIdx > 0)
            matrix.cellValues[rIdx][cIdx] = matrix.cellValues[rIdx][cIdx] + 1;
        }
      }
    }
    if (node.children!=null)
      for (ExTreeNode child:node.children)
        countAttributeUsesInNode(child,action,matrix);
  }
  
  public CountMatrix countActionsPerSectors() {
    if (topNodes==null || topNodes.isEmpty() || sectors==null || sectors.isEmpty())
      return null;
    CountMatrix matrix=new CountMatrix();
    matrix.rowNames=new String[sectors.size()];
    int idx=0;
    for (String aName:sectors.keySet())
      matrix.rowNames[idx++]=aName;

    matrix.colNames=new String[topNodes.size()+2];
    matrix.colNames[0]="Sector";
    matrix.colNames[1]="Total";
    ArrayList<Integer> actions=new ArrayList<Integer>(topNodes.size());
    for (Integer action:topNodes.keySet())
      actions.add(action);
    Collections.sort(actions);
    idx=2;
    for (int i=actions.size()-1; i>=0; i--)
      matrix.colNames[idx++]=actions.get(i).toString();

    matrix.cellValues=new Integer[matrix.rowNames.length][matrix.colNames.length-1];
    for (int r=0; r<matrix.cellValues.length; r++)
      for (int c=0; c<matrix.cellValues[r].length; c++)
        matrix.cellValues[r][c]=0;
    for (Map.Entry<Integer,ExTreeNode> e:topNodes.entrySet()) {
      countSectorUsesInNode(e.getValue(),e.getKey(),matrix);
    }
    return matrix;
  }
  
  protected void countSectorUsesInNode(ExTreeNode node, Integer action, CountMatrix matrix) {
    if (node==null || matrix==null || matrix.cellValues==null || matrix.rowNames==null)
      return;
    if (node.sectors!=null)
      for (int i=0; i<matrix.rowNames.length; i++) {
        Integer n=node.sectors.get(matrix.rowNames[i]);
        if (n!=null) {
          matrix.cellValues[i][0]=matrix.cellValues[i][0]+n; //total
          int cIdx = matrix.colNames.length - 2 - action; //the actions are in descending order
          if (cIdx > 0)
            matrix.cellValues[i][cIdx] = matrix.cellValues[i][cIdx] + n;
        }
      }
    if (node.children!=null)
      for (ExTreeNode child:node.children)
        countSectorUsesInNode(child,action,matrix);
  }
  
  public CountMatrix countAttributesPerSectors() {
    if (topNodes==null || topNodes.isEmpty() || sectors==null || sectors.isEmpty())
      return null;
    CountMatrix matrix=new CountMatrix();
    matrix.rowNames=new String[sectors.size()];
    int idx=0;
    for (String aName:sectors.keySet())
      matrix.rowNames[idx++]=aName;
    
    matrix.colNames=new String[attributesWithSectors.size()+2];
    matrix.colNames[0]="Sector";
    matrix.colNames[1]="Total";
    
    ArrayList<Map.Entry> attrList=new ArrayList<Map.Entry>(attributesWithSectors.size());
    for (Map.Entry<String,Integer> e:attributesWithSectors.entrySet()) {
      int eIdx=-1;
      for (int i=0; i<attrList.size() && eIdx<0; i++)
        if (e.getValue()>(Integer)attrList.get(i).getValue())
          eIdx=i;
      if (eIdx<0)
        attrList.add(e);
      else
        attrList.add(eIdx,e);
    }
    
    for (int i=0; i<attrList.size(); i++)
      matrix.colNames[i+2]=(String)attrList.get(i).getKey();

    matrix.cellValues=new Integer[matrix.rowNames.length][matrix.colNames.length-1];
    for (int r=0; r<matrix.cellValues.length; r++)
      for (int c=0; c<matrix.cellValues[r].length; c++)
        matrix.cellValues[r][c]=0;
  
    for (Map.Entry<Integer,ExTreeNode> e:topNodes.entrySet()) {
      countAttrPlusSectorUsesInNode(e.getValue(),matrix);
    }
    
    return matrix;
  }
  
  protected void countAttrPlusSectorUsesInNode(ExTreeNode node, CountMatrix matrix) {
    if (node==null || matrix==null || matrix.cellValues==null || matrix.rowNames==null)
      return;
    if (node.sectors!=null)
      for (int i=0; i<matrix.rowNames.length; i++) {
        Integer n=node.sectors.get(matrix.rowNames[i]);
        if (n!=null) {
          matrix.cellValues[i][0]=matrix.cellValues[i][0]+n; //total
          if (node.attrName != null) {
            int cIdx = -1;
            for (int j=2; j<matrix.colNames.length && cIdx<0; j++)
              if (node.attrName.equals(matrix.colNames[j]))
                cIdx=j-1;
              if (cIdx > 0)
                matrix.cellValues[i][cIdx] = matrix.cellValues[i][cIdx] + n;
          }
        }
      }
    if (node.children!=null)
      for (ExTreeNode child:node.children)
        countAttrPlusSectorUsesInNode(child,matrix);
  }
  
  public boolean reconstructExTree(ArrayList<Explanation> explanations) {
    if (explanations==null || explanations.isEmpty())
      return false;
    for (int e=0; e<explanations.size(); e++) {
      Explanation ex=explanations.get(e);
      if (ex==null || ex.eItems==null)
        continue;
      if (topNodes==null)
        topNodes=new Hashtable<Integer,ExTreeNode>(20);
      ExTreeNode currNode=topNodes.get(ex.action);
      if (currNode==null) {
        currNode=new ExTreeNode();
        topNodes.put(ex.action,currNode);
        currNode.attrName="Action = "+ex.action;
        currNode.level=-1;
      }
      currNode.addUse();
      for (int j = 0; j < ex.eItems.length; j++) {
        ExplanationItem eIt = ex.eItems[j];
        if (eIt == null)
          continue;
        ExTreeNode child = currNode.findChild(eIt.attr, eIt.interval);
        if (child == null) {
          child = new ExTreeNode();
          child.attrName = eIt.attr;
          child.level = currNode.level+1;
          child.condition = eIt.interval.clone();
          child.isInteger=eIt.isInteger;
          currNode.addChild(child);
        }
        child.addUse();
        currNode = child;
    
        if (attributes==null)
          attributes=new Hashtable<String,Integer>(100);
        Integer n=attributes.get(child.attrName);
        if (n==null) n=0;
        attributes.put(child.attrName,n+1);
    
        if (eIt.sector!=null && !eIt.sector.equalsIgnoreCase("null")) {
          child.addSectorUse(eIt.sector);
          if (sectors==null)
            sectors=new Hashtable<String,Integer>(100);
          n=sectors.get(eIt.sector);
          if (n==null) n=0;
          sectors.put(eIt.sector,n+1);
          if (attributesWithSectors==null)
            attributesWithSectors=new Hashtable<String,Integer>(100);
          n=attributesWithSectors.get(child.attrName);
          if (n==null) n=0;
          attributesWithSectors.put(child.attrName,n+1);
        }
      }
    }
    for (int e=0; e<explanations.size(); e++) {
      Explanation ex=explanations.get(e);
      if (ex==null || ex.eItems==null)
        continue;
        ExplanationItem combItems[] = Explanation.getExplItemsCombined(ex.eItems);
        if (combItems != null) {
          if (topNodesExCombined == null)
            topNodesExCombined = new Hashtable<Integer, ExTreeNode>(20);
          ExTreeNode currNode = topNodesExCombined.get(ex.action);
          if (currNode == null) {
            currNode = new ExTreeNode();
            topNodesExCombined.put(ex.action, currNode);
            currNode.attrName = "Action = " + ex.action;
            currNode.level=-1;
          }
          currNode.addUse();
          for (int j = 0; j < combItems.length; j++) {
            ExplanationItem eIt = combItems[j];
            if (eIt == null)
              continue;
            ExTreeNode child = currNode.findChild(eIt.attr, eIt.interval);
            if (child == null) {
              child = new ExTreeNode();
              child.attrName = eIt.attr;
              child.level = currNode.level+1;
              child.condition = eIt.interval.clone();
              child.isInteger=eIt.isInteger;
              currNode.addChild(child);
            }
            child.addUse();
            currNode = child;
          }
        }
        if (attrMinMaxValues!=null) {
          ExplanationItem intItems[] = ex.getExplItemsAsIntegeres(ex.eItems, attrMinMaxValues);
          if (intItems!=null) {
            if (topNodesInt==null)
              topNodesInt = new Hashtable<Integer, ExTreeNode>(20);
            ExTreeNode currNode = topNodesInt.get(ex.action);
            if (currNode == null) {
              currNode = new ExTreeNode();
              topNodesInt.put(ex.action, currNode);
              currNode.attrName = "Action = " + ex.action;
              currNode.level=-1;
            }
            currNode.addUse();
            for (int j = 0; j < intItems.length; j++) {
              ExplanationItem eIt = intItems[j];
              if (eIt == null)
                continue;
              ExTreeNode child = currNode.findChild(eIt.attr, eIt.interval);
              if (child == null) {
                child = new ExTreeNode();
                child.attrName = eIt.attr;
                child.level = currNode.level+1;
                child.condition = eIt.interval.clone();
                child.isInteger=eIt.isInteger;
                currNode.addChild(child);
              }
              child.addUse();
              currNode = child;
            }
          }
          ExplanationItem combIntItems[]=(intItems==null)?null:Explanation.getExplItemsCombined(intItems);
          if (combIntItems!=null) {
            if (topNodesIntExCombined==null)
              topNodesIntExCombined = new Hashtable<Integer, ExTreeNode>(20);
            ExTreeNode currNode = topNodesIntExCombined.get(ex.action);
            if (currNode == null) {
              currNode = new ExTreeNode();
              topNodesIntExCombined.put(ex.action, currNode);
              currNode.attrName = "Action = " + ex.action;
              currNode.level=-1;
            }
            currNode.addUse();
            for (int j = 0; j < combIntItems.length; j++) {
              ExplanationItem eIt = combIntItems[j];
              if (eIt == null)
                continue;
              ExTreeNode child = currNode.findChild(eIt.attr, eIt.interval);
              if (child == null) {
                child = new ExTreeNode();
                child.attrName = eIt.attr;
                child.level = currNode.level+1;
                child.condition = eIt.interval.clone();
                child.isInteger=eIt.isInteger;
                currNode.addChild(child);
              }
              child.addUse();
              currNode = child;
            }
          }
        }
      }
    countLeaves();
    return topNodes!=null && !topNodes.isEmpty();
  }
}
