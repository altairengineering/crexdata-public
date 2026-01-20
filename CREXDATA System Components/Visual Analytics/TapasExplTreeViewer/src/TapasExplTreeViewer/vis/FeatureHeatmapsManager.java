package TapasExplTreeViewer.vis;

import TapasExplTreeViewer.rules.ValuesFrequencies;

import javax.swing.*;

public class FeatureHeatmapsManager {
  /**
   * Frequencies of feature value intervals across a set of rules
   */
  public ValuesFrequencies freq[][]=null;
  
  public int nClasses=0, nFeatures=0, nIntervals=1;
  public String featureNames[]=null, classLabels[]=null;
  public double featureBreaks[][]=null;
  /**
   * Description of the rules, which may include the description of the filter conditions
   * applied for extracting the rule subset
   */
  public String info=null;
  /**
   * Heatmap drawers showing the distribution of the feature values by classes or intervals of the prediction results
   */
  public HeatmapDrawer hmDrawByClasses[]=null, hmDrawByFeatures[]=null;

  public MultiHeatmapPanel hmPanelClasses=null, hmPanelFeatures=null;
  
  public JTextArea infoArea=null;
  
  public boolean getCountsByClasses() {
    if (freq==null || nClasses<1 || nFeatures<1 || nIntervals<2)
      return false;
    int countsByClasses[][][]=new int[nClasses][][];
    if (hmDrawByClasses==null) {
      hmDrawByClasses=new HeatmapDrawer[nClasses];
      for (int i=0; i<nClasses; i++)
        hmDrawByClasses[i]=null;
    }
    for (int cIdx=0; cIdx<nClasses; cIdx++) {
      int counts[][]=new int[nFeatures][];
      countsByClasses[cIdx]=counts;
      int nPresent[]=new int[nFeatures], nAbsent[]=new int[nFeatures];
      int nRulesCounted=0;
      for (int fIdx=0; fIdx<nFeatures; fIdx++) {
        nPresent[fIdx]=freq[fIdx][cIdx].nRulesWithFeature;
        nAbsent[fIdx]=freq[fIdx][cIdx].nAbsences;
        nRulesCounted=Math.max(nRulesCounted,nPresent[fIdx]);
        counts[fIdx]=freq[fIdx][cIdx].counts;
        if (counts[fIdx]==null) {
          //make an array of zeros
          counts[fIdx]=new int[nIntervals];
          for (int i=0; i<nIntervals; i++)
            counts[fIdx][i]=0;
          freq[fIdx][cIdx].counts=counts[fIdx];
        }
      }
      if (hmDrawByClasses[cIdx]==null)
        hmDrawByClasses[cIdx]=new HeatmapDrawer(counts,nRulesCounted,nPresent,nAbsent,featureBreaks,
            classLabels[cIdx],"feature values",null,featureNames);
      else
        hmDrawByClasses[cIdx].updateData(counts,nRulesCounted,nPresent,nAbsent);
    }
    return true;
  }
  
  public boolean getCountsByFeatures() {
    if (freq==null || nClasses<1 || nFeatures<1 || nIntervals<2)
      return false;
    int countsByFeatures[][][]=new int[nFeatures][][];
    if (hmDrawByFeatures==null) {
      hmDrawByFeatures=new HeatmapDrawer[nFeatures];
      for (int i=0; i<nFeatures; i++)
        hmDrawByFeatures[i]=null;
    }
    for (int fIdx=0; fIdx<nFeatures; fIdx++) {
      int counts[][]=new int[nClasses][];
      countsByFeatures[fIdx]=counts;
      int nPresent[]=new int[nClasses], nAbsent[]=new int[nClasses];
      int nRulesCounted=0;
      double breaks[][]=new double[nClasses][];
      for (int cIdx=0; cIdx<nClasses; cIdx++) {
        breaks[cIdx]=featureBreaks[fIdx]; //all are the same
        counts[cIdx] = freq[fIdx][cIdx].counts;
        nPresent[cIdx]=freq[fIdx][cIdx].nRulesWithFeature;
        nAbsent[cIdx]=freq[fIdx][cIdx].nAbsences;
        nRulesCounted=Math.max(nRulesCounted,freq[fIdx][cIdx].nRulesWithFeature);
      }
      if (hmDrawByFeatures[fIdx]==null)
        hmDrawByFeatures[fIdx]=new HeatmapDrawer(counts,nRulesCounted,nPresent,nAbsent,breaks,
            featureNames[fIdx],"feature values",null,classLabels);
      else
        hmDrawByFeatures[fIdx].updateData(counts,nRulesCounted,nPresent,nAbsent);
    }
    return true;
  }
  
  public JComponent createFeatureDistributionsDisplay(ValuesFrequencies freq[][], String info) {
    this.freq=freq; this.info=info;
    if (freq==null)
      return null;
  
    nClasses=freq[0].length; nFeatures=freq.length; nIntervals=freq[0][0].breaks.length+1;
    if (nClasses<1 || nFeatures<1 || nIntervals<2)
      return null;
  
    featureNames=new String[nFeatures];
    classLabels=new String[nClasses];
    featureBreaks=new double[nFeatures][];
    for (int fIdx=0; fIdx<nFeatures; fIdx++) {
      featureNames[fIdx] = freq[fIdx][0].featureName;
      featureBreaks[fIdx]=freq[fIdx][0].breaks;
    }
    for (int cIdx=0; cIdx<nClasses; cIdx++)
      if (freq[0][cIdx].resultMinMax!=null)
        classLabels[cIdx]=String.format("[%.3f..%.3f)",freq[0][cIdx].resultMinMax[0],freq[0][cIdx].resultMinMax[1]);
      else
        classLabels[cIdx]="Class "+freq[0][cIdx].action;

    if (!getCountsByClasses() || !getCountsByFeatures())
      return null;
    
    //create an instance of HeatmapDrawer for each class or interval of predicted values
    hmPanelClasses=new MultiHeatmapPanel();
    for (int cIdx=0; cIdx<nClasses; cIdx++)
      hmPanelClasses.addHeatmap(hmDrawByClasses[cIdx],classLabels[cIdx]+
          " ("+freq[0][cIdx].classSize+" rules)");
    
    hmPanelFeatures=new MultiHeatmapPanel();
    for (int fIdx=0; fIdx<nFeatures; fIdx++) {
      int count=0;
      for (int cIdx=0; cIdx<nClasses; cIdx++)
        count+=freq[fIdx][cIdx].nRulesWithFeature;
      hmPanelFeatures.addHeatmap(hmDrawByFeatures[fIdx],featureNames[fIdx]+" ("+count+" rules)");
    }
    JTabbedPane tabbedPane=new JTabbedPane();
    tabbedPane.addTab("Class-wise",hmPanelClasses);
    tabbedPane.addTab("Feature-wise",hmPanelFeatures);
    
    infoArea=new JTextArea();
    infoArea.setText(info);
    JSplitPane sp=new JSplitPane(JSplitPane.VERTICAL_SPLIT,tabbedPane, infoArea);
    sp.setResizeWeight(0.95);
    return sp;
  }
  
  public boolean hasData() {
    return freq!=null;
  }
  
  public boolean viewCreated() {
    return hmDrawByFeatures!=null && hmDrawByClasses!=null;
  }
  
  public boolean updateData(ValuesFrequencies freq[][], String info) {
    if (!viewCreated())
      return false;
    this.freq=freq; this.info=info;
    if (!getCountsByClasses() || !getCountsByFeatures())
      return false;
    for (int cIdx=0; cIdx<nClasses; cIdx++)
      hmPanelClasses.setHeatmapTitle(classLabels[cIdx]+
          " ("+freq[0][cIdx].classSize+" rules)",cIdx);
    hmPanelClasses.repaint();
    for (int fIdx=0; fIdx<nFeatures; fIdx++) {
      int count=0;
      for (int cIdx=0; cIdx<nClasses; cIdx++)
        count+=freq[fIdx][cIdx].nRulesWithFeature;
      hmPanelFeatures.setHeatmapTitle(featureNames[fIdx]+" ("+count+" rules)",fIdx);
    }
    hmPanelFeatures.repaint();
    infoArea.setText(info);
    return true;
  }
}
