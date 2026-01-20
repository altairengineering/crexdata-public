package TapasExplTreeViewer.ui;

import TapasExplTreeViewer.rules.RuleSet;
import TapasExplTreeViewer.util.SliderWithBounds;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class RuleFilterUI extends JPanel {
  public RuleSet ruleSet=null;

  private Map<String, SliderWithBounds> sliders=null;
  private Map<String, JCheckBox> checkBoxes=null;
  private JButton applyFilterButton=null, extractRulesButton=null, highlightRulesButton=null;

  private boolean rangesInsideLimits =true;

  private ArrayList<ChangeListener> changeListeners=null;

  public RuleFilterUI(RuleSet ruleSet) {
    this.ruleSet=ruleSet;

    setLayout(new BorderLayout());

    JPanel filtersPanel = new JPanel();
    filtersPanel.setLayout(new GridLayout(0, 1, 5, 5));
    sliders = new HashMap<>();
    checkBoxes = new HashMap<>();

    ArrayList<String> featureNames=(ruleSet.orderedFeatureNames!=null)?
        ruleSet.orderedFeatureNames:ruleSet.listOfFeatures;

    JCheckBox cbDynamic=new JCheckBox("Apply filters dynamically",true);

    for (int i=0; i<featureNames.size(); i++) {
      String featureName = featureNames.get(i);
      float minmax[]=ruleSet.attrMinMax.get(featureName);
      if (minmax==null)
        continue;

      JPanel featurePanel = new JPanel(new BorderLayout());
      featurePanel.setBorder(BorderFactory.createTitledBorder(featureName));

      // Checkbox for excluding rules not involving this feature
      JCheckBox checkBox = new JCheckBox("Select rules not involving this feature");
      featurePanel.add(checkBox, BorderLayout.NORTH);
      checkBoxes.put(featureName, checkBox);

      // RangeSlider for selecting feature value range
      SliderWithBounds rangeSlider = new SliderWithBounds(0,1000,0,1000);
      rangeSlider.setRealMinMax(minmax[0],minmax[1]);
      featurePanel.add(rangeSlider, BorderLayout.CENTER);
      sliders.put(featureName, rangeSlider);

      rangeSlider.addChangeListener(new ChangeListener() {
        @Override
        public void stateChanged(ChangeEvent e) {
          if (cbDynamic.isSelected())
            notifyListeners();
        }
      });

      checkBox.addItemListener(new ItemListener() {
        @Override
        public void itemStateChanged(ItemEvent e) {
          // Disable RangeSlider when checkbox is selected
          rangeSlider.setEnabled(!checkBox.isSelected());
          if (cbDynamic.isSelected())
            notifyListeners();
        }
      });

      filtersPanel.add(featurePanel);
    }

    add(new JScrollPane(filtersPanel), BorderLayout.CENTER);

    JPanel p=new JPanel();
    add(p, BorderLayout.SOUTH);
    p.setLayout(new GridLayout(0,1));

    p.add(new JLabel("Feature ranges in rules MUST"));
    ButtonGroup bg=new ButtonGroup();
    JRadioButton rbIncluded=new JRadioButton("be contained in filter limits", rangesInsideLimits);
    JRadioButton rbOverlap=new JRadioButton("overlap with filter limits",!rangesInsideLimits);
    bg.add(rbIncluded);
    bg.add(rbOverlap);
    p.add(rbIncluded);
    p.add(rbOverlap);

    ActionListener aList=new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        setLimitsApplicationMode(rbIncluded.isSelected());
      }
    };
    rbIncluded.addActionListener(aList);
    rbOverlap.addActionListener(aList);

    p.add(cbDynamic);
    JPanel pp=new JPanel();
    p.add(pp);
    pp.setLayout(new FlowLayout(FlowLayout.CENTER,30,0));
    applyFilterButton = new JButton("Apply");
    pp.add(applyFilterButton);
    JButton clearButton=new JButton("Clear");
    pp.add(clearButton);

    pp=new JPanel();
    pp.setLayout(new FlowLayout(FlowLayout.CENTER,30,0));
    extractRulesButton=new JButton("Extract rules");
    pp.add(extractRulesButton);
    highlightRulesButton=new JButton("Highlight rules");
    pp.add(highlightRulesButton);
    p.add(pp);

    clearButton.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        clearFilters();
      }
    });
  }

  public void setLimitsApplicationMode (boolean mustBeIncluded) {
    if (rangesInsideLimits==mustBeIncluded)
      return;
    rangesInsideLimits =mustBeIncluded;
    notifyListeners();
  }

  public void clearFilters() {
    boolean changed=false;
    for (String feature : sliders.keySet()) {
      SliderWithBounds slider = sliders.get(feature);
      if (checkBoxes.get(feature).isSelected()) {
        checkBoxes.get(feature).setSelected(false);
        slider.setEnabled(true);
        slider.resetLimitsToMinMax();
        changed=true;
      } else {
        if (slider.resetLimitsToMinMax())
          changed=true;
      }
    }
    if (changed)
      notifyListeners();
  }

  public void addChangeListener(ChangeListener lst) {
    if (lst==null) return;
    if (changeListeners==null)
      changeListeners=new ArrayList<ChangeListener>(5);
    if (!changeListeners.contains(lst))
      changeListeners.add(lst);
  }

  public void notifyListeners() {
    if (changeListeners==null || changeListeners.isEmpty())
      return;
    for (ChangeListener lst:changeListeners)
      lst.stateChanged(new ChangeEvent(this));
  }

  public Map<String, Object> getFilters() {
    Map<String, Object> filters = new HashMap<>();
    for (String feature : sliders.keySet()) {
      if (checkBoxes.get(feature).isSelected()) {
        filters.put(feature, "exclude");
      } else {
        SliderWithBounds slider = sliders.get(feature);
        double limits[]=new double[]{slider.getLowerValue(), slider.getUpperValue()};
        if (limits[0]<=slider.getRealMin() && limits[1]>=slider.getRealMax())
          continue; //no limits for this feature
        if (limits[0]<=slider.getRealMin())
          limits[0]=Double.NEGATIVE_INFINITY;
        else
          if (limits[1]>=slider.getRealMax())
            limits[1]=Double.POSITIVE_INFINITY;
        filters.put(feature,limits);
      }
    }
    return (filters.isEmpty())?null:filters;
  }

  public boolean mustRangesBeInsideLimits() {
    return rangesInsideLimits;
  }

  public ArrayList<String> describeFilters() {
    ArrayList<String> conditions=new ArrayList<String>();
    for (String feature : sliders.keySet()) {
      if (checkBoxes.get(feature).isSelected()) {
        conditions.add(feature + " is missing");
      } else {
        SliderWithBounds slider = sliders.get(feature);
        double limits[]=new double[]{slider.getLowerValue(), slider.getUpperValue()};
        if (limits[0]<=slider.getRealMin() && limits[1]>=slider.getRealMax())
          continue; //no limits for this feature
        if (limits[0]<=slider.getRealMin())
          limits[0]=Double.NEGATIVE_INFINITY;
        else
          if (limits[1]>=slider.getRealMax())
            limits[1]=Double.POSITIVE_INFINITY;
        conditions.add(feature+": "+limitsToString(limits));
      }
    }
   if (conditions.isEmpty())
     return null;
   return conditions;
 }
  public String limitsToString (double limits[]) {
    if (limits==null)
      return "null limits";
    StringBuffer sb=new StringBuffer();
    sb.append("from "+((Double.isInfinite(limits[0]))?"-infinity":String.format("%.3f",limits[0])));
    sb.append(" to  "+((Double.isInfinite(limits[1]))?"+infinity":String.format("%.3f",limits[1])));
    return sb.toString();
  }
  
  public JButton getApplyFilterButton() {
    return applyFilterButton;
  }

  public JButton getHighlightRulesButton() {
    return highlightRulesButton;
  }

  public JButton getExtractRulesButton() {
    return extractRulesButton;
  }
}
