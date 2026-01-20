package TapasExplTreeViewer.ui;

import TapasDataReader.CommonExplanation;
import TapasExplTreeViewer.rules.RuleMaster;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Vector;

public class RuleSelector {
  protected ArrayList<CommonExplanation> origRules =null, selectedRules=null;
  protected ChangeListener changeListener=null;
  /**
   * Features that were used in computing the distances
   */
  public ArrayList<String> features=null;
  public HashSet<String> selectedFeatures =null;

  protected JDialog queryDialog=null;
  protected boolean isRunning=false, queryExecuting=false;
  public String queryStr=null;

  public JRadioButton rbHighlight=null, rbExtract=null;

  public FeatureSelector feaSel=null;
  public JCheckBox cbMust=null, cbNot=null;
  public JRadioButton rbAll=null, rbAny=null;

  public boolean makeQueryInterface(ArrayList<CommonExplanation> rules,
                                    ArrayList<String> features,
                                    HashSet<String> selectedFeatures,
                                    ChangeListener changeListener) {
    if (rules==null || rules.isEmpty())
      return false;
    this.origRules =rules;
    this.features=features; this.selectedFeatures =selectedFeatures;
    this.changeListener=changeListener;
    
    int minClass=-1, maxClass=-1, minTreeId=-1, maxTreeId=-1, minTreeCluster=-1, maxTreeCluster=-1;
    double minValue=Double.NaN, maxValue=Double.NaN;
    Vector<String> categories=null;
    
    for (CommonExplanation r: rules) {
      if (r.action>=0) {
        if (minClass<0 || minClass>r.action) minClass=r.action;
        if (maxClass<r.action) maxClass=r.action;
      }
      if (!Double.isNaN(r.minQ)) {
        if (Double.isNaN(minValue) || minValue>r.minQ)
          minValue=r.minQ;
        if (Double.isNaN(maxValue) || maxValue<r.minQ)
          maxValue=r.minQ;
      }
      if (!Double.isNaN(r.maxQ) && r.maxQ>r.minQ) {
        if (Double.isNaN(maxValue) || maxValue<r.maxQ)
          maxValue=r.maxQ;
      }
      if (r.treeId>=0) {
        if (minTreeId<0 || minTreeId>r.treeId)
          minTreeId=r.treeId;
        if (maxTreeId<0 || maxTreeId<r.treeId)
          maxTreeId=r.treeId;
      }
      if (r.treeCluster>=0) {
        if (minTreeCluster<0 || minTreeCluster>r.treeCluster)
          minTreeCluster=r.treeCluster;
        if (maxTreeCluster<0 || maxTreeCluster<r.treeCluster)
          maxTreeCluster=r.treeCluster;
      }
      if (r.category!=null && r.category.length()>0) {
        if (categories==null)
          categories=new Vector<String>(20);
        if (!categories.contains(r.category))
          categories.add(r.category);
      }
    }
    JPanel topP=new JPanel();
    topP.setLayout(new GridLayout(0,1));
    JCheckBox cb[]=new JCheckBox[5];
    JTextField tfMin[]=new JTextField[4], tfMax[]=new JTextField[4];
    JComboBox<String> comboCategory=null;

    for (int i=0; i<4; i++) {
      cb[i]=null; tfMin[i]=tfMax[i]=null;
    }

    if (minClass<maxClass) {
      JPanel p=new JPanel();
      p.setLayout(new FlowLayout(FlowLayout.LEFT));
      cb[0]=new JCheckBox("Predicted class: from");
      p.add(cb[0]);
      tfMin[0]=new JTextField(Integer.toString(minClass),2);
      p.add(tfMin[0]);
      p.add(new JLabel("to",JLabel.RIGHT));
      tfMax[0]=new JTextField(Integer.toString(maxClass),2);
      p.add(tfMax[0]);
      topP.add(p);
    }
    if (minValue<maxValue) {
      JPanel p=new JPanel();
      p.setLayout(new FlowLayout(FlowLayout.LEFT));
      cb[1]=new JCheckBox("Predicted value: from");
      p.add(cb[1]);
      tfMin[1]=new JTextField(String.format("%.5f",minValue),10);
      p.add(tfMin[1]);
      p.add(new JLabel("to",JLabel.RIGHT));
      tfMax[1]=new JTextField(String.format("%.5f",maxValue),10);
      p.add(tfMax[1]);
      topP.add(p);
    }
    if (minTreeId<maxTreeId) {
      JPanel p=new JPanel();
      p.setLayout(new FlowLayout(FlowLayout.LEFT));
      cb[2]=new JCheckBox("Tree identifier: from");
      p.add(cb[2]);
      tfMin[2]=new JTextField(Integer.toString(minTreeId),2);
      p.add(tfMin[2]);
      p.add(new JLabel("to",JLabel.RIGHT));
      tfMax[2]=new JTextField(Integer.toString(maxTreeId),2);
      p.add(tfMax[2]);
      topP.add(p);
    }
    if (minTreeCluster<maxTreeCluster) {
      JPanel p=new JPanel();
      p.setLayout(new FlowLayout(FlowLayout.LEFT));
      cb[3]=new JCheckBox("Tree cluster: from");
      p.add(cb[3]);
      tfMin[3]=new JTextField(Integer.toString(minTreeCluster),2);
      p.add(tfMin[3]);
      p.add(new JLabel("to",JLabel.RIGHT));
      tfMax[3]=new JTextField(Integer.toString(maxTreeCluster),2);
      p.add(tfMax[3]);
      topP.add(p);
    }
    if (categories!=null && categories.size()>1) {
      Collections.sort(categories);
      JPanel p=new JPanel();
      p.setLayout(new FlowLayout(FlowLayout.LEFT));
      cb[4]=new JCheckBox("Category:");
      p.add(cb[4]);
      comboCategory=new JComboBox<String>(categories);
      p.add(comboCategory);
      topP.add(p);
    }

    if (topP.getComponentCount()<1)
      topP=null;

    JPanel fp=null;

    if (features!=null && !features.isEmpty()) {
      feaSel=new FeatureSelector(features,selectedFeatures);
      fp=new JPanel();
      fp.setLayout(new BorderLayout());
      fp.add(feaSel,BorderLayout.CENTER);
      JPanel cp=new JPanel();
      fp.add(cp,BorderLayout.NORTH);
      cp.setLayout(new FlowLayout());
      cp.add(new JLabel("Selected rules"));
      cbMust=new JCheckBox("must",false);
      cp.add(cbMust);
      cbNot=new JCheckBox("not",false);
      cp.add(cbNot);
      cp.add(new JLabel("involve"));
      rbAll=new JRadioButton("all",false);
      rbAny=new JRadioButton("any",false);
      cp.add(rbAll);
      cp.add(rbAny);
      cp.add(new JLabel("of the selected features:"));
      ButtonGroup rbg=new ButtonGroup();
      rbg.add(rbAll);
      rbg.add(rbAny);
      cbMust.addItemListener(new ItemListener() {
        @Override
        public void itemStateChanged(ItemEvent e) {
          if (cbMust.isSelected() && !rbAll.isSelected() && !rbAny.isSelected())
            rbAll.setSelected(true);
        }
      });
    }

    if (topP==null && feaSel==null)
      return false;

    JPanel mainP=new JPanel();
    mainP.setLayout(new BorderLayout());
    if (topP!=null)
      mainP.add(topP, BorderLayout.NORTH);
    if (fp!=null)
      mainP.add(fp,BorderLayout.CENTER);

    JPanel bp=new JPanel();
    bp.setLayout(new FlowLayout(FlowLayout.CENTER,20,5));
    JButton b=new JButton("Select");
    bp.add(b);
    rbHighlight=new JRadioButton("highlight",false);
    rbExtract=new JRadioButton("extract",true);
    ButtonGroup rbg=new ButtonGroup();
    rbg.add(rbHighlight); rbg.add(rbExtract);
    bp.add(rbHighlight);
    bp.add(rbExtract);

    Window owner=FocusManager.getCurrentManager().getActiveWindow();
    queryDialog=new JDialog(owner, "Select rules", Dialog.ModalityType.MODELESS);
    queryDialog.setLayout(new BorderLayout());
    queryDialog.getContentPane().add(new JLabel("Set conditions for selecting rules:",JLabel.CENTER),
        BorderLayout.NORTH);
    queryDialog.getContentPane().add(mainP,BorderLayout.CENTER);
    queryDialog.getContentPane().add(bp,BorderLayout.SOUTH);
    queryDialog.pack();
    queryDialog.setLocationRelativeTo(owner);
    queryDialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
    queryDialog.setVisible(true);
    isRunning=true;

    final JComboBox<String> comboCatCopy=comboCategory;

    b.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (queryExecuting) {
          JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
              "The previous query is still being executed!",
              "Wait...",JOptionPane.WARNING_MESSAGE);
        }
        int minClass=-1, maxClass=-1, minTreeId=-1, maxTreeId=-1, minTreeCluster=-1, maxTreeCluster=-1;
        double minValue=Double.NaN, maxValue=Double.NaN;
        String category=null;
        queryStr="";
        if (cb[0]!=null && cb[0].isSelected()) {
          try {
            minClass=Integer.parseInt(tfMin[0].getText());
          } catch (Exception ex) {}
          try {
            maxClass=Integer.parseInt(tfMax[0].getText());
          } catch (Exception ex) {}
          if (minClass>=0 && maxClass>=minClass)
            queryStr+="Class in ["+minClass+","+maxClass+"]; ";
        }
        if (cb[1]!=null && cb[1].isSelected()) {
          try {
            minValue=Double.parseDouble(tfMin[1].getText());
          } catch (Exception ex) {}
          try {
            maxValue=Double.parseDouble(tfMax[1].getText());
          } catch (Exception ex) {}
          if (!Double.isNaN(minValue) && maxValue>=minValue)
            queryStr+="Value in ["+minValue+","+maxValue+"]; ";
        }
        if (cb[2]!=null && cb[2].isSelected()) {
          try {
            minTreeId=Integer.parseInt(tfMin[2].getText());
          } catch (Exception ex) {}
          try {
            maxTreeId=Integer.parseInt(tfMax[2].getText());
          } catch (Exception ex) {}
          if (minTreeId>=0 && maxTreeId>=minTreeId)
            queryStr+="Tree id in ["+minTreeId+","+maxTreeId+"]; ";
        }
        if (cb[3]!=null && cb[3].isSelected()) {
          try {
            minTreeCluster=Integer.parseInt(tfMin[3].getText());
          } catch (Exception ex) {}
          try {
            maxTreeCluster=Integer.parseInt(tfMax[3].getText());
          } catch (Exception ex) {}
          if (minTreeCluster>=0 && maxTreeCluster>=minTreeCluster)
            queryStr+="Tree cluster in ["+minTreeCluster+","+maxTreeCluster+"]; ";
        }
        if (cb[4]!=null && cb[4].isSelected()) {
          category=comboCatCopy.getSelectedItem().toString();
          queryStr+="category = "+category+"; ";
        }

        HashSet<String> selFeatures=null;
        if (feaSel!=null && cbMust.isSelected()) {
          selFeatures=feaSel.getSelection();
          if (selFeatures!=null && !selFeatures.isEmpty()) {
            queryStr+="; must ";
            if (cbNot.isSelected()) queryStr+="NOT ";
            queryStr+="involve "+((rbAll.isSelected())?"all":"any")+" of ";
            for (String featureName:selFeatures)
              queryStr+=featureName+"; ";
            queryStr=queryStr.substring(0,queryStr.length()-2);
          }
        }
        if (queryStr.length()<3) {
          JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
              "The query conditions are not set properly!",
              "Invalid conditions!", JOptionPane.ERROR_MESSAGE);
          return;
        }
        if (queryStr.endsWith("; "))
          queryStr=queryStr.substring(0,queryStr.length()-2);

        selectRulesByQuery(minClass,maxClass,minValue,maxValue,
            minTreeId,maxTreeId,minTreeCluster,maxTreeCluster,category,
            selFeatures,(cbNot==null)?false:cbNot.isSelected(),(rbAll==null)?false:rbAll.isSelected());
      }
    });

    Object source=this;
    queryDialog.addWindowListener(new WindowAdapter() {
      @Override
      public void windowClosing(WindowEvent e) {
        super.windowClosing(e);
        isRunning=false;
        changeListener.stateChanged(new ChangeEvent(source));
      }
    });

    return true;
  }

  public void selectRulesByQuery(int minClass, int maxClass,
                                 double minValue, double maxValue,
                                 int minTreeId, int maxTreeId,
                                 int minTreeCluster, int maxTreeCluster,
                                 String category,
                                 HashSet<String> selectedFeatures, boolean mustNot, boolean allFeatures) {
    Object source=this;
    SwingWorker worker=new SwingWorker() {
      @Override
      public Boolean doInBackground() {
        queryExecuting=true;
        selectedRules=null;
        Vector<String> categories=null;
        if (category!=null) {
          categories=new Vector<String>(1);
          categories.add(category);
        }

        selectedRules= RuleMaster.selectByQuery(origRules,minClass,maxClass,minValue,maxValue,
            minTreeId,maxTreeId,minTreeCluster,maxTreeCluster,categories);
        if (selectedFeatures!=null && selectedRules!=null && !selectedRules.isEmpty()) {
          selectedRules=(ArrayList<CommonExplanation>)selectedRules.clone();
          for (int i=selectedRules.size()-1; i>=0; i--) {
            CommonExplanation rule=selectedRules.get(i);
            int nContained=0;
            for (String featureName:selectedFeatures)
              if (rule.hasFeature(featureName)) {
                ++nContained;
                if (!allFeatures)
                  break;
              }
            if (nContained==0 || (allFeatures && nContained<selectedFeatures.size()))
              if (!mustNot)
                selectedRules.remove(i);
              else;
            else
              if (mustNot)
                selectedRules.remove(i);
          }
        }
        return true;
      }

      @Override
      protected void done() {
        queryExecuting=false;
        if (selectedRules==null) {
          JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
              "The query result is null!",
              "Query failed!",JOptionPane.WARNING_MESSAGE);
          return;
        }
        if (changeListener!=null)
          changeListener.stateChanged(new ChangeEvent(source));
      }
    };
    System.out.println("Running rule selection in background");
    worker.execute();
  }

  public boolean mustExtract() {
    return rbExtract.isSelected();
  }

  public void toFront() {
    queryDialog.toFront();
  }

  public ArrayList<CommonExplanation> getSelectedRules() {
    return selectedRules;
  }
}
