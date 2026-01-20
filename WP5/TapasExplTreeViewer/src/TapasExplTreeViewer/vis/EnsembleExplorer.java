package TapasExplTreeViewer.vis;

import TapasDataReader.CommonExplanation;
import TapasExplTreeViewer.rules.RuleMaster;
import TapasExplTreeViewer.util.MatrixWriter;
import TapasUtilities.ItemSelectionManager;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Hashtable;

public class EnsembleExplorer implements ChangeListener {
  public ArrayList<CommonExplanation> rules=null;
  public String rulesInfoText=null;
  public ChangeListener owner=null;
  public Hashtable<String,float[]> attrMinMaxValues=null;
  public HashSet<String> featuresToUse=null;
  public int treeIds[]=null;
  public double treeDistances[][]=null;
  public JPanel uiPanel=null, mainPanel=null;
  protected JMenuItem mitExtract=null;
  protected ItemSelectionManager selector=null;
  public ArrayList<CommonExplanation> selectedRules=null;
  public String selectedRulesInfo=null, selectionInfoShort=null;
  /**
   * When a file is created, it is registered in this list, to be deleted afterwards
   */
  public ArrayList<File> createdFiles=null;

  public void setOwner(ChangeListener owner) {
    this.owner=owner;
  }

  public void setFileRegister(ArrayList<File> createdFiles) {
    this.createdFiles=createdFiles;
  }

  public JPanel startEnsembleExplorer (ArrayList<CommonExplanation> rules, String rulesInfoText,
                           Hashtable<String,float[]> attrMinMaxValues, HashSet<String> featuresToUse) {
    if (rules==null || rules.size()<5)
      return null;
    this.rules=rules; this.rulesInfoText=rulesInfoText;
    this.attrMinMaxValues=attrMinMaxValues; this.featuresToUse=featuresToUse;
    
    treeIds=RuleMaster.getAllTreeIds(rules);
    
    if (treeIds==null || treeIds.length<2) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "There are no distinct tree identifiers in the rule set!","No distinct trees!",
          JOptionPane.INFORMATION_MESSAGE);
      return null;
    }
    if (treeIds.length<5) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "There are only "+treeIds.length+" distinct tree identifiers in the rule set!",
          "Too few distinct trees!",
          JOptionPane.INFORMATION_MESSAGE);
      return null;
    }
    HashMap<Integer,Integer> treeClusters=RuleMaster.getTreeClusterIds(rules);

    uiPanel=new JPanel();
    uiPanel.setLayout(new BorderLayout());
    JTextArea textArea=new JTextArea((rulesInfoText!=null && rulesInfoText.length()>5)?rulesInfoText:
                                         "Set of "+rules.size()+" rules");
    textArea.setLineWrap(true);
    textArea.setWrapStyleWord(true);
    textArea.append("\nNumber of distinct trees: "+treeIds.length);
    if (featuresToUse!=null && !featuresToUse.isEmpty()) {
      textArea.append("\n"+featuresToUse.size()+" features are used in computing distances: ");
      int n=0;
      for (String featureName:featuresToUse) {
        textArea.append((n==0)?featureName:"; "+featureName);
        ++n;
      }
    }
    uiPanel.add(textArea,BorderLayout.NORTH);
    mainPanel=new JPanel();
    mainPanel.setLayout(new BorderLayout());
    mainPanel.add(new JLabel("Computing distances between the trees ...",JLabel.CENTER),BorderLayout.CENTER);
    System.out.println("Computing distances between the trees in background mode...");
    uiPanel.add(mainPanel,BorderLayout.CENTER);
    
    ChangeListener changeListener=this;
    
    SwingWorker worker=new SwingWorker() {
      @Override
      protected Object doInBackground() throws Exception {
        treeDistances=RuleMaster.computeDistancesBetweenTrees(rules,treeIds,featuresToUse,attrMinMaxValues);
        return null;
      }
      @Override
      protected void done() {
        mainPanel.removeAll();
        if (treeDistances==null) {
          mainPanel.add(new JLabel("Failed to compute distances between the trees!!!",JLabel.CENTER),
              BorderLayout.CENTER);
          System.out.println("Failed to compute distances between the trees!!!");
        }
        else {
          //mainPanel.add(new JLabel("Successfully computed the distances between the trees!!!",JLabel.CENTER),
              //BorderLayout.CENTER);
          System.out.println("Successfully computed the distances between the trees!!!");
          //MatrixPainter matrixPainter=new MatrixPainter(treeDistances);
          //mainPanel.add(matrixPainter,BorderLayout.CENTER);
          TSNE_Runner tsne=new TSNE_Runner();
          tsne.setFileRegister(createdFiles);
          tsne.setPerplexity(10);
          ProjectionPlot2D pp=new ProjectionPlot2D();
          pp.setDistanceMatrix(treeDistances);
          pp.setProjectionProvider(tsne);
          pp.setToChangeFrameTitle(false);
          String labels[]=new String[treeIds.length];
          for (int i=0; i<treeIds.length; i++)
            labels[i]=Integer.toString(treeIds[i]);
          pp.setLabels(labels);
          if (treeClusters!=null) {
            Color colors[]=new Color[treeIds.length];
            for (int i=0; i<treeIds.length; i++) {
              int clusterId=treeClusters.get(treeIds[i]);
              colors[i] = MyColors.getNiceColorExt(clusterId);
              colors[i]=new Color(colors[i].getRed(),colors[i].getGreen(),colors[i].getBlue(),128);
              labels[i]+="/"+clusterId;
            }
            pp.setColors(colors);
          }

          selector=pp.getSelector();
          selector.addChangeListener(changeListener);
          mainPanel.add(pp,BorderLayout.CENTER);

          JPopupMenu menu=new JPopupMenu();
          JMenuItem mit=new JMenuItem("Export the distance matrix to a file");
          menu.add(mit);
          menu.add(mit);
          mit.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
              MatrixWriter.writeMatrixToFile(treeDistances,"treeDistances.csv",true);
            }
          });

          mit=new JMenuItem("Re-run t-SNE with another perplexity setting");
          menu.add(mit);
          mit.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
              String value=JOptionPane.showInputDialog(FocusManager.getCurrentManager().getActiveWindow(),
                  "Perplexity (integer; suggested range from 5 to 50) :",
                  tsne.getPerplexity());
              if (value==null)
                return;
              try {
                int p=Integer.parseInt(value);
                if (p<5 || p>100) {
                  JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
                      "Illegal perplexity: "+p,
                      "Error",JOptionPane.ERROR_MESSAGE);
                  return;
                }
                tsne.setPerplexity(p);
                tsne.runAlgorithm();
              } catch (Exception ex) {
                JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
                    "Illegal perplexity: "+value,
                    "Error",JOptionPane.ERROR_MESSAGE);
                return;
              }
            }
          });
          mitExtract=new JMenuItem("Extract the rules of the selected trees to a new table");
          menu.add(mitExtract);
          mitExtract.setEnabled(false);
          mitExtract.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
              extractRulesOfSelectedTrees();
            }
          });
          pp.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
              super.mousePressed(e);
              if (e.getButton()>MouseEvent.BUTTON1) {
                menu.show(pp,e.getX(),e.getY());
              }
            }
          });
        }
        mainPanel.invalidate();
        mainPanel.validate();
      };
    };
    worker.execute();

    return uiPanel;
  }

  public void extractRulesOfSelectedTrees() {
    ArrayList currSel=selector.getSelected();
    if (currSel==null || currSel.isEmpty())
      return;
    HashSet<Integer> selTrees=new HashSet<Integer>(currSel.size());
    for (int i=0; i<currSel.size(); i++)
      selTrees.add(treeIds[(Integer)currSel.get(i)]);
    selectedRules=RuleMaster.selectRulesOfTrees(rules,selTrees);
    if (selectedRules==null) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Failed to select the rules!!!",
          "Failure",JOptionPane.ERROR_MESSAGE);
      return;
    }
    selectedRulesInfo=Integer.toString(selectedRules.size())+" rules originating from "+selTrees.size()+
        " interactively selected ensemble components (trees) extracted from the rule set \""+
        rulesInfoText+"\".\nThe identifiers of the selected trees: "+selTrees.toString();
    selectionInfoShort=Integer.toString(selectedRules.size())+" rules from "+selTrees.size()+
        " interactively selected trees "+selTrees.toString();
    if (owner!=null)
      owner.stateChanged(new ChangeEvent(this));
  }

  public ArrayList<CommonExplanation> getSelectedRules() {
    return selectedRules;
  }

  public String getSelectedRulesInfo() {
    return selectedRulesInfo;
  }

  public String getSelectionInfoShort() {
    return selectionInfoShort;
  }

  public void stateChanged(ChangeEvent e) {
    mitExtract.setEnabled(selector.hasSelection());
  }
}
