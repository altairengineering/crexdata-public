package TapasExplTreeViewer.ui;

import TapasDataReader.CommonExplanation;
import TapasDataReader.Explanation;
import TapasDataReader.ExplanationItem;
import TapasExplTreeViewer.MST.Edge;
import TapasExplTreeViewer.MST.Prim;
import TapasExplTreeViewer.MST.Vertex;
import TapasExplTreeViewer.clustering.*;
import TapasExplTreeViewer.rules.*;
import TapasExplTreeViewer.util.CSVDataLoader;
import TapasExplTreeViewer.util.CoordinatesReader;
import TapasExplTreeViewer.util.MatrixWriter;
import TapasExplTreeViewer.vis.*;
import TapasUtilities.*;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.LineBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

public class ShowRules implements RulesPresenter, ChangeListener {
  public static String RULES_FOLDER=null;
  
  public static Border highlightBorder=new LineBorder(ProjectionPlot2D.highlightColor,1);
  /**
   * The very original rule set (before any transformations have been applied)
   */
  public RuleSet origRules =null;
  /**
   * The rules or explanations to be visualized
   */
  public RuleSet ruleSet=null;
  /**
   * Showing rules of the rule set owned by this instance of ShowRules
   */
  public RulesTableViewer rulesView =null;
  /**
   * A common manager across all instances of ShowRules that puts rules viewers
   * in tabs of a single frame
   */
  public RulesViewerManager rulesViewerManager=null;

  /**
   * The highlighter and selector for the original rule set
   */
  public SingleHighlightManager origHighlighter=null;
  public ItemSelectionManager origSelector=null, localSelector=null;

  /**
   * The data instances (cases) the rules apply to.
   */
  public AbstractList<Explanation> dataInstances =null;
  /**
   * Rule ordering being obtained from OPTICS
   */
  protected boolean orderingInProgress=false;
  
  protected ClustererByOPTICS clOptics=null;

  protected RuleSelector ruleSelector=null;

  public ShowRules (RuleSet rs) {
    ruleSet=rs;
    init();
  }

  public ShowRules(ArrayList<CommonExplanation> exList,
                   Hashtable<String,float[]> attrMinMax,
                   double distances[][]) {
    ruleSet=RuleSet.createInstance(exList);
    ruleSet.attrMinMax=attrMinMax;
    ruleSet.distanceMatrix=distances;
    init();
  }
  
  public ShowRules(ArrayList<CommonExplanation> exList, Hashtable<String,float[]> attrMinMax) {
    this(exList,attrMinMax,null);
  }

  public void init() {
    if (ruleSet==null || !ruleSet.hasRules())
      return;
    origRules=ruleSet.getOriginalRuleSet();

    if (ruleSet.title==null) {
      int nUses=0, nCond=0;
      for (CommonExplanation cEx:ruleSet.rules) {
        nUses+=cEx.nUses;
        nCond+=cEx.eItems.length;
      }
      ruleSet.title = ((ruleSet.expanded) ? "Expanded aggregated rules " :
          (ruleSet.aggregated) ? "Aggregated rules" :
              (ruleSet.nonSubsumed) ? "Extracted non-subsumed rules" :
                  "Original distinct rules or explanations") +
          " (" + ruleSet.rules.size() + ")" +
          ", N conditions (" +nCond + ")" +
          ", Total uses (" +nUses + ")" +
          ((ruleSet.aggregated) ? "; obtained with coherence threshold " +
              String.format("%.3f", ruleSet.accThreshold) : "");
      if (ruleSet.maxQDiff > 0)
        ruleSet.title += " and max Q difference " + String.format("%.5f", ruleSet.maxQDiff);
    }

    if (ruleSet.description==null)
      ruleSet.description=ruleSet.title;

    if (ruleSet.rules.get(0).numId<0)
      for (int i=0; i<ruleSet.rules.size(); i++)
        ruleSet.rules.get(i).numId=i+1;
    if (ruleSet.distanceMatrix==null && ruleSet.rules.size()<500)
      computeDistanceMatrix();
    ToolTipManager.sharedInstance().setDismissDelay(30000);
    ToolTipManager.sharedInstance().setInitialDelay(1000);
  }
  
  public void setFeaturesAreBinary(boolean binary) {
    if (ruleSet!=null)
      ruleSet.setFeaturesAreBinary(binary);
  }

  public AbstractList<Explanation> getDataInstances() {
    return dataInstances;
  }
  
  public void setDataInstances(AbstractList<Explanation> exData, boolean actionsDiffer) {
    this.dataInstances = exData;
    if (ruleSet!=null)
      ruleSet.actionsDiffer=actionsDiffer;
  }
  
  public void countRightAndWrongRuleApplications() {
    if (dataInstances!=null && !dataInstances.isEmpty() && ruleSet!=null && ruleSet.hasRules())
      for (CommonExplanation ex:ruleSet.rules)
        ex.countRightAndWrongApplications(dataInstances,ruleSet.actionsDiffer);
  }
  public ArrayList getRules() {
    if (ruleSet!=null)
      return ruleSet.rules;
    return null;
  }

  public void setRulesViewerManager(RulesViewerManager rulesViewerManager) {
    this.rulesViewerManager = rulesViewerManager;
  }

  public void setOrigHighlighter(SingleHighlightManager origHighlighter) {
    this.origHighlighter = origHighlighter;
  }
  
  public void setOrigSelector(ItemSelectionManager origSelector) {
    this.origSelector = origSelector;
  }

  public ItemSelectionManager getLocalSelector(boolean showOriginalRules) {
    if (localSelector==null)
      localSelector=(showOriginalRules)?origSelector:new ItemSelectionManager();
    return localSelector;
  }

  protected JDialog selectFeaturesDialog=null;
  protected FeatureSelector featureSelector=null;

  public void selectFeaturesToComputeDistances() {
    if (selectFeaturesDialog!=null) {
      selectFeaturesDialog.toFront();
      return;
    }
    if (ruleSet==null || !ruleSet.hasRules() || ruleSet.rules.size()<3)
      return;
    if (ruleSet.listOfFeatures==null)
      ruleSet.listOfFeatures=RuleMaster.getListOfFeatures(ruleSet.rules);
    if (ruleSet.listOfFeatures==null || ruleSet.listOfFeatures.size()<2) {
      JOptionPane.showMessageDialog(null, "Found "+
              ((ruleSet.listOfFeatures==null)?"0":ruleSet.listOfFeatures.size())+" features!",
          "Error", JOptionPane.ERROR_MESSAGE);
      return;
    }

    selectFeaturesDialog=new JDialog(rulesViewerManager.mainFrame, "Select features",
        Dialog.ModalityType.MODELESS);
    selectFeaturesDialog.setLayout(new BorderLayout());
    selectFeaturesDialog.add(new JLabel("Select features to use in computing distances:"),
        BorderLayout.NORTH);
    featureSelector=new FeatureSelector(ruleSet.listOfFeatures,ruleSet.featuresInDistances);
    selectFeaturesDialog.getContentPane().add(featureSelector, BorderLayout.CENTER);
    JPanel bp=new JPanel(new FlowLayout(FlowLayout.CENTER, 5,5));
    selectFeaturesDialog.add(bp,BorderLayout.SOUTH);
    JButton b=new JButton("Apply");
    bp.add(b);
    b.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        ruleSet.featuresInDistances=featureSelector.getSelection();
        ruleSet.distanceMatrix=null;
        clOptics=null;
        if (JOptionPane.showConfirmDialog(featureSelector,"Re-compute distances and ordering?",
            "Re-compute?",JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE)==JOptionPane.YES_OPTION)
          computeDistanceMatrix();
      }
    });
    selectFeaturesDialog.pack();
    selectFeaturesDialog.setLocationRelativeTo(rulesViewerManager.mainFrame);
    selectFeaturesDialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
    selectFeaturesDialog.setVisible(true);
    selectFeaturesDialog.addWindowListener(new WindowAdapter() {
      @Override
      public void windowClosed(WindowEvent e) {
        super.windowClosed(e);
        selectFeaturesDialog=null;
      }
    });
  }

  public void computeDistanceMatrix(){
    if (ruleSet==null || !ruleSet.hasRules())
      return;
    if (ruleSet.distanceMatrix==null)  {
      SwingWorker worker=new SwingWorker() {
        @Override
        public Boolean doInBackground() {
          orderingInProgress=true;
          System.out.println("Computing distance matrix in background mode ...");
          ruleSet.distanceMatrix = CommonExplanation.computeDistances(ruleSet.rules,
              ruleSet.featuresInDistances, ruleSet.attrMinMax);
          return true;
        }

        @Override
        protected void done() {
          orderingInProgress=false;
          if (ruleSet.distanceMatrix == null) {
            System.out.println("Failed to compute a matrix of distances between the rules (explanations)!");
          }
          else {
            System.out.println("Distance matrix ready!");
            if (rulesView !=null)
              runOptics(rulesView.getTableModel());
          }
        }
      };
      worker.execute();
    }
  }

  public void runOptics(ChangeListener changeListener){
    if (orderingInProgress || clOptics!=null)
      return;
    if (ruleSet==null || !ruleSet.hasRules())
      return;
    clOptics=(ruleSet.distanceMatrix!=null && ruleSet.distanceMatrix.length>5)?new ClustererByOPTICS():null;
    if (clOptics!=null) {
      boolean showOriginalRules=ruleSet.equals(origRules);
      SingleHighlightManager highlighter=(showOriginalRules)?origHighlighter:new SingleHighlightManager();
      clOptics.setDistanceMatrix(ruleSet.distanceMatrix);
      clOptics.setHighlighter(highlighter);
      clOptics.setSelector(getLocalSelector(showOriginalRules));
      clOptics.addChangeListener(changeListener);
      clOptics.addChangeListener(this);
      System.out.println("Running OPTICS clustering in background mode ...");
      orderingInProgress=true;
      clOptics.doClustering();
    }
  }

  public RulesTableViewer showRulesInTable() {
    if (rulesViewerManager==null) {
      rulesViewerManager = new RulesViewerManager();
      rulesViewerManager.dataFolder=RULES_FOLDER;
    }

    boolean showOriginalRules=ruleSet.equals(origRules);
    if (showOriginalRules)
      origRules.determinePredictionRanges();

    if (showOriginalRules && origHighlighter==null) {
      origHighlighter=new SingleHighlightManager();
      origSelector=new ItemSelectionManager();
    }

    SingleHighlightManager highlighter=(showOriginalRules)?origHighlighter:new SingleHighlightManager();
    ItemSelectionManager selector=getLocalSelector(showOriginalRules);

    int minA=origRules.minAction, maxA=origRules.maxAction;
    double minQ=origRules.minQValue, maxQ=origRules.maxQValue;
  
    rulesView =new RulesTableViewer(ruleSet,highlighter,selector);
    rulesViewerManager.addRulesViewer(rulesView);

    ExListTableModel tblModel= rulesView.getTableModel();
    JTable table= rulesView.getTable();

    JPopupMenu menu=new JPopupMenu();

    JCheckBoxMenuItem cbmit=new JCheckBoxMenuItem("Show texts for intervals",false);
    menu.add(cbmit);
    cbmit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        for (int i=0; i<tblModel.listOfFeatures.size(); i++) {
          TableCellRenderer tcr=table.getColumnModel().
              getColumn(tblModel.listOfColumnNames.size()+i).getCellRenderer();
          if (tcr instanceof JLabel_Subinterval) {
            ((JLabel_Subinterval)tcr).setDrawTexts(cbmit.getState());
          }
        }
        tblModel.fireTableDataChanged();
      }
    });
    JCheckBoxMenuItem cbmit1=new JCheckBoxMenuItem("Mark values",false);
    JCheckBoxMenuItem cbmit2=new JCheckBoxMenuItem("Show statistics",false);
    menu.add(cbmit1);
    cbmit1.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        for (int i=0; i<tblModel.listOfFeatures.size(); i++) {
          TableCellRenderer tcr=table.getColumnModel().getColumn(
              tblModel.listOfColumnNames.size()+i).getCellRenderer();
          if (tcr instanceof JLabel_Subinterval) {
            ((JLabel_Subinterval)tcr).setDrawValues(cbmit1.getState());
          }
        }
        tblModel.setDrawValuesOrStatsForIntervals(cbmit1.getState() || cbmit2.getState());
        tblModel.fireTableDataChanged();
      }
    });
    menu.add(cbmit2);
    cbmit2.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        for (int i=0; i<tblModel.listOfFeatures.size(); i++) {
          TableCellRenderer tcr=table.getColumnModel().
              getColumn(tblModel.listOfColumnNames.size()+i).getCellRenderer();
          if (tcr instanceof JLabel_Subinterval) {
            ((JLabel_Subinterval)tcr).setDrawStats(cbmit2.getState());
          }
        }
        tblModel.setDrawValuesOrStatsForIntervals(cbmit1.getState() || cbmit2.getState());
        tblModel.fireTableDataChanged();
      }
    });
    JMenuItem mit=new JMenuItem("Table -> Clipboard");
    menu.add(mit);
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        tblModel.putTableToClipboard();
      }
    });
    menu.addSeparator();
  
    mit=new JMenuItem("Represent rules by glyphs");
    menu.add(mit);
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        showRuleGlyphs(ruleSet.rules,rulesView.getAttrs(),highlighter,selector);
      }
    });

    menu.add(mit=new JMenuItem("Show distributions of feature values"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        showFeatureIntervalsDistributions();
      }
    });

    menu.add(mit=new JMenuItem("Encode rules for topic modelling"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        //applyTopicModelling();
        encodeRulesForTopicModelling();
      }
    });

    menu.addSeparator();
    menu.add(mit=new JMenuItem("Select feature subset for computing distances"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        selectFeaturesToComputeDistances();
      }
    });

    if (RuleMaster.hasDistinctTreeIds(ruleSet.rules)) {
      menu.add(mit = new JMenuItem("Explore similarities and distances between the trees"));
      mit.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          openTreeExplorer();
        }
      });
    }

    menu.add(mit=new JMenuItem("Show the OPTICS reachability plot"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (orderingInProgress) {
          JOptionPane.showMessageDialog(null, "Ordering is in progress.",
              "Wait...", JOptionPane.WARNING_MESSAGE);
          return;
        }
        if (clOptics!=null)
          rulesView.addFrame(clOptics.showPlot());
        else {
          if (ruleSet.distanceMatrix==null)
            computeDistanceMatrix();
          else
            runOptics((ExListTableModel) table.getModel());
          if (clOptics!=null)
            rulesView.addFrame(clOptics.showPlot());
        }
      }
    });
    
    menu.add(mit=new JMenuItem("Show the t-SNE projection"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        showProjection(ruleSet.rules,ruleSet.distanceMatrix,highlighter,selector);
      }
    });

    menu.add(mit=new JMenuItem("Apply hierarchical clustering"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        hierClustering(ruleSet.rules,ruleSet.distanceMatrix,minA,maxA,minQ,maxQ);
      }
    });

    if (ruleSet.aggregated && !ruleSet.expanded) {
      menu.add(mit=new JMenuItem("Expand rule hierarchies"));
      mit.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          boolean applyToSelection=
              selector.hasSelection() &&
                  JOptionPane.showConfirmDialog(FocusManager.getCurrentManager().getActiveWindow(),
                      "Apply the operation to the selected subset?",
                      "Apply to selection?",JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE)
                      ==JOptionPane.YES_OPTION;
          ArrayList rules=(applyToSelection)?getSelectedRules(ruleSet.rules,selector):ruleSet.rules;
          ArrayList<UnitedRule> expanded=RuleMaster.expandRuleHierarchies(rules);
          if (expanded!=null) {
            ArrayList<CommonExplanation> ex=new ArrayList<CommonExplanation>(expanded.size());
            ex.addAll(expanded);
            ShowRules showRules=createShowRulesInstance(ex);
            showRules.ruleSet.setNonSubsumed(true);
            showRules.ruleSet.setAggregated(true);
            showRules.ruleSet.setExpanded(true);
            showRules.ruleSet.orderedFeatureNames=ruleSet.orderedFeatureNames;
            showRules.ruleSet.description=ex.size()+" rules obtained by expanded hierarchies of "+
                rules.size()+((applyToSelection)?" selected":"")+" aggregated rules";
            showRules.countRightAndWrongRuleApplications();
            showRules.showRulesInTable();
          }
        }
      });
      menu.add(mit=new JMenuItem("Show the links between the original rules in a t-SNE projection"));
      mit.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
         showAggregationInProjection(selector);
        }
      });
    }
    
    JMenuItem mitExtract=new JMenuItem("Extract the (non-) selected subset to a separate view");
    menu.add(mitExtract);
    mitExtract.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        extractSubset(selector,ruleSet.rules,ruleSet.distanceMatrix);
      }
    });

    menu.add(mit=new JMenuItem("Clear selection"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        selector.deselectAll();
      }
    });
    
    if (!ruleSet.expanded) {
      menu.addSeparator();
      menu.add(mit = new JMenuItem("Remove contradictory rules"));
      mit.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          removeContradictory(ruleSet.rules);
        }
      });
      menu.add(mit = new JMenuItem("Extract the non-subsumed rules to a separate view"));
      mit.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          getNonSubsumed(ruleSet.rules, ruleSet.attrMinMax);
        }
      });
      menu.add(mit = new JMenuItem("Select or extract rule subset through a query"));
      ChangeListener changeListener=this;
      mit.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          if (ruleSelector!=null && ruleSelector.isRunning)
            ruleSelector.toFront();
          else {
            ruleSelector=new RuleSelector();
            if (!ruleSelector.makeQueryInterface(ruleSet.rules,
                ruleSet.listOfFeatures,ruleSet.featuresInDistances,changeListener))
              ruleSelector=null;
          }
        }
      });

      menu.addSeparator();
      menu.add(mit = new JMenuItem("Aggregate and generalise rules"));
      mit.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          aggregate(ruleSet.rules, ruleSet.attrMinMax);
        }
      });
    }
    menu.addSeparator();
    JMenuItem mitExportData=new JMenuItem("Export previously loaded data");
    mitExportData.setEnabled(rulesViewerManager.currentData!=null);
    mitExportData.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        exportDataToCSVFile();
      }
    });

    menu.add(mit=new JMenuItem("Apply rules to data"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        applyRulesToData();
        if (rulesViewerManager.currentData!=null)
          mitExportData.setEnabled(true);
      }
    });

    menu.add(mit=new JMenuItem("Load (other) data from file"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        DataSet data=loadData();
        if (data!=null) {
          rulesViewerManager.currentData=data;
          mitExportData.setEnabled(true);
        }
      }
    });

    menu.add(mitExportData);

    menu.addSeparator();
    menu.add(mit=new JMenuItem("Export rules to file in text format"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        boolean applyToSelection=
            selector.hasSelection() &&
                JOptionPane.showConfirmDialog(FocusManager.getCurrentManager().getActiveWindow(),
                    "Apply the operation to the selected subset?",
                    "Apply to selection?",JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE)
                    ==JOptionPane.YES_OPTION;
        ArrayList rules=(applyToSelection)?getSelectedRules(ruleSet.rules,selector):ruleSet.rules;
        // Select file to save
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Specify a file to save");
        fileChooser.setCurrentDirectory(new File(RULES_FOLDER));
        FileNameExtensionFilter filter = new FileNameExtensionFilter("Text file", "txt");
        fileChooser.setFileFilter(filter);
        int userSelection = fileChooser.showSaveDialog(null);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
          File fileToSave = fileChooser.getSelectedFile();
          String fName=fileToSave.getName().toLowerCase();
          if (!fName.endsWith(".txt") && !fName.endsWith(".csv"))
            fileToSave = new File(fileToSave.getAbsolutePath() + ".txt");
          boolean ok=RuleMaster.exportRulesToFile(fileToSave, rules);
          if (ok)
            JOptionPane.showMessageDialog(null, "Rules exported successfully!");
          else
            JOptionPane.showMessageDialog(null, "Failed to export rules.",
                "Error", JOptionPane.ERROR_MESSAGE);
        }
      }
    });
    menu.add(mit=new JMenuItem("Export rules to file in structured (table) format"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        boolean applyToSelection=
            selector.hasSelection() &&
                JOptionPane.showConfirmDialog(FocusManager.getCurrentManager().getActiveWindow(),
                    "Apply the operation to the selected subset?",
                    "Apply to selection?",JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE)
                    ==JOptionPane.YES_OPTION;
        ArrayList rules=(applyToSelection)?getSelectedRules(ruleSet.rules,selector):ruleSet.rules;
        // Select file to save
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Specify a file to save the table");
        fileChooser.setCurrentDirectory(new File(RULES_FOLDER));
        // Set the file extension filter
        FileNameExtensionFilter filter = new FileNameExtensionFilter("CSV file", "csv");
        fileChooser.setFileFilter(filter);
        int userSelection = fileChooser.showSaveDialog(null);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
          File fileToSave = fileChooser.getSelectedFile();
          String fName=fileToSave.getName().toLowerCase();
          if (!fName.endsWith(".csv"))
            fileToSave = new File(fileToSave.getAbsolutePath() + ".csv");
          boolean ok=RuleMaster.exportRulesToTable(fileToSave, rules,ruleSet.attrMinMax);
          if (ok)
            JOptionPane.showMessageDialog(null, "Rules exported successfully!");
          else
            JOptionPane.showMessageDialog(null, "Failed to export rules.",
                "Error", JOptionPane.ERROR_MESSAGE);
        }
      }
    });

    menu.addSeparator();
    if (rulesViewerManager.getViewerIndex(rulesView)>0) {
      menu.add(mit=new JMenuItem("Remove this rule set"));
      mit.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
          rulesViewerManager.removeRulesViewer(rulesView);
          rulesView.closeAllFrames();
        }
      });
    }

    menu.add(mit=new JMenuItem("Quit"));
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (rulesViewerManager.mayQuit()) {
          rulesViewerManager.eraseCreatedFiles();
          System.exit(0);
        }
      }
    });
    
    table.addMouseListener(new MouseAdapter() {
      @Override
      public void mousePressed(MouseEvent e) {
        super.mousePressed(e);
        if (e.getButton()>MouseEvent.BUTTON1) {
          ArrayList selected=selector.getSelected();
          mitExtract.setEnabled(selected!=null && selected.size()>0);
          menu.show(table,e.getX(),e.getY());
        }
      }
      @Override
      public void mouseClicked(MouseEvent e) {
        super.mouseClicked(e);
        if (e.getButton()>MouseEvent.BUTTON1)
          return;
        if (e.getClickCount()>1) {
          selector.deselectAll();
          return;
        }
        JPopupMenu selMenu=new JPopupMenu();;
        int rowIndex=table.rowAtPoint(e.getPoint());
        if (rowIndex<0)
          return;
        int realRowIndex = table.convertRowIndexToModel(rowIndex);
        if (ruleSet.getRule(realRowIndex) instanceof UnitedRule) {
          UnitedRule rule=(UnitedRule)ruleSet.getRule(realRowIndex);
          if (rule.fromRules!=null && !rule.fromRules.isEmpty()) {
            JMenuItem selItem = new JMenuItem("Show derivation hierarchy of rule " + rule.numId);
            selMenu.add(selItem);
            selItem.addActionListener(new ActionListener() {
              @Override
              public void actionPerformed(ActionEvent e) {
                showRuleHierarchy(rule,rulesView.getAttrs());
              }
            });
          }
        }
        if (ruleSet.getRule(realRowIndex) instanceof UnitedRule) {
          UnitedRule rule=(UnitedRule)ruleSet.getRule(realRowIndex);
          if (origRules!=null) {
            //selMenu = new JPopupMenu();
            JMenuItem selItem = new JMenuItem("Extract all coverages of rule " + rule.numId);
            selMenu.add(selItem);
            selItem.addActionListener(new ActionListener() {
              @Override
              public void actionPerformed(ActionEvent e) {
                ArrayList<CommonExplanation> valid=rule.extractValidCoverages(origRules.rules,
                    !origRules.actionsDiffer);
                ArrayList<CommonExplanation> wrong=rule.extractWrongCoverages(origRules.rules,
                    !origRules.actionsDiffer);
                if ((valid==null || valid.isEmpty()) && (wrong==null || wrong.isEmpty())) {
                  JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
                      "Neither valid nor invalid coverages found!",
                      "Very strange!", JOptionPane.INFORMATION_MESSAGE);
                  return;
                }
                int n=1;
                if (valid!=null) n+=valid.size();
                if (wrong!=null) n+=wrong.size();
                ArrayList<CommonExplanation> all=new ArrayList<CommonExplanation>(n);
                if (valid!=null)
                  all.addAll(valid);
                if (wrong!=null)
                  all.addAll(wrong);
                all.add(0,rule);
                ShowRules showRules=createShowRulesInstance(all);
                showRules.ruleSet.title=showRules.ruleSet.description="All coverages of rule # "+rule.numId;
                showRules.showRulesInTable();
              }
            });
            selMenu.add(selItem = new JMenuItem("Extract invalid coverages of rule " + rule.numId));
            selItem.addActionListener(new ActionListener() {
              @Override
              public void actionPerformed(ActionEvent e) {
                ArrayList<CommonExplanation> wrong=rule.extractWrongCoverages(origRules.rules,
                    !origRules.actionsDiffer);
                if (wrong==null || wrong.isEmpty()) {
                  JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
                      "No invalid coverages found!",
                      "No exceptions", JOptionPane.INFORMATION_MESSAGE);
                  return;
                }
                wrong.add(0,rule);
                ShowRules showRules=createShowRulesInstance(wrong);
                showRules.ruleSet.title=showRules.ruleSet.description="Invalid coverages of rule # "+rule.numId;
                showRules.showRulesInTable();
              }
            });
            selMenu.add(selItem = new JMenuItem("Extract valid coverages of rule " + rule.numId));
            selItem.addActionListener(new ActionListener() {
              @Override
              public void actionPerformed(ActionEvent e) {
                ArrayList<CommonExplanation> valid=rule.extractValidCoverages(origRules.rules,
                    !origRules.actionsDiffer);
                if (valid==null || valid.isEmpty()) {
                  JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
                      "No valid coverages found!",
                      "Very strange!", JOptionPane.INFORMATION_MESSAGE);
                  return;
                }
                valid.add(0,rule);
                ShowRules showRules=createShowRulesInstance(valid);
                showRules.ruleSet.title=showRules.ruleSet.description="Valid coverages of rule # "+rule.numId;
                showRules.showRulesInTable();
              }
            });
          }
          if (ruleSet.expanded) {
            if (rule.upperId >= 0) {
              //if (selMenu == null)
                //selMenu = new JPopupMenu();
              JMenuItem selItem = new JMenuItem("Select rules that include rule " + rule.numId);
              selMenu.add(selItem);
              selItem.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                  ArrayList<Integer> toSelect = new ArrayList<Integer>(10);
                  UnitedRule r = rule;
                  do {
                    int idx = RuleMaster.findRuleInList(ruleSet.rules, r.upperId);
                    if (idx >= 0) {
                      toSelect.add(new Integer(idx));
                      r = (UnitedRule) ruleSet.getRule(idx);
                    }
                    else
                      r = null;
                  } while (r != null && r.upperId >= 0);
                  if (!toSelect.isEmpty())
                    selector.select(toSelect);
                }
              });
            }

            if (rule.fromRules != null && !rule.fromRules.isEmpty()) {
              //if (selMenu == null)
                //selMenu = new JPopupMenu();
              JMenuItem selItem = new JMenuItem("Select rules directly included in " + rule.numId);
              selMenu.add(selItem);
              selItem.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                  ArrayList<Integer> toSelect = new ArrayList<Integer>(10);
                  for (int i = 0; i < rule.fromRules.size(); i++) {
                    int idx = RuleMaster.findRuleInList(ruleSet.rules, rule.fromRules.get(i).numId);
                    if (idx >= 0)
                      toSelect.add(idx);
                  }
                  if (!toSelect.isEmpty())
                    selector.select(toSelect);
                }
              });
              selMenu.add(selItem = new JMenuItem("Select all rules included in " + rule.numId));
              selItem.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                  ArrayList<UnitedRule> included = rule.putHierarchyInList(null);
                  ArrayList<Integer> toSelect = new ArrayList<Integer>(10);
                  for (int i = 0; i < included.size(); i++) {
                    int idx = RuleMaster.findRuleInList(ruleSet.rules, included.get(i).numId);
                    if (idx >= 0)
                      toSelect.add(idx);
                  }
                  if (!toSelect.isEmpty())
                    selector.select(toSelect);
                }
              });
            }
          }
          //if (selMenu==null)
            //return;
        }
        if (rulesViewerManager.loadedData!=null && !rulesViewerManager.loadedData.isEmpty()) {
          CommonExplanation rule = ruleSet.getRule(realRowIndex);
          String title = "Select data for the rule " +
              rule.numId + ", action=" + rule.action +
              ", N records=" + rule.nUses;
          JMenuItem selItem = new JMenuItem(title);
          selItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
              DataSet testData=rulesViewerManager.loadedData.get(0);
              if (rulesViewerManager.loadedData.size()>1) {
                //allow the user to select the data version (the versions differ in the classes treated as original)
                String options[]=new String[rulesViewerManager.loadedData.size()];
                for (int i=0; i<rulesViewerManager.loadedData.size(); i++)
                  options[i]=rulesViewerManager.loadedData.get(i).versionLabel;
                JList list=new JList(options);
                list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
                list.setSelectedIndex(rulesViewerManager.loadedData.size()-1);
                JScrollPane scrollPane = new JScrollPane(list);
                JPanel p=new JPanel();
                p.setLayout(new BorderLayout());
                p.add(scrollPane,BorderLayout.CENTER);
                p.add(new JLabel("Select one of the loaded data sets"),
                    BorderLayout.NORTH);
                // Show the JOptionPane with the list inside
                int result = JOptionPane.showConfirmDialog(
                    rulesViewerManager.mainFrame,
                    scrollPane,
                    "Select dataset version",
                    JOptionPane.OK_CANCEL_OPTION,
                    JOptionPane.PLAIN_MESSAGE
                );
                if (result!=JOptionPane.OK_OPTION)
                  return;
                testData=rulesViewerManager.loadedData.get(list.getSelectedIndex());
              }
              ArrayList<DataRecord> data=RuleMaster.selectDataForRule(rule,testData);
              if (data==null) {
                JOptionPane.showMessageDialog(rulesViewerManager.mainFrame,
                    "No data found for the selected rule!","No data found!",
                    JOptionPane.INFORMATION_MESSAGE);
                return;
              }
              DataSet ds=new DataSet();
              testData.copyFields(ds);
              ds.description="Selected data records satisfying the conditions of the rule\n"+
                  rule.toString();
              ds.records=data;
              DataTableViewer dViewer=new DataTableViewer(ds,
                  ruleSet.listOfFeatures.toArray(new String[ruleSet.listOfFeatures.size()]),
                  ShowRules.this);
              Dimension size=Toolkit.getDefaultToolkit().getScreenSize();
              JFrame frame=new JFrame("Selected data for rule "+rule.numId);
              frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
              frame.getContentPane().add(dViewer);
              frame.pack();
              frame.setSize((int)Math.min(frame.getWidth(),0.7*size.width),
                  (int)Math.min(frame.getHeight(),0.7*size.height));
              frame.setLocationRelativeTo(rulesViewerManager.mainFrame);
              frame.setVisible(true);
              rulesView.addFrame(frame);
            }
          });
          /*
          selItem.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
              DataForRuleTableModel dataTblModel =
                  new DataForRuleTableModel(rule, tblModel.listOfFeatures, ruleSet.attrMinMax);
              JTable dataTbl = new JTable(dataTblModel);
              Dimension size = Toolkit.getDefaultToolkit().getScreenSize();
              dataTbl.setPreferredScrollableViewportSize(
                  new Dimension(Math.round(size.width * 0.7f), Math.round(size.height * 0.8f)));
              dataTbl.setFillsViewportHeight(true);
              dataTbl.setAutoCreateRowSorter(true);
              dataTbl.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
              dataTbl.setRowSelectionAllowed(true);
              dataTbl.setColumnSelectionAllowed(false);
              for (int i = dataTblModel.columnNames.length; i < dataTblModel.getColumnCount(); i++) {
                JLabel_Subinterval renderer = new JLabel_Subinterval();
                renderer.setDrawValues(true);
                //renderer.setDrawTexts(true);
                dataTbl.getColumnModel().getColumn(i).setCellRenderer(renderer);
              }
              JScrollPane scrollPane = new JScrollPane(dataTbl);
              JFrame fr = new JFrame(title);
              fr.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
              fr.getContentPane().add(scrollPane, BorderLayout.CENTER);
              //Display the window.
              fr.pack();
              fr.setLocationRelativeTo(rulesViewerManager.mainFrame);
              fr.setVisible(true);
            }
          });
          */
          selMenu.addSeparator();
          selMenu.add(selItem);
        }
        selMenu.addSeparator();
        selMenu.add(new JMenuItem("Cancel"));
        selMenu.show(table,e.getX(),e.getY());
      }
    });

    return rulesView;
  }

  public JFrame getMainFrame() {
    if (rulesViewerManager==null)
      return null;
    return rulesViewerManager.mainFrame;
  }
  
  protected ShowRules createShowRulesInstance(ArrayList rules) {
    RuleSet rs=RuleSet.createInstance(rules);
    rs.determinePredictionRanges();
    ruleSet.addChild(rs);

    rs.minAction=Integer.MAX_VALUE; rs.maxAction=Integer.MIN_VALUE;
    rs.minQValue=Double.NaN; rs.maxQValue=Double.NaN;
    rs.determinePredictionRanges();

    ShowRules showRules=new ShowRules(rs);
    showRules.setRulesViewerManager(rulesViewerManager);
    showRules.setDataInstances(dataInstances,rs.actionsDiffer);
    showRules.setOrigHighlighter(origHighlighter);
    showRules.setOrigSelector(origSelector);
    rs.setAccThreshold(ruleSet.getAccThreshold());
    rs.setNonSubsumed(ruleSet.nonSubsumed);
    rs.setAggregated(ruleSet.aggregated);
    if (!Double.isNaN(ruleSet.maxQDiff))
      rs.setMaxQDiff(ruleSet.maxQDiff);
    rs.setExpanded(ruleSet.expanded);
    return showRules;
  }
  
  public void showRules(ArrayList rules, String infoText){
    ShowRules showRules=createShowRulesInstance(rules);
    showRules.ruleSet.setTitle(infoText);
    showRules.showRulesInTable();
  }
  
  
  public ArrayList getSelectedRules(ArrayList allRules, ItemSelectionManager selector) {
    if (allRules==null || allRules.isEmpty() || selector==null || !selector.hasSelection())
      return null;
    ArrayList selected=selector.getSelected();
    if (selected==null || selected.isEmpty())
      return null;
    ArrayList result=new ArrayList(selected.size());
    for (int i=0; i<selected.size(); i++) {
      int idx=(Integer)selected.get(i);
      result.add(allRules.get(idx));
    }
    if (result.isEmpty())
      return null;
    return result;
  }

  public JFrame showRuleGlyphs(ArrayList exList,
                               Vector<String> attributes,
                               SingleHighlightManager highlighter,
                               ItemSelectionManager selector) {
    boolean applyToSelection=
        selector.hasSelection() &&
            JOptionPane.showConfirmDialog(FocusManager.getCurrentManager().getActiveWindow(),
                "Apply the operation to the selected subset?",
                "Apply to selection?",JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE)
                ==JOptionPane.YES_OPTION;
    ArrayList rules=(applyToSelection)?getSelectedRules(exList,selector):exList;
    RuleSetVis vis=new RuleSetVis(rules,exList,attributes,ruleSet.attrMinMax);
    vis.setOrigExList(origRules.rules);
    vis.setHighlighter(highlighter);
    vis.setSelector(selector);
    vis.setRulesOrderer(rulesView);
  
    JScrollPane scrollPane = new JScrollPane(vis);
    scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
    scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);

    Dimension size=Toolkit.getDefaultToolkit().getScreenSize();
    JFrame plotFrame=new JFrame(ruleSet.title+((applyToSelection)?" (selection)":""));
    plotFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    plotFrame.getContentPane().add(scrollPane);
    plotFrame.pack();
    plotFrame.setSize((int)Math.min(plotFrame.getWidth(),0.7*size.width),
        (int)Math.min(plotFrame.getHeight(),0.7*size.height));
    plotFrame.setLocation(size.width-plotFrame.getWidth()-30, size.height-plotFrame.getHeight()-50);
    plotFrame.setVisible(true);
    rulesView.addFrame(plotFrame);
    return plotFrame;
  }
  
  public JFrame showRuleHierarchy(UnitedRule rule,Vector<String> attributes) {
    RuleHierarchyVis vis=new RuleHierarchyVis(rule,origRules.rules,attributes,ruleSet.attrMinMax);

    Dimension size=Toolkit.getDefaultToolkit().getScreenSize();
    JFrame plotFrame=new JFrame("Derivation hierarchy of rule "+rule.numId);
    plotFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    Dimension d=vis.getPreferredSize();
    if (d.width>0.7*size.width || d.height>0.7*size.height) {
      ScrollPane scrollPane = new ScrollPane(ScrollPane.SCROLLBARS_AS_NEEDED);
      scrollPane.add(vis);
      scrollPane.setPreferredSize(new Dimension((int)Math.min(d.getWidth()+30,0.75*size.width),
          (int)Math.min(d.getHeight()+30,0.75*size.height)));
      plotFrame.getContentPane().add(scrollPane);
    }
    else
      plotFrame.getContentPane().add(vis);
    plotFrame.pack();
    plotFrame.setSize((int)Math.min(plotFrame.getWidth(),0.75*size.width),
        (int)Math.min(plotFrame.getHeight(),0.75*size.height));
    plotFrame.setLocation(size.width-plotFrame.getWidth()-30, size.height-plotFrame.getHeight()-50);
    plotFrame.setVisible(true);
    rulesView.addFrame(plotFrame);
    return plotFrame;
  }

  public void encodeRulesForTopicModelling () {
    EncodeRulesForTopicModelling.encode(ruleSet,rulesViewerManager.dataFolder);
  }

  public JFrame showProjection(ArrayList<CommonExplanation> exList,
                               double distanceMatrix[][],
                               SingleHighlightManager highlighter,
                               ItemSelectionManager selector){
    if (distanceMatrix==null) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "No distance matrix!",
          "Error",JOptionPane.ERROR_MESSAGE);
      return null;
    }
    TSNE_Runner tsne=new TSNE_Runner();
    tsne.setFileRegister(rulesViewerManager.createdFiles);
    String value=JOptionPane.showInputDialog(FocusManager.getCurrentManager().getActiveWindow(),
        "Perplexity (integer; suggested range from 5 to 50) :",
        tsne.getPerplexity());
    if (value==null)
      return null;
    try {
      int p=Integer.parseInt(value);
      if (p<5 || p>100) {
        JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
            "Illegal perplexity: "+p,
            "Error",JOptionPane.ERROR_MESSAGE);
        return null;
      }
      tsne.setPerplexity(p);
    } catch (Exception ex) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Illegal perplexity!"+value,
          "Error",JOptionPane.ERROR_MESSAGE);
      return null;
    }

    ExplanationsProjPlot2D pp=new ExplanationsProjPlot2D(ruleSet.attrMinMax,
        rulesView.getAttrs(),rulesView.getMinMax());
    pp.setExplanations(exList,origRules.rules);
    pp.setDistanceMatrix(distanceMatrix);
    pp.setFeatures(ruleSet.getSelectedFeatures());
    pp.setProjectionProvider(tsne);
    pp.setHighlighter(highlighter);
    pp.setSelector(selector);
    pp.setPreferredSize(new Dimension(800,800));
    
    if (ruleSet.expanded && (exList.get(0) instanceof UnitedRule)){
      HashSet<ArrayList<Vertex>> graphs=null;
      for (int i=0; i<exList.size(); i++) {
        UnitedRule rule=(UnitedRule)exList.get(i);
        if (rule.upperId>=0 || rule.fromRules==null || rule.fromRules.isEmpty())
          continue;
        ArrayList<UnitedRule> hList=rule.putHierarchyInList(null);
        if (hList==null || hList.size()<2)
          continue;
        ArrayList<Vertex> graph=new ArrayList<Vertex>(hList.size());
        Hashtable<String,Integer> vertIndexes=new Hashtable<String,Integer>(hList.size());
        for (int j=0; j<hList.size(); j++) {
          UnitedRule r=hList.get(j);
          String label=Integer.toString(r.numId);
          vertIndexes.put(label,j);
          graph.add(new Vertex(label));
        }
        for (int j=0; j<hList.size(); j++) {
          UnitedRule r=hList.get(j);
          if (r.upperId<0)
            continue;
          Edge e=new Edge(1);
          e.setIncluded(true);
          Vertex v1=graph.get(j), v2=graph.get(vertIndexes.get(Integer.toString(r.upperId)));
          v1.addEdge(v2,e);
          v2.addEdge(v1,e);
        }
        if (graphs==null)
          graphs=new HashSet<ArrayList<Vertex>>(exList.size()/10);
        graphs.add(graph);
      }
      if (graphs!=null)
        pp.setGraphs(graphs);
    }
  
    Dimension size=Toolkit.getDefaultToolkit().getScreenSize();
    JFrame plotFrame=new JFrame(pp.getProjectionProvider().getProjectionTitle());
    plotFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    plotFrame.getContentPane().add(pp);
    plotFrame.pack();
    plotFrame.setLocation(size.width-plotFrame.getWidth()-30, size.height-plotFrame.getHeight()-50);
    plotFrame.setVisible(true);
    rulesView.addFrame(plotFrame);

    JPopupMenu menu=new JPopupMenu();
    JMenuItem mitExtract=new JMenuItem("Extract the selected subset to a separate view");
    menu.add(mitExtract);
    mitExtract.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        extractSubset(selector,exList,distanceMatrix);
      }
    });
  
    JMenuItem mit=new JMenuItem("Read point coordinates from a file");
    menu.add(mit);
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        double coords[][]= CoordinatesReader.readCoordinatesFromChosenFile();
        if (coords==null) {
          JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
              "No coordinates could be read!",
              "Error",JOptionPane.ERROR_MESSAGE);
          return;
        }
        if (coords.length!=exList.size()) {
          JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
              "The new coordinates are for "+coords.length+
                                 " points but must be for "+exList.size()+" points!",
              "Error",JOptionPane.ERROR_MESSAGE);
          return;
        }
        System.out.println("Trying to create another plot...");
        ExplanationsProjPlot2D anotherPlot=
            new ExplanationsProjPlot2D(ruleSet.attrMinMax,
                rulesView.getAttrs(),rulesView.getMinMax(),exList,origRules.rules,coords);
        anotherPlot.setPreferredSize(new Dimension(800,800));
        anotherPlot.setSelector(selector);
        anotherPlot.setHighlighter(highlighter);
      
        anotherPlot.addMouseListener(new MouseAdapter() {
          @Override
          public void mousePressed(MouseEvent e) {
            super.mousePressed(e);
            if (e.getButton()>MouseEvent.BUTTON1) {
              ArrayList selected=selector.getSelected();
              mitExtract.setEnabled(selected!=null && selected.size()>0);
              menu.show(anotherPlot,e.getX(),e.getY());
            }
          }
        });
      
        JFrame fr=new JFrame(CoordinatesReader.lastFileName);
        fr.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        fr.getContentPane().add(anotherPlot);
        fr.pack();
        Point p=plotFrame.getLocationOnScreen();
        fr.setLocation(Math.max(10,p.x-30), Math.max(10,p.y-50));
        fr.setVisible(true);
        rulesView.addFrame(fr);
      }
    });
  
    mit=new JMenuItem("Export the distance matrix to a file");
    menu.add(mit);
    mit.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        MatrixWriter.writeMatrixToFile(distanceMatrix,"allDistances.csv",true);
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
  
    pp.addMouseListener(new MouseAdapter() {
      @Override
      public void mousePressed(MouseEvent e) {
        super.mousePressed(e);
        if (e.getButton()>MouseEvent.BUTTON1) {
          ArrayList selected=selector.getSelected();
          mitExtract.setEnabled(selected!=null && selected.size()>0);
          menu.show(pp,e.getX(),e.getY());
        }
      }
    });
    return plotFrame;
  }
  
  public JFrame hierClustering (ArrayList<CommonExplanation> exList,
                                double distanceMatrix[][],
                                int minA, int maxA,
                                double minQ, double maxQ) {
    if (distanceMatrix==null)
      return null;
  
    String options[]={"mean pairwise distance between members",
                      "distance between cluster medoids"};
    String selOption = (String)JOptionPane.showInputDialog(FocusManager.getCurrentManager().getActiveWindow(),
        "How to compute the distances between clusters?",
        "What is the distance between clusters?",JOptionPane.QUESTION_MESSAGE,
        null,options,options[0]);
    
    ClusterContent topCluster=HierarchicalClusterer.doClustering(distanceMatrix,selOption.equals(options[1]));
    if (topCluster==null) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Clustering failed!",
          "Error",JOptionPane.ERROR_MESSAGE);
      return null;
    }

    int oIds[]=new int[exList.size()];
    for (int i=0; i<exList.size(); i++)
      oIds[i]=exList.get(i).numId;
    topCluster.setObjIds(oIds);

    JFrame frame = new JFrame("Cluster hierarchy; depth = "+topCluster.hierDepth+"; "+selOption);
    frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    Choice ch=new Choice();
    for (int i=0; i<topCluster.hierDepth+1; i++)
      ch.add("level "+i+": "+topCluster.getNClustersAtLevel(i)+" clusters");
    ch.select(2);
    putHierClustersToTable(topCluster, ch.getSelectedIndex());
    JScrollPane scpDendrogram=getHierClusteringPanel(topCluster, ch.getSelectedIndex());
    ClustersTable clTable=new ClustersTable(topCluster.getClustersAtLevel(ch.getSelectedIndex()),
        distanceMatrix,exList, rulesView.getRuleRenderer(),ruleSet.attrMinMax,minA,maxA,minQ,maxQ);
    scpDendrogram.setPreferredSize(new Dimension(100,200));
    JSplitPane splitPane=new JSplitPane(JSplitPane.VERTICAL_SPLIT,clTable.scrollPane,scpDendrogram);
    splitPane.setOneTouchExpandable(true);
    splitPane.setDividerLocation(frame.getHeight()/2);
    frame.getContentPane().add(ch, BorderLayout.NORTH);
    frame.getContentPane().add(splitPane,BorderLayout.CENTER);

    ch.addItemListener(new ItemListener() {
      @Override
      public void itemStateChanged(ItemEvent e) {
        putHierClustersToTable(topCluster, ch.getSelectedIndex());
        ClustersTable clTable=new ClustersTable(topCluster.getClustersAtLevel(ch.getSelectedIndex()),
            distanceMatrix,exList, rulesView.getRuleRenderer(),ruleSet.attrMinMax,minA,maxA,minQ,maxQ);
        splitPane.setTopComponent(clTable.scrollPane);
      }
    });

/*
    String vstr=JOptionPane.showInputDialog(FocusManager.getCurrentManager().getActiveWindow(),
        "Hierarchical clustering done\nHierarchy depth is "+topCluster.hierDepth+"\nSet desired N clusters here (max="+exList.size()+")", "7");
        //"Success!",JOptionPane.INFORMATION_MESSAGE);
    int v=Integer.valueOf(vstr);
    int level;
    for (level=0; level<topCluster.hierDepth && topCluster.getNClustersAtLevel(level)<v; level++);
*/
    Dimension size=Toolkit.getDefaultToolkit().getScreenSize();
    frame.pack();
    splitPane.setDividerLocation(0.5);
    frame.setSize(new Dimension(Math.min(frame.getWidth(),Math.round(0.8f*size.width)),
        Math.min(frame.getHeight(),Math.round(0.8f*size.height))));
    frame.setLocation(size.width-frame.getWidth()-30, size.height-frame.getHeight()-50);
    frame.setVisible(true);
    rulesView.addFrame(frame);
    return frame;
  }

  protected void putHierClustersToTable (ClusterContent topCluster, int level) {
    ClusterContent clusters[]=topCluster.getClustersAtLevel(level);
    if (clusters!=null && clusters.length>1) {
      ClustersAssignments clAss=new ClustersAssignments();
      clAss.objIndexes=new int[ruleSet.rules.size()];
      clAss.clusters=new int[ruleSet.rules.size()];
      for (int i=0; i<ruleSet.rules.size(); i++) {
        clAss.objIndexes[i]=i;
        clAss.clusters[i]=-1;
      }
      clAss.minSize=ruleSet.rules.size()+10;
      for (int i=0; i<clusters.length; i++) {
        int n=0;
        for (int j = 0; j < ruleSet.rules.size(); j++)
          if (clusters[i].member[j]) {
            clAss.clusters[j] = i;
            ++n;
          }
        clAss.minSize=Math.min(n,clAss.minSize);
        clAss.maxSize=Math.max(n,clAss.maxSize);
      }
      ExListTableModel eTblModel= rulesView.getTableModel();
      eTblModel.setCusterAssignments(clAss);
    }
  }
  protected JScrollPane getHierClusteringPanel (ClusterContent topCluster, int level) {
    JPanel pp=topCluster.makePanel();
    if (pp==null) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
              "Failed to visualize the hierarchy!",
              "Error",JOptionPane.ERROR_MESSAGE);
      return null;
    }
    JScrollPane scp=new JScrollPane(pp);
    return scp;
  }

  public void extractSubset(ItemSelectionManager selector,
                              ArrayList<CommonExplanation> exList,
                              double distanceMatrix[][]) {
    ArrayList selected = selector.getSelected();
    if (selected.size() < 1)
      return;
    int result=JOptionPane.showConfirmDialog(FocusManager.getCurrentManager().getActiveWindow(),
        "Answer \"Yes\" to extract the selected rules and \"No\" " +
            "to extract the remaining (not selected) rules.","Selected or not?",
        JOptionPane.YES_NO_CANCEL_OPTION);
    if (result==JOptionPane.CANCEL_OPTION)
      return;
    boolean extractSelected=result==JOptionPane.YES_OPTION;

    int nRulesToExtract=(extractSelected)?selected.size():exList.size()-selected.size();

    ArrayList<CommonExplanation> exSubset = new ArrayList<CommonExplanation>(nRulesToExtract);
    int idx[] = new int[nRulesToExtract];
    int nEx = 0;
    if (extractSelected) {
      for (int i = 0; i < selected.size(); i++)
        if (selected.get(i) instanceof Integer) {
          idx[nEx] = (Integer) selected.get(i);
          exSubset.add(exList.get(idx[nEx]));
          ++nEx;
        }
    }
    else
      for (int i=0; i<exList.size(); i++)
        if (!selected.contains(i)) {
          idx[nEx] =i;
          exSubset.add(exList.get(i));
          ++nEx;
        }

    double distances[][] = null;
    if (distanceMatrix!=null) {
      distances=new double[nEx][nEx];
      for (int i = 0; i < nEx; i++) {
        distances[i][i] = 0;
        int ii = idx[i];
        for (int j = i + 1; j < nEx; j++) {
          int jj = idx[j];
          distances[i][j] = distances[j][i] = distanceMatrix[ii][jj];
        }
      }
    }
    ShowRules showRules=createShowRulesInstance(exSubset);
    showRules.ruleSet.description=showRules.ruleSet.title=
        (extractSelected)?"Subset with "+exSubset.size()+" rules selected from the set of "+
        exList.size()+((exList.equals(origRules))?" original rules":" earlier selected or derived rules"):
        "Subset with "+exSubset.size()+" rules remaining after exclusion of "+selected.size()+
            " selected rules from the set of "+exList.size()+
            ((exList.equals(origRules))?" original rules":" earlier selected or derived rules");
    showRules.showRulesInTable();
  }
  
  /**
   * From the given set of rules or explanations, extracts those rules that are not subsumed
   * by any other rules. Shows the resulting rule set in a table view.
   */
  public void getNonSubsumed(ArrayList<CommonExplanation> exList,
                             Hashtable<String,float[]> attrMinMax) {
    if (exList==null || exList.size()<2)
      return;
    boolean noActions=RuleMaster.noActionDifference(exList);
    double maxQDiff=Double.NaN;
    if (noActions) {
      String value=JOptionPane.showInputDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "The rules do not differ in actions (decisions) and can be compared by closeness of " +
              "the Q values. Enter a threshold for the difference in Q :",
          String.format("%.5f",RuleMaster.suggestMaxQDiff(exList)));
      if (value==null)
        return;
      try {
        maxQDiff=Double.parseDouble(value);
        if (maxQDiff<0) {
          JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
              "Illegal threshold value for the Q difference; must be >=0!",
              "Error",JOptionPane.ERROR_MESSAGE);
          return;
        }
      } catch (Exception ex) {
        JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
            "Illegal threshold value for the Q difference!",
            "Error",JOptionPane.ERROR_MESSAGE);
      }
    }
    System.out.println("Trying to reduce the explanation set by removing less general explanations...");
    ArrayList<CommonExplanation> exList2= RuleMaster.removeLessGeneral(exList,origRules.rules,attrMinMax,
        noActions,maxQDiff);
    if (exList2.size()<exList.size()) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Reduced the number of explanations from " +
                                             exList.size() + " to " + exList2.size(),
          "Reduced rule set",JOptionPane.INFORMATION_MESSAGE);
    }
    else {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Did not manage to reduce the set of explanations!",
          "Fail",JOptionPane.WARNING_MESSAGE);
      return;
    }
    ShowRules showRules=createShowRulesInstance(exList2);
    showRules.ruleSet.setNonSubsumed(true);
    showRules.countRightAndWrongRuleApplications();
    showRules.ruleSet.title=showRules.ruleSet.description=exList2.size()+" non-subsumed rules selected from the set of "+
        exList.size()+((exList.equals(origRules))?" original rules":" earlier selected or derived rules");
    showRules.showRulesInTable();
  }

  public void removeContradictory (ArrayList<CommonExplanation> exList) {
    if (exList==null || exList.size()<2)
      return;
    boolean noActions=RuleMaster.noActionDifference(exList);
    double maxQDiff=Double.NaN;
    if (noActions) {
      String value=JOptionPane.showInputDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "The rules do not differ in actions (decisions) and can be compared by closeness of " +
              "the Q values. Enter a threshold for the difference in Q :",
          String.format("%.5f",RuleMaster.suggestMaxQDiff(exList)));
      if (value==null)
        return;
      try {
        maxQDiff=Double.parseDouble(value);
        if (maxQDiff<0) {
          JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
              "Illegal threshold value for the Q difference; must be >=0!",
              "Error",JOptionPane.ERROR_MESSAGE);
          return;
        }
      } catch (Exception ex) {
        JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
            "Illegal threshold value for the Q difference!",
            "Error",JOptionPane.ERROR_MESSAGE);
      }
    }
    System.out.println("Trying to find and remove contradictory rules...");
    ArrayList<CommonExplanation> exList2= RuleMaster.removeContradictory(exList,noActions,maxQDiff);
    if (exList2.size()<exList.size()) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "The removal of contradictory rules has reduced the number of explanations from " +
              exList.size() + " to " + exList2.size(),
          "Removed contradictions!",JOptionPane.INFORMATION_MESSAGE);
    }
    else {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Did not find any contradictory rules!",
          "No contradictions!",JOptionPane.INFORMATION_MESSAGE);
      return;
    }
    ShowRules showRules=createShowRulesInstance(exList2);
    showRules.countRightAndWrongRuleApplications();
    showRules.ruleSet.description=showRules.ruleSet.title=exList2.size()+" from the "+
        exList.size()+((exList.equals(origRules))?" original rules":" earlier selected or derived rules")+
        " remaining after removal of "+(exList.size()-exList2.size())+" contradictory rules.";
    showRules.showRulesInTable();

    ArrayList<UnitedRule> excluded=new ArrayList<UnitedRule>(exList.size()-exList2.size());
    for (CommonExplanation ce:exList)
      if (!exList2.contains(ce)) {
        UnitedRule r=UnitedRule.getRule(ce);
        if (noActions)
          r.countRightAndWrongCoveragesByQ(exList2);
        else
          r.countRightAndWrongCoverages(exList2);
        excluded.add(r);
      }

    showRules=createShowRulesInstance(excluded);
    showRules.countRightAndWrongRuleApplications();
    showRules.ruleSet.description=showRules.ruleSet.title=excluded.size()+
        " contradictory rules extracted and removed from the set of "+
        exList.size()+((exList.equals(origRules))?" original rules":" earlier selected or derived rules");
    showRules.showRulesInTable();
  }

  private long aggrT0=-1;
  
  public void stateChanged(ChangeEvent e) {
    if (e.getSource() instanceof ClustersAssignments) {
      if (orderingInProgress) {
        orderingInProgress=false;
        clOptics.removeChangeListener(this);
      }
    }
    else
    if (e.getSource() instanceof AggregationRunner) {
      AggregationRunner aRun=(AggregationRunner)e.getSource();
      ArrayList<UnitedRule> aggRules=aRun.aggRules;
      if (aggRules==null || aggRules.isEmpty() || aggRules.size()>=ruleSet.rules.size()) {
        JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
            "Failed to aggregate!",
            "Failed",JOptionPane.WARNING_MESSAGE);
        return;
      }
      
      if (aRun.finished) {
        if (aggRules.size()>=ruleSet.rules.size()) {
          JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
              "The given rules could not be aggregated!",
              "Failed",JOptionPane.WARNING_MESSAGE);
          return;
        }
      }

      long tDiff=System.currentTimeMillis()-aggrT0;
      aggrT0=-1;
      System.out.println("Aggregation took "+(1.0*tDiff/60000)+" minutes.");
  
      ArrayList<CommonExplanation> aggEx=new ArrayList<CommonExplanation>(aggRules.size());
      aggEx.addAll(aggRules);
      ShowRules showRules=createShowRulesInstance(aggEx);
      showRules.ruleSet.setNonSubsumed(true);
      showRules.ruleSet.setAggregated(true);
      showRules.ruleSet.setAccThreshold(aRun.currAccuracy);
      if (aRun.aggregateByQ)
        showRules.ruleSet.setMaxQDiff(aRun.maxQDiff);
      showRules.countRightAndWrongRuleApplications();
      showRules.ruleSet.title=showRules.ruleSet.description=
          aggEx.size()+" aggregated and generalized rules obtained from the set of "+
              ruleSet.rules.size()+((ruleSet.equals(origRules))?" original rules":" earlier selected or derived rules")+
              " using the following parameter settings: min coherence = "+String.format("%.3f",aRun.minAccuracy)+
              ((aRun.aggregateByQ)?String.format(", max value difference = %.5f",aRun.maxQDiff):"")+
              (!Double.isNaN(aRun.accStep)?String.format("; stepwise aggregation starting from %.3f with step %.3f",
                  aRun.initAccuracy,aRun.accStep):"");
      showRules.showRulesInTable();
    }
    else
    if (e.getSource().equals(ruleSelector))  {
      if (!ruleSelector.isRunning)
        ruleSelector=null;
      else {
        ArrayList<CommonExplanation> selRules=ruleSelector.getSelectedRules();
        if (selRules!=null)
          if (ruleSelector.mustExtract()) {
            ShowRules showRules=createShowRulesInstance(selRules);
            showRules.ruleSet.setTitle("Subset by query "+ruleSelector.queryStr);
            showRules.ruleSet.description="Subset with "+selRules.size()+" rules selected from the set of "+
                ruleSet.rules.size()+((ruleSet.rules.equals(origRules))?" original rules":" earlier selected or derived rules")+
                " by the following query: "+ruleSelector.queryStr;
            showRules.showRulesInTable();
          }
          else {
            //select the rules by highlighting
            if (localSelector==null)
              return;
            ArrayList<Integer> selIndexes=new ArrayList<Integer>(selRules.size());
            for (int i=0; i<ruleSet.rules.size(); i++)
              if (selRules.contains(ruleSet.rules.get(i)))
                selIndexes.add(i);
            localSelector.select(selIndexes);
          }
      }
    }
    else
    if (e.getSource() instanceof EnsembleExplorer)  {
      EnsembleExplorer eEx=(EnsembleExplorer)e.getSource();
      ArrayList<CommonExplanation> selRules=eEx.getSelectedRules();
      if (selRules!=null) {
        ShowRules showRules=createShowRulesInstance(selRules);
        showRules.ruleSet.setTitle(eEx.getSelectionInfoShort());
        showRules.ruleSet.description=eEx.getSelectedRulesInfo();
        showRules.showRulesInTable();
      }
    }
  }
  
  public void aggregate(ArrayList<CommonExplanation> exList,
                        Hashtable<String,float[]> attrMinMax) {
    if (exList==null || exList.size()<2)
      return;
  
    boolean noActions=RuleMaster.noActionDifference(exList);
    
    JPanel pDialog=new JPanel();
    pDialog.setLayout(new BoxLayout(pDialog,BoxLayout.Y_AXIS));
    pDialog.add(new JLabel("Set parameters for rule aggregation and generalisation:",JLabel.CENTER));
    JPanel pp=new JPanel(new BorderLayout(10,0));
    pp.add(new JLabel("Coherence threshold from 0 to 1 :",JLabel.RIGHT),BorderLayout.CENTER);
    
    JTextField tfAcc=new JTextField(String.format("%.3f", (float)Math.max(0.6f,ruleSet.accThreshold-0.25f)),5);
    pp.add(tfAcc,BorderLayout.EAST);
    pDialog.add(pp);
    JCheckBox cbData=(dataInstances==null)?null:
                         new JCheckBox("Additionally check the data-based rule fidelity",true);
    if (cbData!=null)
      pDialog.add(cbData);
    
    JTextField tfQDiff=null;
    if (noActions) {
      pDialog.add(Box.createVerticalStrut(5)); // a spacer
      pDialog.add(new JLabel("The rules do not differ in resulting " +
                           "class labels / actions / decisions.",JLabel.CENTER));
      pDialog.add(new JLabel("They can be joined based on close " +
                           "real-valued results (denoted as Q).",JLabel.CENTER));
      pp=new JPanel(new BorderLayout(10,0));
      pp.add(new JLabel("Enter a threshold for the difference in Q :",JLabel.RIGHT),BorderLayout.CENTER);
      tfQDiff=new JTextField(String.format("%.4f",RuleMaster.suggestMaxQDiff(exList)),7);
      pp.add(tfQDiff,BorderLayout.EAST);
      pDialog.add(pp);
    }
    pDialog.add(Box.createVerticalStrut(5)); // a spacer
    /**/
    pDialog.add(new JLabel("The aggregation can be fulfilled in a step-wise manner:",JLabel.CENTER));
    pDialog.add(new JLabel("The process is executed several times", JLabel.CENTER));
    pDialog.add(new JLabel("with decreasing the coherence threshold in each step.", JLabel.CENTER));
    /**/
    pDialog.add(Box.createVerticalStrut(5)); // a spacer
    JCheckBox cbIterative=new JCheckBox("Do step-wise aggregation",false);
    pDialog.add(cbIterative);
    JTextField tfAcc0=new JTextField("1.00",5);
    pp=new JPanel(new BorderLayout(10,0));
    pp.add(new JLabel("Initial c threshold:",JLabel.RIGHT),BorderLayout.CENTER);
    pp.add(tfAcc0,BorderLayout.EAST);
    pDialog.add(pp);
    JTextField tfAccStep=new JTextField("0.05",5);
    pp=new JPanel(new BorderLayout(10,0));
    pp.add(new JLabel("Decrease the threshold in each run by",JLabel.RIGHT),BorderLayout.CENTER);
    pp.add(tfAccStep,BorderLayout.EAST);
    pDialog.add(pp);
  
    int result = JOptionPane.showConfirmDialog(FocusManager.getCurrentManager().getActiveWindow(), pDialog,
        "Rule aggregation parameters", JOptionPane.OK_CANCEL_OPTION);
    if (result != JOptionPane.OK_OPTION)
      return;
    
    double minAccuracy=0;
    double maxQDiff=Double.NaN;
    
    String value=tfAcc.getText();
    if (value == null)
      return;
    try {
      minAccuracy = Double.parseDouble(value);
      if (minAccuracy < 0 || minAccuracy > 1) {
        JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
            "Illegal threshold value for the coherence; must be from 0 to 1!",
            "Error", JOptionPane.ERROR_MESSAGE);
        return;
      }
    } catch (Exception ex) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Illegal threshold value for the coherence!",
          "Error", JOptionPane.ERROR_MESSAGE);
      return;
    }
    
    if (noActions) {
      value=tfQDiff.getText();
      if (value==null)
        return;
      try {
        maxQDiff=Double.parseDouble(value);
        if (maxQDiff<0) {
          JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
              "Illegal threshold value for the Q difference; must be >=0!",
              "Error",JOptionPane.ERROR_MESSAGE);
          return;
        }
      } catch (Exception ex) {
        JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
            "Illegal threshold value for the Q difference!",
            "Error",JOptionPane.ERROR_MESSAGE);
      }
    }
    
    boolean checkWithData=cbData!=null && cbData.isSelected();

    Window win=FocusManager.getCurrentManager().getActiveWindow();
    System.out.println("Trying to reduce the explanation set by removing less general explanations...");
    JDialog dialog=new JDialog(win, "Trying to remove less general rules ...",
        Dialog.ModalityType.MODELESS);
    dialog.setLayout(new BorderLayout());
    JPanel p = new JPanel(new FlowLayout(FlowLayout.CENTER, 50, 20));
    p.add(new JLabel("Trying to reduce the explanation set " +
        "by removing less general explanations...", JLabel.CENTER));
    dialog.getContentPane().add(p, BorderLayout.CENTER);
    dialog.pack();
    dialog.setLocationRelativeTo(win);
    dialog.setDefaultCloseOperation(JDialog.DO_NOTHING_ON_CLOSE);
    dialog.setVisible(true);

    ArrayList<CommonExplanation> exList2= RuleMaster.removeLessGeneral(exList,origRules.rules,attrMinMax,
      noActions,maxQDiff);
    dialog.dispose();

    if (exList2.size()<exList.size()) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "After excluding the subsumed (less general) rules, the number of the rules has decreased from " +
              exList.size() + " to " + exList2.size(),
          "Reduced rule set",JOptionPane.INFORMATION_MESSAGE);
      exList=exList2;
    }

    System.out.println("Trying to aggregate the rules...");

    ArrayList<UnitedRule> rules=UnitedRule.getRules(exList);
    ArrayList<UnitedRule> aggRules=null;
    AbstractList<Explanation> data=(checkWithData)?dataInstances:null;
    
    AggregationRunner aggRunner=new AggregationRunner(rules,origRules.rules,attrMinMax,data);
    aggRunner.setOwner(this);
    aggRunner.setAggregateByQ(noActions);
    aggRunner.setMinAccuracy(minAccuracy);
    aggRunner.setMaxQDiff(maxQDiff);
    
    if (cbIterative.isSelected()) {
      double initAccuracy=Double.NaN;
      value=tfAcc0.getText();
      if (value!=null)
        try {
          initAccuracy=Float.parseFloat(value);
        } catch (Exception ex) {}
      if (Double.isNaN(initAccuracy) || initAccuracy<0.5 || initAccuracy>1) {
        result = JOptionPane.showConfirmDialog (FocusManager.getCurrentManager().getActiveWindow(),
            "Illegal value for the initial coherence threshold! Should it be set to 1?",
            "Illegal initial threshold value!",JOptionPane.YES_NO_OPTION);
        if (result != JOptionPane.YES_OPTION)
          return;
        initAccuracy=1;
      }
      double accStep=Double.NaN;
      value=tfAccStep.getText();
      if (value!=null)
        try {
          accStep=Double.parseDouble(value);
        } catch (Exception ex) {}
      if (Double.isNaN(accStep) || accStep<=0 || initAccuracy-accStep<=0) {
        result = JOptionPane.showConfirmDialog (FocusManager.getCurrentManager().getActiveWindow(),
            "Illegal value for the threshold decrement in each step! Should it be set to 0.1?",
            "Illegal value for threshold decrement!",JOptionPane.YES_NO_OPTION);
        if (result != JOptionPane.YES_OPTION)
          return;
        accStep=0.1;
      }
      aggRunner.setIterationParameters(initAccuracy,accStep);
    }

    aggrT0=System.currentTimeMillis();

    aggRunner.aggregate(FocusManager.getCurrentManager().getActiveWindow());
    
  }
  
  public void showAggregationInProjection(ItemSelectionManager selector) {
    if (ruleSet==null || !ruleSet.hasRules() || !(ruleSet.rules.get(0) instanceof UnitedRule))
      return;
    boolean applyToSelection=
        selector.hasSelection() &&
            JOptionPane.showConfirmDialog(FocusManager.getCurrentManager().getActiveWindow(),
                "Apply the operation to the selected subset?",
                "Apply to selection?",JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE)
                ==JOptionPane.YES_OPTION;
    ArrayList rules=(applyToSelection)?getSelectedRules(ruleSet.rules,selector):ruleSet.rules;
    ArrayList<CommonExplanation> origList=new ArrayList<CommonExplanation>(rules.size());
    HashSet<ArrayList<Vertex>> graphs=null;
    ArrayList<Integer> unionIds=new ArrayList<Integer>(rules.size()*5);
    for (int i=0; i<rules.size(); i++) {
      UnitedRule rule=(UnitedRule)rules.get(i);
      if (rule.fromRules==null || rule.fromRules.isEmpty()) {
        origList.add(rule);
        unionIds.add(i);
        continue;
      }
      //create a graph
      ArrayList<UnitedRule> ruleGroup=addOrigRules(rule,null);
      if (ruleGroup.size()>1) { //create a graph
        ArrayList<Vertex> graph=new ArrayList<Vertex>(ruleGroup.size());
        for (int j=0; j<ruleGroup.size(); j++)
          graph.add(new Vertex(Integer.toString(j+origList.size())));
        for (int j1=0; j1<ruleGroup.size()-1; j1++) {
          Vertex v=graph.get(j1);
          for (int j2 = j1 + 1; j2 < ruleGroup.size(); j2++) {
            double d = UnitedRule.distance(ruleGroup.get(j1), ruleGroup.get(j2), ruleSet.attrMinMax);
            if (Double.isNaN(d)) {
              System.out.println("!!! NaN distance value !!!");
              System.out.println(ruleGroup.get(j1).toString());
              System.out.println(ruleGroup.get(j2).toString());
            }
            else {
              Edge e=new Edge((float) d);
              Vertex v2=graph.get(j2);
              v.addEdge(v2, e);
              v2.addEdge(v, e);
            }
          }
        }
        Prim prim=new Prim(graph);
        prim.run();
        if (graphs==null)
          graphs=new HashSet<ArrayList<Vertex>>(rules.size());
        graphs.add(graph);
      }
      for (int j=0; j<ruleGroup.size(); j++) {
        origList.add(ruleGroup.get(j));
        unionIds.add(i);
      }
    }
    
    TSNE_Runner tsne=new TSNE_Runner();
    tsne.setFileRegister(rulesViewerManager.createdFiles);
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
    } catch (Exception ex) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Illegal perplexity!"+value,
          "Error",JOptionPane.ERROR_MESSAGE);
      return;
    }
    
    double d[][]=CommonExplanation.computeDistances(origList,ruleSet.featuresInDistances,ruleSet.attrMinMax);
  
    ExplanationsProjPlot2D pp=new ExplanationsProjPlot2D(ruleSet.attrMinMax,
        rulesView.getAttrs(),rulesView.getMinMax());
    pp.setExplanations(origList,origRules.rules);
    pp.setDistanceMatrix(d);
    pp.setFeatures(ruleSet.getSelectedFeatures());
    pp.setGraphs(graphs);
    int uIds[]=new int[unionIds.size()];
    for (int j=0; j<unionIds.size(); j++)
      uIds[j]=unionIds.get(j);
    pp.setUnionIds(uIds);
    pp.setProjectionProvider(tsne);
    pp.setPreferredSize(new Dimension(800,800));
  
    Translator translator=(origRules!=null && (origHighlighter!=null || origSelector!=null))?
                              createTranslator(origList,origRules.rules):null;
    if (translator!=null) {
      if (origHighlighter!=null) {
        HighlightTranslator hTrans=new HighlightTranslator();
        hTrans.setTranslator(translator);
        hTrans.setOtherHighlighter(origHighlighter);
        pp.setHighlighter(hTrans);
      }
      if (origSelector!=null) {
        SelectTranslator sTrans=new SelectTranslator();
        sTrans.setTranslator(translator);
        sTrans.setOtherSelector(origSelector);
        pp.setSelector(sTrans);
      }
    }
  
    Dimension size=Toolkit.getDefaultToolkit().getScreenSize();
    JFrame plotFrame=new JFrame(pp.getProjectionProvider().getProjectionTitle());
    plotFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    plotFrame.getContentPane().add(pp);
    plotFrame.pack();
    plotFrame.setLocation(size.width-plotFrame.getWidth()-30, size.height-plotFrame.getHeight()-50);
    plotFrame.setVisible(true);
    rulesView.addFrame(plotFrame);

    JPopupMenu menu=new JPopupMenu();
    
    JMenuItem mitSelectLinked=new JMenuItem("Select linked items");
    menu.add(mitSelectLinked);
    mitSelectLinked.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        pp.selectLinkedToSelected();
      }
    });
  
    JMenuItem mitSelectUnions=new JMenuItem("Select the union(s) of the linked items");
    menu.add(mitSelectUnions);
    mitSelectUnions.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        ArrayList<Integer> uIds=pp.getUnionIdsOfSelected();
        if (uIds!=null)
          selector.select(uIds);
      }
    });
    
    JMenuItem mitExtract=new JMenuItem("Extract the selected subset to a separate view");
    menu.add(mitExtract);
    mitExtract.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        extractSubset(pp.getSelector(),origList,d);
      }
    });
  
    JMenuItem mit=new JMenuItem("Re-run t-SNE with another perplexity setting");
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
  
    pp.addMouseListener(new MouseAdapter() {
      @Override
      public void mousePressed(MouseEvent e) {
        super.mousePressed(e);
        if (e.getButton()>MouseEvent.BUTTON1) {
          ArrayList selected=pp.getSelector().getSelected();
          boolean someSelected=selected!=null && selected.size()>0;
          mitExtract.setEnabled(someSelected);
          mitSelectLinked.setEnabled(someSelected);
          mitSelectUnions.setEnabled(someSelected);
          menu.show(pp,e.getX(),e.getY());
        }
      }
    });
  }
  
  protected static ArrayList<UnitedRule> addOrigRules(UnitedRule rule, ArrayList<UnitedRule> origList) {
    if (rule==null)
      return origList;
    if (origList==null)
      origList=new ArrayList<UnitedRule>(20);
    if (rule.fromRules==null || rule.fromRules.isEmpty())
      origList.add(rule);
    else
      for (UnitedRule r:rule.fromRules)
        addOrigRules(r,origList);
    return origList;
  }
  
  /**
   * For two lists consisting (at least partly) of same objects finds correspondences between
   * the indexes of the common objects and creates a translator that will keep the correspondences.
   */
  public static Translator createTranslator(ArrayList list1, ArrayList list2) {
    if (list1==null || list2==null || list1.isEmpty() || list2.isEmpty())
      return null;
    ArrayList<Object[]> pairs=new ArrayList<Object[]>(Math.min(list1.size(),list2.size()));
    for (int i=0; i<list1.size(); i++) {
      int idx=list2.indexOf(list1.get(i));
      if (idx>=0) {
        Object pair[]={new Integer(i),new Integer(idx)};
        pairs.add(pair);
      }
    }
    if (pairs.isEmpty())
      return null;
    Translator trans=new Translator();
    trans.setPairs(pairs);
    return trans;
  }

  public void openTreeExplorer() {
    if (ruleSet.rules==null || !ruleSet.hasRules() || !RuleMaster.hasDistinctTreeIds(ruleSet.rules))
      return;
    EnsembleExplorer eEx=new EnsembleExplorer();
    eEx.setOwner(this);
    eEx.setFileRegister(rulesViewerManager.createdFiles);
    JPanel eExPanel=eEx.startEnsembleExplorer(ruleSet.rules, rulesView.getGeneralInfo(),
        ruleSet.attrMinMax,ruleSet.featuresInDistances);
    if (eExPanel==null)
      return;
    Dimension size=Toolkit.getDefaultToolkit().getScreenSize();
    JFrame plotFrame=new JFrame("Distances between trees");
    plotFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    plotFrame.getContentPane().add(eExPanel);
    plotFrame.setSize(500,500);
    plotFrame.setLocation(size.width-plotFrame.getWidth()-30, 100);
    plotFrame.setVisible(true);
    rulesView.addFrame(plotFrame);
  }

  public DataSet loadData() {
    DataSet data=null;
    try {
      String filePath = CSVDataLoader.selectFilePathThroughDialog(true);
      if (filePath == null)
        return null;
      data = CSVDataLoader.loadDataFromCSVFile(filePath);
    } catch (IOException ex) { }
    if (data == null) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Failed to load data!", "Data loading failed!",
          JOptionPane.ERROR_MESSAGE);
    }
    JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
        "Loaded " + data.records.size() + " data records.", "Data loaded!",
        JOptionPane.INFORMATION_MESSAGE);
    data.description="Set with "+data.records.size()+" original data records loaded from file "+
        data.filePath;
    rulesViewerManager.addLoadedDataSet(data);

    DataTableViewer dViewer=new DataTableViewer(data,
        ruleSet.listOfFeatures.toArray(new String[ruleSet.listOfFeatures.size()]),this);
    DataVersionsViewer dViewFrame=new DataVersionsViewer(dViewer,null,null);
    dViewFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    rulesViewerManager.addSharedFrame(dViewFrame);
    return data;
  }

  public void applyRulesToData() {
    boolean applyToSelection=localSelector!=null &&
        localSelector.hasSelection() &&
            JOptionPane.showConfirmDialog(FocusManager.getCurrentManager().getActiveWindow(),
                "Use only the selected subset of the rules?",
                "Use selected rules?",JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE)
                ==JOptionPane.YES_OPTION;
    ArrayList rules=(applyToSelection)?getSelectedRules(ruleSet.rules,localSelector):ruleSet.rules;
    DataSet testData;
    ArrayList<DataSet> dataSets=rulesViewerManager.getDataSetsList();
    if (dataSets==null || dataSets.isEmpty())
      testData=rulesViewerManager.currentData=loadData();
    else
    if (dataSets.size()==1)
      testData=dataSets.get(0);
    else {
      //allow the user to select the data version (the versions differ in the classes treated as original)
      String options[]=new String[dataSets.size()];
      for (int i=0; i<dataSets.size(); i++)
        options[i]=dataSets.get(i).versionLabel;
      JList list=new JList(options);
      list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
      list.setSelectedIndex(0);
      JScrollPane scrollPane = new JScrollPane(list);
      JPanel p=new JPanel();
      p.setLayout(new BorderLayout());
      p.add(scrollPane,BorderLayout.CENTER);
      p.add(new JLabel("Select one of the data versions with true or predicted classes or values"),
          BorderLayout.NORTH);
      // Show the JOptionPane with the list inside
      int result = JOptionPane.showConfirmDialog(
          rulesViewerManager.mainFrame,
          scrollPane,
          "Select dataset version",
          JOptionPane.OK_CANCEL_OPTION,
          JOptionPane.PLAIN_MESSAGE
      );
      if (result!=JOptionPane.OK_OPTION)
        return;
      testData=dataSets.get(list.getSelectedIndex());
    }
    if (testData==null)
      return;

    String[] options = { "Test rules", "Obtain predictions" };

    boolean testRules=0==JOptionPane.showOptionDialog(rulesViewerManager.mainFrame,
        "Is your goal to test the rules on the data or to obtain predictions?",
        "Intended operation",0,JOptionPane.QUESTION_MESSAGE,null,options,options[0]);

    boolean ok=false;
    if (testRules)
      ok=RuleMaster.testRulesOnData(rules,testData);
    else {
      testData = testData.makeNewVersion();
      ok=RuleMaster.applyRulesToData(rules, testData.records);
    }

    if (!ok) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Failed to apply the rules to data!",
          "Rule application failed", JOptionPane.ERROR_MESSAGE);
      return;
    }

    JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
        "Completed application of " + rules.size() + " rules to " +
            testData.records.size() + " data records.", "Rule application done",
        JOptionPane.INFORMATION_MESSAGE);

    if (testRules) {
      rulesView.updateDataInTable();
      getMainFrame().toFront();
      getMainFrame().requestFocus();
    }
    else {
      testData.description = "Data records with predictions obtained by applying " + rules.size() + " rules " +
          ((applyToSelection) ? "selected from rule set " : "") + "described as " + rulesView.getGeneralInfo();

      DataTableViewer dViewer = new DataTableViewer(testData,
          ruleSet.listOfFeatures.toArray(new String[ruleSet.listOfFeatures.size()]), this);

      ClassConfusionMatrix cMatrix = new ClassConfusionMatrix();
      if (!cMatrix.makeConfusionMatrix(testData))
        cMatrix = null;

      DataSet origData = testData.getOriginalVersion();

      DataVersionsViewer dViewFrame = rulesViewerManager.findDataViewFrame(origData);
      if (dViewFrame != null)
        dViewFrame.addDataViewer(dViewer, cMatrix, rulesView.getGeneralInfo());
      else {
        dViewFrame = new DataVersionsViewer(dViewer, cMatrix, rulesView.getGeneralInfo());
        dViewFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        rulesViewerManager.addSharedFrame(dViewFrame);
      }

      if (!applyToSelection)
        rulesViewerManager.currentData = testData;
    }
  }

  public void exportDataToCSVFile() {
    if (rulesViewerManager.currentData==null ||
        rulesViewerManager.currentData.records==null ||
        rulesViewerManager.currentData.records.isEmpty()) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "No data have been loaded in the system!","No data",
          JOptionPane.INFORMATION_MESSAGE);
      return;
    }
    String pathName=CSVDataLoader.selectFilePathThroughDialog(false);
    if (pathName==null)
      return;
    if (rulesViewerManager.currentData.exportToCSV(pathName))
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Successfully exported the data to file "+pathName,"Data exported",
          JOptionPane.INFORMATION_MESSAGE);
    else
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Failed to export the data!","Export failed",
          JOptionPane.ERROR_MESSAGE);
  }

  public void showFeatureIntervalsDistributions() {
    boolean applyToSelection=localSelector!=null &&
        localSelector.hasSelection() &&
        JOptionPane.showConfirmDialog(FocusManager.getCurrentManager().getActiveWindow(),
            "Use only the selected subset of the rules?",
            "Use selected rules?",JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE)
            ==JOptionPane.YES_OPTION;
    ArrayList rules=(applyToSelection)?getSelectedRules(ruleSet.rules,localSelector):ruleSet.rules;

    int nFeatureIntervals=(ruleSet.isFeaturesAreBinary())?2:10;
    if (!ruleSet.isFeaturesAreBinary()) {
      String input = JOptionPane.showInputDialog(rulesViewerManager.mainFrame,
          "Enter the number of intervals for dividing the feature value ranges:", nFeatureIntervals);
      try {
        int intervals = Integer.parseInt(input);
        if (intervals <= 0) {
          throw new NumberFormatException();
        }
        nFeatureIntervals = intervals;
      } catch (NumberFormatException ex) {
        JOptionPane.showMessageDialog(rulesViewerManager.mainFrame,
            "Invalid number of intervals. Please enter a positive integer.",
            "Error", JOptionPane.ERROR_MESSAGE);
      }
    }
    ArrayList<Double> resultBreaks=null;
    if (ruleSet.isRegression()) {
      IntervalBreaksDialog dia=new IntervalBreaksDialog(rulesViewerManager.mainFrame,
          ruleSet.minQValue,ruleSet.maxQValue);
      dia.setVisible(true);
      resultBreaks=dia.getBreaks();
      if (resultBreaks==null)
        return;
    }

    ValuesFrequencies freq[][]=ruleSet.getFeatureValuesDistributions(rules,nFeatureIntervals,resultBreaks);
    if (freq==null || freq.length<1) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Failed to count the frequencies of the feature values!","No frequencies obtained!",
          JOptionPane.ERROR_MESSAGE);
      return;
    }
  
    FeatureHeatmapsManager fMan=new FeatureHeatmapsManager();
    String rulesInfo=((applyToSelection)?rules.size()+" rules selected from ":"")+rulesView.getGeneralInfo();
    JComponent vis=fMan.createFeatureDistributionsDisplay(freq,rulesInfo);
    if (vis==null) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "Failed to generate a display of the distribution of the feature values intervals!",
          "No display generated!",
          JOptionPane.ERROR_MESSAGE);
      return;
    }
    vis.setName("Feature values distribution view");
    JPanel distributionVisPanel=new JPanel();
    distributionVisPanel.setLayout(new BorderLayout());
    distributionVisPanel.add(vis,BorderLayout.CENTER);

    RuleFilterUI filterUI=new RuleFilterUI(ruleSet);

    JSplitPane splitPane=new JSplitPane(JSplitPane.HORIZONTAL_SPLIT,distributionVisPanel,filterUI);
    splitPane.setResizeWeight(0.9); // Optional: balance initial position

    Dimension size=Toolkit.getDefaultToolkit().getScreenSize();
    JFrame valuesDistributionsFrame=new JFrame("Filtering rules by feature values intervals");
    valuesDistributionsFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    valuesDistributionsFrame.getContentPane().add(splitPane);
    valuesDistributionsFrame.setSize(Math.round(0.8f*size.width),Math.round(0.7f*size.height));
    valuesDistributionsFrame.setLocation(size.width-valuesDistributionsFrame.getWidth()-30,
            size.height-valuesDistributionsFrame.getHeight()-50);
    valuesDistributionsFrame.setVisible(true);
    rulesView.addFrame(valuesDistributionsFrame);

    JFrame noFilterDistributionsFrame=new JFrame("Original feature values distributions");
    noFilterDistributionsFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    noFilterDistributionsFrame.setSize(Math.round(0.8f*size.width),Math.round(0.7f*size.height));
    noFilterDistributionsFrame.setLocation(30, size.height-valuesDistributionsFrame.getHeight()-50);
    rulesView.addFrame(noFilterDistributionsFrame);
  
    FeatureHeatmapsManager filterResultsViewManager=new FeatureHeatmapsManager();
    JLabel filterStateMessage=new JLabel(String.format("%100s", ""),JLabel.CENTER);
    JDialog filterStateDialog=createFilterStateDialog(noFilterDistributionsFrame,filterStateMessage);
    
    filterUI.getApplyFilterButton().addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        applyFilters(filterUI, rulesInfo, freq, filterResultsViewManager,
                noFilterDistributionsFrame, distributionVisPanel, filterStateDialog, filterStateMessage);
      }
    });
    filterUI.getExtractRulesButton().addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        extractOrHighlightFilteredRules(filterUI,true);
      }
    });
    filterUI.getHighlightRulesButton().addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        extractOrHighlightFilteredRules(filterUI,false);
      }
    });
    filterUI.addChangeListener(new ChangeListener() {
      @Override
      public void stateChanged(ChangeEvent e) {
        applyFilters(filterUI,rulesInfo,freq,filterResultsViewManager, noFilterDistributionsFrame,
                distributionVisPanel,filterStateDialog, filterStateMessage);
      }
    });
  }

  private JDialog createFilterStateDialog(Frame win, JLabel filterStateMessage) {
    JDialog filterStateDialog = new JDialog(win, "Status of rule filtering", Dialog.ModalityType.MODELESS);
    filterStateDialog.setLayout(new BorderLayout());
    JPanel p=new JPanel(new FlowLayout(FlowLayout.CENTER,10,20));
    p.add(filterStateMessage);
    JPanel pp=new JPanel(new GridLayout(0,1));
    pp.add(p);
    pp.add(new JLabel(""));
    filterStateDialog.getContentPane().add(pp, BorderLayout.CENTER);
    filterStateDialog.pack();
    filterStateDialog.setLocationRelativeTo(win);
    filterStateDialog.setDefaultCloseOperation(JDialog.DO_NOTHING_ON_CLOSE);
    //filterStateDialog.setVisible(true);
    return filterStateDialog;
  }

  public void extractOrHighlightFilteredRules(RuleFilterUI filterUI, boolean toExtract) {
    if (filterUI==null)
      return;
    if (!toExtract && localSelector==null)
      return;
    Map<String, Object> filters=filterUI.getFilters();
    if (filters==null) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "No filter conditions defined!", "No folter!",
          JOptionPane.INFORMATION_MESSAGE);
      return;
    }
    ArrayList<CommonExplanation> selectedRules=ruleSet.selectRulesByConditionFilters(filters,
        filterUI.mustRangesBeInsideLimits());
    if (selectedRules==null) {
      JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
          "No rules satisfying the filter conditions found!", "Empty result!",
          JOptionPane.INFORMATION_MESSAGE);
      return;
    }
    ArrayList<Integer> classIdxs=new ArrayList<Integer>(10);
    for (CommonExplanation r:selectedRules)
      if (r.action>=0 && !classIdxs.contains(r.action))
        classIdxs.add(r.action);
    if (classIdxs.size()>1) {
      Collections.sort(classIdxs);
      JCheckBox cb[]=new JCheckBox[classIdxs.size()];
      JPanel mainP=new JPanel();
      mainP.setLayout(new GridLayout(0,1));
      mainP.add(new JLabel("Predicted classes:"));
      for (int i=0; i<classIdxs.size(); i++) {
        cb[i] = new JCheckBox("class " + classIdxs.get(i), true);
        mainP.add(cb[i]);
      }
      mainP.add(new JLabel("Deselect to exclude the rules"));
      int result = JOptionPane.showConfirmDialog(
          FocusManager.getCurrentManager().getActiveWindow(),
          mainP,
          "Select classes",
          JOptionPane.OK_CANCEL_OPTION,
          JOptionPane.PLAIN_MESSAGE
      );
      if (result != JOptionPane.OK_OPTION)
        return;
      int sizeOrig=classIdxs.size();
      for (int i=cb.length-1; i>=0; i--)
        if (!cb[i].isSelected())
          classIdxs.remove(i);
      if (classIdxs.isEmpty())
        return;
      if (classIdxs.size()<sizeOrig)
        for (int i=selectedRules.size()-1; i>=0; i--)
          if (!classIdxs.contains(selectedRules.get(i).action))
            selectedRules.remove(i);
    }
    else
      classIdxs=null;

    if (toExtract) {
      ShowRules showRules = createShowRulesInstance(selectedRules);
      String info = selectedRules.size() + " rules satisfying filter conditions";
      ArrayList filterTexts = filterUI.describeFilters();
      if (filterTexts != null)
        for (int i = 0; i < filterTexts.size(); i++)
          info += " " + (i + 1) + ") " + filterTexts.get(i);
      showRules.ruleSet.setTitle(info);
      showRules.ruleSet.description = info + " selected from the set of " +
          ruleSet.rules.size() + ((ruleSet.rules.equals(origRules)) ? " original rules" : " earlier selected or derived rules");
      showRules.showRulesInTable();
    }
    else {
      ArrayList<Integer> selIndexes=new ArrayList<Integer>(selectedRules.size());
      for (int i=0; i<ruleSet.rules.size(); i++)
        if (selectedRules.contains(ruleSet.rules.get(i)))
          selIndexes.add(i);
      localSelector.select(selIndexes);
    }
  }

  public void applyFilters(RuleFilterUI filterUI,
                           String rulesInfo,
                           ValuesFrequencies freqOrig[][],
                           FeatureHeatmapsManager filterResultsViewManager,
                           JFrame noFilterDistributionsFrame,
                           JPanel distributionVisPanel,
                           JDialog filterStateDialog,
                           JLabel filterStateMessage) {
    if (filterUI==null)
      return;
    Map<String, Object> filters=filterUI.getFilters();
    if (filters==null)
      if (filterResultsViewManager.viewCreated())
        filterResultsViewManager.updateData(freqOrig, rulesInfo+" --- NO FILTER");
      else {
        if (filterStateMessage!=null) {
          filterStateMessage.setText("No filter conditions specified!");
          filterStateMessage.revalidate();
        }
      }
    else {
      ArrayList<CommonExplanation> selectedRules=ruleSet.selectRulesByConditionFilters(filters,
          filterUI.mustRangesBeInsideLimits());
      if (selectedRules==null) {
        if (filterStateMessage == null)
          JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
                  "No rules satisfying the filter conditions found!", "Empty result!",
                  JOptionPane.INFORMATION_MESSAGE);
        else {
          filterStateMessage.setText("No rules satisfying the filter conditions found!");
          filterStateMessage.revalidate();
          if (!filterStateDialog.isShowing())
            filterStateDialog.setVisible(true);
          filterStateDialog.toFront();
        }
      }
      else {
        ValuesFrequencies freq[][]=ruleSet.getFeatureValuesDistributions(selectedRules);
        if (freq==null || freq.length<1) {
          if (filterStateMessage == null)
            JOptionPane.showMessageDialog(FocusManager.getCurrentManager().getActiveWindow(),
                    "Failed to count the frequencies of the feature values!", "No frequencies obtained!",
                    JOptionPane.ERROR_MESSAGE);
          else {
            filterStateMessage.setText("Failed to count the frequencies of the feature values!");
            filterStateMessage.revalidate();
            if (!filterStateDialog.isShowing())
              filterStateDialog.setVisible(true);
            filterStateDialog.toFront();
          }
          return;
        }
        String info=selectedRules.size()+" rules selected by "+
                "applying filter conditions";
        if (filterStateMessage!=null) {
          filterStateMessage.setText(info);
          filterStateMessage.revalidate();
        }
        ArrayList filterTexts=filterUI.describeFilters();
        if (filterTexts!=null)
          for (int i=0; i<filterTexts.size(); i++)
            info+=" "+(i+1)+") "+filterTexts.get(i);
        if (filterResultsViewManager.viewCreated())
          filterResultsViewManager.updateData(freq, info);
        else {
          Component heatMapPane=filterResultsViewManager.createFeatureDistributionsDisplay(freq,info);
          if (heatMapPane!=null) {
            if (noFilterDistributionsFrame.getContentPane().getComponentCount()==0) {
              Component vis=distributionVisPanel.getComponent(0);
              distributionVisPanel.remove(0);
              distributionVisPanel.add(heatMapPane);
              distributionVisPanel.revalidate();
              noFilterDistributionsFrame.getContentPane().add(vis);
              noFilterDistributionsFrame.setVisible(true);
              Frame frame=null;
              Component c=filterUI.getParent();
              while (frame==null && c!=null) {
                if (c instanceof Frame)
                  frame=(Frame)c;
                else
                  c=c.getParent();
              }
              if (frame!=null)
                frame.toFront();
            }
            else {
              distributionVisPanel.remove(0);
              distributionVisPanel.add(heatMapPane);
              distributionVisPanel.revalidate();
            }
          }
        }
      }
    }
  }
}
