package TapasExplTreeViewer.ui;

import TapasDataReader.CommonExplanation;
import TapasDataReader.ExplanationItem;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.TableColumnModelEvent;
import javax.swing.event.TableColumnModelListener;
import javax.swing.table.AbstractTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;

public class TMrulesDefineSettings {

  public TMrulesDefineSettings (ArrayList<CommonExplanation> exList,
                                ArrayList<String> listOfFeatures,
                                Hashtable<String,float[]> attrMinMax,
                                Map<String, TreeSet<Float>> uniqueSortedValues,
                                Map<String, List<Float>> allValues,
                                String dataFolder)
  {
    SwingUtilities.invokeLater(() -> {
      AttributeRangeDialog dialog = new AttributeRangeDialog(exList, listOfFeatures, attrMinMax, uniqueSortedValues, allValues, dataFolder);
      dialog.setVisible(true);
    });
  }

}

class AttributeRangeDialog extends JFrame {

  private JLabel fileNameLabel;

  ArrayList<CommonExplanation> exList;
  ArrayList<String> listOfFeatures;
  Map<String, float[]> attrMinMax;
  Map<String, TreeSet<Float>> uniqueSortedValues;

  public AttributeRangeDialog(
          ArrayList<CommonExplanation> exList,
          ArrayList<String> listOfFeatures,
          Map<String, float[]> attrMinMax,
          Map<String, TreeSet<Float>> uniqueSortedValues,
          Map<String, List<Float>> allValues,
          String dataFolder)
  {
    this.exList=exList;
    this.listOfFeatures=listOfFeatures;
    this.attrMinMax=attrMinMax;
    this.uniqueSortedValues=uniqueSortedValues;

    setTitle("Attribute Range Definition");
    setSize(800, 600);
    setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

    AttributeTableModel model = new AttributeTableModel(listOfFeatures, attrMinMax, uniqueSortedValues, allValues);
    JTable table = new JTable(model);

    // Set up JComboBox for mode selection
    JComboBox<String> modeComboBox = new JComboBox<>(new String[]{"distinct","equal intervals", "quantiles"});
    table.getColumnModel().getColumn(5).setCellEditor(new DefaultCellEditor(modeComboBox));

    // Set up JTextField for quantile number input
    JTextField quantileField = new JTextField();
    table.getColumnModel().getColumn(6).setCellEditor(new DefaultCellEditor(quantileField));

    // Disable editing quantile numbers when mode is distinct
    table.getColumnModel().getColumn(6).setCellRenderer((table1, value, isSelected, hasFocus, row, column) -> {
      JTextField textField = new JTextField(value != null ? value.toString() : "");
      if (model.modeMap.get(model.attributes.get(row)).equals("distinct")) {
        textField.setEditable(false);
        textField.setBackground(Color.LIGHT_GRAY);
      } else {
        textField.setEditable(true);
        textField.setBackground(Color.WHITE);
      }
      return textField;
    });

    // Adjust column widths proportionally initially and on resize
    table.addComponentListener(new ComponentAdapter() {
      @Override
      public void componentResized(ComponentEvent e) {
        adjustColumnWidths(table);
      }
    });

    // Set up JScrollPane
    JScrollPane scrollPane = new JScrollPane(table);
    add(scrollPane, BorderLayout.CENTER);

    // Adjust column widths after the table is shown
    addComponentListener(new ComponentAdapter() {
      @Override
      public void componentShown(ComponentEvent e) {
        adjustColumnWidths(table);
      }
    });

    // Create and set up the panel for the button and file name label
    JPanel bottomPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
    JButton goButton = new JButton("Go");
    fileNameLabel = new JLabel("");

    // Add action listener to the button
    goButton.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        String fname=new File(ShowRules.RULES_FOLDER,generateFileName()).getPath();
        fileNameLabel.setText("File: " + fname);
        saveToFile(fname, model.modeMap, model.intervalsMap);
      }
    });

    bottomPanel.add(goButton);
    bottomPanel.add(fileNameLabel);

    // Add the panel to the frame
    add(bottomPanel, BorderLayout.SOUTH);
  }

/*
  private void saveToFile (ArrayList<CommonExplanation> exList, String fname) {
    try (BufferedWriter writer = new BufferedWriter(new FileWriter(fname))) {
      // Write header
      writer.write("ruleNumber,outcome,conditions");
      writer.newLine();

      // Write each rule
      for (int i = 0; i < exList.size(); i++) {
        CommonExplanation rule = exList.get(i);
        StringBuilder sb = new StringBuilder();
        sb.append(i + 1).append(","); // ruleNumber (starting from 1)
        sb.append(rule.action).append(","); // outcome
        for (ExplanationItem item : rule.eItems) {
          sb.append("[").append(item.interval[0]).append(" - ").append(item.interval[1]).append("] ");
        }
        // Remove the last space
        sb.setLength(sb.length() - 1);
        writer.write(sb.toString());
        writer.newLine();
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
*/

  private void saveToFile(String fname, Map<String,String> modeMap, Map<String, List<Float>> intervalsMap) {
    try (BufferedWriter writer = new BufferedWriter(new FileWriter(fname))) {
      // Write header
      writer.write("ruleNumber,ruleAsText,");
      for (String attribute: listOfFeatures)
        writer.write(attribute+",");
      if (Float.isNaN(exList.get(0).meanQ))
        writer.write("outcome,conditions,conditionsAsMasks");
      else
        writer.write("outcome_min,outcome_avg,outcome_max,conditions,conditionsAsMasks");
      writer.newLine();

      // Prepare interval mapping for each attribute
      Map<String, List<String>> labelsMap = new HashMap<>();
      Map<String, List<Double>> breaksMap = new HashMap<>();

      for (String attribute : attrMinMax.keySet()) {
        List<String> labels = new ArrayList<>();
        List<Double> breaks = new ArrayList<>();

        if (modeMap.get(attribute).equals("distinct")) {
          // Distinct mode
          List<Float> values = new ArrayList<>(uniqueSortedValues.get(attribute));
          int numIntervals = values.size();
          int numDigits = String.valueOf(numIntervals).length();
          for (int i = 0; i < values.size(); i++) {
            labels.add(attribute + "_" + String.format("%0" + numDigits + "d", i));
            breaks.add((double) values.get(i));
          }
          breaks.add(Double.MAX_VALUE); // Add max value for the last interval
        } else {
          // Quantiles or equal intervals mode
          int n=intervalsMap.get(attribute).size();
          float[] minMax = attrMinMax.get(attribute);
          float min = minMax[0];
          float max = minMax[1];
          breaks.add((double) min);
          for (int i = 0; i < n; i++) {
            breaks.add((double)intervalsMap.get(attribute).get(i));
          }
          breaks.add((double) max);
          n+=1;
          int numDigits = String.valueOf(n).length();
          for (int i = 0; i < n; i++) {
            labels.add(attribute + "_" + String.format("%0" + numDigits + "d", i));
          }
        }
        labelsMap.put(attribute, labels);
        breaksMap.put(attribute, breaks);
      }

      // Write each rule
      for (int i = 0; i < exList.size(); i++) {
        CommonExplanation rule = exList.get(i);
        StringBuilder sb = new StringBuilder();
        sb.append(i + 1).append(","); // ruleNumber (starting from 1)
        // write rule as text
        sb.append(ruleToPlainString(rule)).append(",");
        //sb.append((Double.isNaN(rule.meanQ))?rule.action:rule.minQ+".."+rule.maxQ).append(","); // outcome

        // write masks for all attributes
        for (String attribute: listOfFeatures) {
          String mask="";
          for (ExplanationItem item : rule.eItems)
            if (attribute.equals(item.attr)) {
              List<String> intervals = labelsMap.get(attribute);
              List<Double> breaks = breaksMap.get(attribute);
              double start = item.interval[0];
              double end = item.interval[1];
              if (Double.isInfinite(end))
                end=attrMinMax.get(attribute)[1];
              for (int j = 0; j < intervals.size(); j++) {
                double intervalStart = breaks.get(j);
                double intervalEnd = breaks.get(j + 1);
                if (start < intervalEnd && end >= intervalStart)
                  mask+="1";
                else
                  mask+="0";
              }
            }
          sb.append(mask).append(",");
        }

        // write outcome
        if (Double.isNaN(rule.meanQ))
          sb.append(rule.action);
        else
          sb.append(rule.minQ+","+rule.meanQ+","+rule.maxQ);
        sb.append(",");

        // write conditions as texts for topic modelling
        for (String attribute: listOfFeatures)
          for (ExplanationItem item : rule.eItems)
            if (attribute.equals(item.attr)) {
              List<String> intervals = labelsMap.get(attribute);
              List<Double> breaks = breaksMap.get(attribute);
              double start = item.interval[0];
              double end = item.interval[1];
              if (Double.isInfinite(end))
                end=attrMinMax.get(attribute)[1];
              if (start<breaks.get(0) && end>=breaks.get(breaks.size()-2))
                continue; //all intervals are present; not informative
              for (int j = 0; j < intervals.size(); j++) {
                double intervalStart = breaks.get(j);
                double intervalEnd = breaks.get(j + 1);
                if (start < intervalEnd && end >= intervalStart)
                  sb.append(intervals.get(j)).append(" ");
              }
            }
        // Remove the last space
        if (sb.length() > 0 && sb.charAt(sb.length() - 1) == ' ')
          sb.setLength(sb.length() - 1);

        sb.append(",");
        // write conditions as masks for topic modelling
        for (String attribute: listOfFeatures) {
          String mask="";
          for (ExplanationItem item : rule.eItems)
            if (attribute.equals(item.attr)) {
              List<String> intervals = labelsMap.get(attribute);
              List<Double> breaks = breaksMap.get(attribute);
              double start = item.interval[0];
              double end = item.interval[1];
              if (Double.isInfinite(end))
                end=attrMinMax.get(attribute)[1];
              if (start<breaks.get(0) && end>=breaks.get(breaks.size()-2))
                continue; //all intervals are present; not informative
              for (int j = 0; j < intervals.size(); j++) {
                double intervalStart = breaks.get(j);
                double intervalEnd = breaks.get(j + 1);
                if (start < intervalEnd && end >= intervalStart)
                  mask+="1";
                else
                  mask+="0";
              }
            }
          if (mask.length()>0 && mask.contains("0")) //al least one 0 symbol
            sb.append(attribute).append("__").append(mask).append(" ");
        }

        writer.write(sb.toString().trim());
        writer.newLine();
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private String ruleToPlainString(CommonExplanation rule) {
    StringBuilder sb = new StringBuilder();
    for (String attr: listOfFeatures)
      for (ExplanationItem item : rule.eItems)
        if (attr.equals(item.attr))
          sb.append(item.attr+":[").append(item.interval[0]).append("..").append(item.interval[1]).append("] ");
    sb.append("=> ").append((Double.isNaN(rule.meanQ))?rule.action:rule.meanQ+"("+rule.minQ+".."+rule.maxQ+")");
    return sb.toString();
  }

  private void adjustColumnWidths(JTable table) {
    int tableWidth = table.getWidth();
    int intervalColumnWidth = (int) (tableWidth * 0.5);
    int otherColumnWidth = (tableWidth - intervalColumnWidth) / (table.getColumnCount() - 1);

    for (int i = 0; i < table.getColumnCount(); i++) {
      if (i == 7) {
        table.getColumnModel().getColumn(i).setPreferredWidth(intervalColumnWidth);
      } else {
        table.getColumnModel().getColumn(i).setPreferredWidth(otherColumnWidth);
      }
    }
  }

  private String generateFileName() {
    SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd-HHmmss");
    return "TM_" + sdf.format(new Date()) + ".csv";
  }

}

class AttributeTableModel extends AbstractTableModel {
  private final String[] columnNames = {
          "Attribute", "Min", "Max", "Count of Distinct Values",
          "Count of All Values", "Mode", "Number of Quantiles", "Class Intervals"
  };

  public final List<String> attributes;
  private final Map<String, float[]> attrMinMax;
  private final Map<String, TreeSet<Float>> uniqueSortedValues;
  private final Map<String, List<Float>> allValues;
  public final Map<String, String> modeMap;
  private final Map<String, Integer> quantileMap;
  public final Map<String, List<Float>> intervalsMap;

  public AttributeTableModel (ArrayList<String> listOfFeatures,
                              Map<String, float[]> attrMinMax,
                              Map<String, TreeSet<Float>> uniqueSortedValues,
                              Map<String, List<Float>> allValues)
  {
    this.attributes = listOfFeatures; // new ArrayList<>(attrMinMax.keySet());
    this.attrMinMax = attrMinMax;
    this.uniqueSortedValues = uniqueSortedValues;
    this.allValues = allValues;
    this.modeMap = new HashMap<String,String>();
    this.quantileMap = new HashMap<>();
    this.intervalsMap = new HashMap<>();

    for (String attribute : attributes) {
      int uniqueCount = uniqueSortedValues.get(attribute).size();
      if (uniqueCount <= 5) {
        modeMap.put(attribute, "distinct"); // Distinct mode
      } else {
        modeMap.put(attribute, "equal intervals"); // equal intervals mode
        quantileMap.put(attribute, 4); // Default number of intervals
      }
      updateClassIntervals(attribute);
    }
  }

  @Override
  public int getRowCount() {
    return attributes.size();
  }

  @Override
  public int getColumnCount() {
    return columnNames.length;
  }

  @Override
  public Object getValueAt(int rowIndex, int columnIndex) {
    String attribute = attributes.get(rowIndex);
    switch (columnIndex) {
      case 0: return attribute;
      case 1: return attrMinMax.get(attribute)[0];
      case 2: return attrMinMax.get(attribute)[1];
      case 3: return uniqueSortedValues.get(attribute).size();
      case 4: return allValues.get(attribute).size();
      case 5: return modeMap.get(attribute);
      case 6: return quantileMap.get(attribute);
      case 7: return intervalsMap.get(attribute).toString();
      default: return null;
    }
  }

  @Override
  public boolean isCellEditable(int rowIndex, int columnIndex) {
    String attribute = attributes.get(rowIndex);
    if (columnIndex == 5) {
      return true;
    }
    if (columnIndex == 6) {
      return !modeMap.get(attribute).equals("distinct");
    }
    return false;
  }

  @Override
  public void setValueAt(Object aValue, int rowIndex, int columnIndex) {
    String attribute = attributes.get(rowIndex);
    if (columnIndex == 5) {
      modeMap.put(attribute,aValue.toString());
      if (modeMap.get(attribute).equals("distinct")) {
        quantileMap.remove(attribute);
      }
      updateClassIntervals(attribute);
      fireTableCellUpdated(rowIndex, 7);
    } else if (columnIndex == 6) {
      try {
        int quantiles = Integer.parseInt(aValue.toString());
        quantileMap.put(attribute, quantiles);
        updateClassIntervals(attribute);
        fireTableCellUpdated(rowIndex, 7);
      } catch (NumberFormatException e) {
        // Handle invalid number input
      }
    }
    fireTableCellUpdated(rowIndex, columnIndex);
  }

  private void updateClassIntervals(String attribute) {
    if (modeMap.get(attribute).equals("distinct")) {
      intervalsMap.put(attribute, new ArrayList<>(uniqueSortedValues.get(attribute)));
    } else {
      // Ensure quantileMap has an entry for the attribute
      if (!quantileMap.containsKey(attribute))
        quantileMap.put(attribute, 4); // Default quantiles
      int n=quantileMap.get(attribute);
      if (modeMap.get(attribute).equals("quantiles")) {
        // Calculate quantiles
        List<Float> values=new ArrayList<>(new HashSet<>(allValues.get(attribute))); // Remove duplicates
        Collections.sort(values);
        List<Float> quantiles=new ArrayList<>();
        for (int i=1; i<n; i++) {
          int index=(int) Math.ceil(i*values.size()/(double) n)-1;
          // Ensure index is within bounds
          index=Math.min(index, values.size()-1);
          float breakPoint=(values.get(index)+values.get(Math.min(index+1, values.size()-1)))/2;
          if (!quantiles.contains(breakPoint)) {
            quantiles.add(breakPoint);
          }
        }
        quantileMap.put(attribute, quantiles.size()+1); // Update the number of quantiles
        intervalsMap.put(attribute, quantiles);
      }
      else {
        //equal intervals
        float minmax[]=attrMinMax.get(attribute);
        double min=(double)minmax[0], step=(minmax[1]-min)/n;
        List<Float> breaks=new ArrayList<>();
        for (int i=1; i<n; i++)
          breaks.add((float)(min+i*step));
        intervalsMap.put(attribute,breaks);
      }
    }
  }

  @Override
  public String getColumnName(int column) {
    return columnNames[column];
  }

  @Override
  public Class<?> getColumnClass(int columnIndex) {
    switch (columnIndex) {
      case 1: case 2: return Float.class;
      case 3: case 4: case 6: return Integer.class;
      case 5: return String.class;
      default: return Object.class;
    }
  }
}
