package TapasExplTreeViewer.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Collections;

public class IntervalBreaksDialog extends JDialog {
  private ArrayList<Double> breaks=new ArrayList<Double>();
  private JTextField breaksField;
  private JTextField numIntervalsField;
  private double minValue, maxValue;

  public IntervalBreaksDialog(Frame parent, double minValue, double maxValue) {
    super(parent, "Edit Breaks", true);
    this.minValue = minValue;
    this.maxValue = maxValue;

    JPanel mainPanel = new JPanel(new GridLayout(0, 1));
    mainPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

    // Label and text field for breaks
    mainPanel.add(new JLabel("Enter break values (space-separated):"));
    JPanel p=new JPanel(new BorderLayout(5,5));
    p.add(new JLabel(String.format("%.4f",minValue)),BorderLayout.WEST);
    breaksField = new JTextField(getBreaksAsString(),50);
    p.add(breaksField,BorderLayout.CENTER);
    p.add(new JLabel(String.format("%.4f",maxValue)),BorderLayout.EAST);
    mainPanel.add(p);

    breaksField.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        updateBreaksFromField();
        if (breaks.isEmpty())
          JOptionPane.showMessageDialog(IntervalBreaksDialog.this,
              "Invalid input. Please enter numeric values separated by spaces.",
              "Error", JOptionPane.ERROR_MESSAGE);
      }
    });

    // Controls for automatic division
    mainPanel.add(new JLabel("Automatic division",JLabel.CENTER));
    p=new JPanel(new FlowLayout(FlowLayout.CENTER,10,0));
    mainPanel.add(p);
    p.add(new JLabel("Number of intervals:"));
    numIntervalsField = new JTextField("5",2);
    p.add(numIntervalsField);
    JButton autoDivideButton = new JButton("Divide");
    autoDivideButton.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        divideIntoEqualIntervals();
      }
    });
    p.add(autoDivideButton);

    // Buttons for Apply and Cancel
    JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT,20,0));
    JButton applyButton = new JButton("Apply");
    JButton cancelButton = new JButton("Cancel");

    applyButton.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        updateBreaksFromField();
        if (breaks.isEmpty())
          JOptionPane.showMessageDialog(IntervalBreaksDialog.this,
              "Invalid input. Please enter numeric values separated by spaces.",
              "Error", JOptionPane.ERROR_MESSAGE);
        else
          dispose();
      }
    });

    cancelButton.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        breaks=null;
        dispose();
      }
    });

    buttonPanel.add(applyButton);
    buttonPanel.add(cancelButton);
    mainPanel.add(buttonPanel, BorderLayout.SOUTH);

    add(mainPanel);
    pack();
    setLocationRelativeTo(parent);
  }

  private String getBreaksAsString() {
    if (breaks==null || breaks.isEmpty())
      return "";
    String str=String.format("%.4f",breaks.get(0));
    for (int i=1; i<breaks.size(); i++)
      str+=String.format(" %.4f",breaks.get(i));
    return str;
  }

  private void updateBreaksFromField() {
    String[] parts = breaksField.getText().trim().split("\\s+");
    breaks.clear();
    for (String part : parts) {
      try {
        double b=Double.parseDouble(part);
        if (b<=minValue || b>=maxValue)
          JOptionPane.showMessageDialog(IntervalBreaksDialog.this,
              String.format("Invalid break value: ["+part+"]; must be between %.4f and %.4f",minValue,maxValue),
              "Error", JOptionPane.ERROR_MESSAGE);
        else
          breaks.add(b);
      } catch (NumberFormatException ex) {
        JOptionPane.showMessageDialog(IntervalBreaksDialog.this,
            "Invalid input: ["+part+"]",
            "Error", JOptionPane.ERROR_MESSAGE);
      }
    }
    Collections.sort(breaks);
    breaksField.setText(getBreaksAsString());
  }

  private void divideIntoEqualIntervals() {
    try {
      int numIntervals = Integer.parseInt(numIntervalsField.getText().trim());
      if (numIntervals < 2) throw new NumberFormatException("Invalid interval count");

      double step = (maxValue - minValue) / numIntervals;
      breaks.clear();
      for (int i=1; i<numIntervals; i++)
        breaks.add(minValue + i * step);

      breaksField.setText(getBreaksAsString());
    } catch (NumberFormatException ex) {
      JOptionPane.showMessageDialog(this,
          "Please enter a valid positive integer for the number of intervals.",
          "Error", JOptionPane.ERROR_MESSAGE);
    }
  }

  public ArrayList<Double> getBreaks() {
    return breaks;
  }

}
