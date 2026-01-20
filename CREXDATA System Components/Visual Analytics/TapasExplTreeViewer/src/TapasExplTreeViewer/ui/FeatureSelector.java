package TapasExplTreeViewer.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.HashSet;

public class FeatureSelector extends JPanel {
  public ArrayList<String> features=null;
  protected JCheckBox cbFeatures[]=null;

  public FeatureSelector(ArrayList<String> features, HashSet<String> selected) {
    this.features=features;
    if (features==null || features.isEmpty())
      return;
    cbFeatures=new JCheckBox[features.size()];
    JPanel mainP=new JPanel();
    mainP.setLayout(new GridLayout(0,1));
    for (int i=0; i<features.size(); i++) {
      cbFeatures[i]=new JCheckBox(features.get(i),selected==null || selected.contains(features.get(i)));
      mainP.add(cbFeatures[i]);
    }
    JPanel p=new JPanel();
    p.setLayout(new FlowLayout(FlowLayout.CENTER,20,5));
    JButton b=new JButton("Select all");
    p.add(b);
    b.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        for (JCheckBox cb:cbFeatures)
          cb.setSelected(true);
      }
    });
    b=new JButton("Deselect all");
    p.add(b);
    b.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        for (JCheckBox cb:cbFeatures)
          cb.setSelected(false);
      }
    });
    setLayout(new BorderLayout());
    add(mainP,BorderLayout.CENTER);
    add(p,BorderLayout.SOUTH);
  }

  public HashSet<String> getSelection() {
    if (cbFeatures==null)
      return null;
    HashSet<String> selection=new HashSet<String>(cbFeatures.length);
    for (JCheckBox cb:cbFeatures)
      if (cb.isSelected())
        selection.add(cb.getText());
    if (selection.isEmpty())
      return null;
    return selection;
  }
}
