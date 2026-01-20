package TapasExplTreeViewer.ui;

import TapasDataReader.ExTreeNode;
import TapasDataReader.ExplanationItem;

import javax.swing.*;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreeNode;
import javax.swing.tree.TreePath;
import java.awt.*;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Map;

/**
 * Visualizes a collection of explanation trees
 */
public class ExTreePanel extends JPanel {
  /**
   * The collection of trees, each explains one type of decision
   */
  public Hashtable<Integer,ExTreeNode> topNodes=null;
  /**
   * The root of the Swing tree
   */
  protected ExSwingTreeNode root=null;
  /**
   * The visual representation of the tree collection
   */
  protected JTree exTree=null;
  
  public ExTreePanel(Hashtable<Integer,ExTreeNode> topNodes) {
    super();
    this.topNodes=topNodes;
    if (topNodes==null || topNodes.isEmpty())
      return;
    int nUses=0, nLeaves=0;
    for (Map.Entry<Integer,ExTreeNode> e:topNodes.entrySet()) {
      nUses += e.getValue().nUses;
      nLeaves+=e.getValue().nLeavesBelow;
    }
    root=new ExSwingTreeNode("All; uses: "+nUses+"; terminals: "+nLeaves,null);
    for (Map.Entry<Integer,ExTreeNode> e:topNodes.entrySet())
      attachNode(root, e.getValue());
    exTree=new JTree(root);
    exTree.setExpandsSelectedPaths(true);
    setLayout(new BorderLayout());
    add(new JScrollPane(exTree),BorderLayout.CENTER);
  
    Dimension size=Toolkit.getDefaultToolkit().getScreenSize();
    setPreferredSize(new Dimension(Math.round(0.25f*size.width), Math.round(0.4f*size.height)));
  }
  
  protected void attachNode(DefaultMutableTreeNode parent, ExTreeNode exNode) {
    if (exNode==null)
      return;
    ExSwingTreeNode node=new ExSwingTreeNode(exNode.getLabel()+
                                                 "; uses: "+exNode.nUses+
                                                 "; terminals: "+exNode.nLeavesBelow,exNode);
    parent.add(node);
    if (exNode.children!=null)
      for (ExTreeNode child:exNode.children)
        attachNode(node,child);
  }
  
  public void expandExplanation(int action, ExplanationItem eItems[]) {
    collapseChildren(root);
    ExSwingTreeNode node=root.findNodeForExplanation(action, eItems);
    if (node!=null) {
      TreePath path=new TreePath(node.getPath());
      if (path!=null) {
        exTree.setSelectionPath(path);
        exTree.scrollPathToVisible(path);
      }
    }
  }
  
  public void highlightExplanationItem(int action, ExplanationItem eItems[], int itemIdx) {
    if (itemIdx<0 || eItems==null || itemIdx>=eItems.length) {
      expandExplanation(action, eItems);
      return;
    }
    ExSwingTreeNode node=root.findNodeForExplanation(action, eItems);
    if (node==null)
      return;
    
    TreeNode nodeSeq[]=node.getPath();
    itemIdx+=nodeSeq.length-eItems.length; //the first one or two nodes in the sequence
                                           //may correspond to the root of the whole tree or
                                           //the root of the subtree corresponding to the action
    
    TreePath path=new TreePath(((ExSwingTreeNode)nodeSeq[itemIdx]).getPath());
    if (path==null)
      return;
    if (path!=null) {
      exTree.setSelectionPath(path);
      exTree.scrollPathToVisible(path);
    }
  }
  
  public void expandPathsToNodes(ArrayList<ExSwingTreeNode> nodes) {
    if (nodes==null || nodes.isEmpty())
      return;
    TreePath path=null;
    for (ExSwingTreeNode node:nodes)
      exTree.setSelectionPath(path=new TreePath(node.getPath()));
    if (path!=null)
      exTree.scrollPathToVisible(path);
  }
  
  public void expandToLevel(int level) {
    ArrayList<ExSwingTreeNode> nodes=new ArrayList<ExSwingTreeNode>(1000);
    root.getNodesUpToLevel(level,nodes);
    expandPathsToNodes(nodes);
  }
  
  public void collapseAllBranches(){
    collapseChildren(root);
  }
  
  public void collapseChildren(ExSwingTreeNode node) {
    if (node==null || node.getChildCount()<1)
      return;
    for (int i=0; i<node.getChildCount(); i++) {
      ExSwingTreeNode child=(ExSwingTreeNode)node.getChildAt(i);
      if (child.getChildCount()>0)
        collapseChildren(child);
    }
    TreePath path=new TreePath(node.getPath());
    exTree.collapsePath(path);
  }
  
  public void expandNodesWithAttribute(String attrName){
    ArrayList<ExSwingTreeNode> nodes=new ArrayList<ExSwingTreeNode>(1000);
    root.getNodesWithAttribute(attrName,nodes);
    if (nodes.isEmpty())
      return;
    collapseAllBranches();
    expandPathsToNodes(nodes);
  }
  
}
