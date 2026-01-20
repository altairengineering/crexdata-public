package TapasExplTreeViewer;

import TapasDataReader.CommonExplanation;
import TapasDataReader.Explanation;
import TapasDataReader.ExplanationItem;
import TapasDataReader.Flight;
import TapasExplTreeViewer.MST.Edge;
import TapasExplTreeViewer.MST.Prim;
import TapasExplTreeViewer.MST.Vertex;
import TapasExplTreeViewer.ui.ShowRules;
import TapasExplTreeViewer.vis.ProjectionPlot2D;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.LineBorder;
import java.io.*;
import java.util.*;

public class SeeExList {
  
  public static Border highlightBorder=new LineBorder(ProjectionPlot2D.highlightColor,1);

  public static void main(String[] args) {
    String parFileName = (args != null && args.length > 0) ? args[0] : "params.txt";

    if (!parFileName.startsWith("params")) {
      mainSingleFile(parFileName);
      return;
    }
  
    String path=null;
    Hashtable<String,String> fNames=new Hashtable<String,String>(10);
    try {
      BufferedReader br = new BufferedReader(
          new InputStreamReader(
              new FileInputStream(new File(parFileName)))) ;
      String strLine;
      try {
        while ((strLine = br.readLine()) != null) {
          String str=strLine.replaceAll("\"","").replaceAll(" ","");
          String[] tokens=str.split("=");
          if (tokens==null || tokens.length<2)
            continue;
          String parName=tokens[0].trim().toLowerCase();
          if (parName.equals("path") || parName.equals("data_path"))
            path=tokens[1].trim();
          else
            fNames.put(parName,tokens[1].trim());
        }
      } catch (IOException io) {
        System.out.println(io);
      }
    } catch (IOException io) {
      System.out.println(io);
    }
    if (path!=null) {
      for (Map.Entry<String,String> e:fNames.entrySet()) {
        String fName=e.getValue();
        if (!fName.startsWith("\\") && !fName.contains(":\\")) {
          fName=path+fName;
          fNames.put(e.getKey(),fName);
        }
      }
    }
    else
      path="";
  
    String fName=fNames.get("decisions");
    if (fName==null) {
      System.out.println("No decisions file name in the parameters!");
      System.exit(1);
    }
  
    System.out.println("Decisions file name = "+fName);
    /**/
    TreeSet<Integer> steps=TapasDataReader.Readers.readStepsFromDecisions(fName);
    //System.out.println(steps);
    Hashtable<String, Flight> flights=
        TapasDataReader.Readers.readFlightDelaysFromDecisions(fName,steps);
    if (flights==null || flights.isEmpty()) {
      System.out.println("Failed to get original data!");
      System.exit(1);
    }
    Hashtable<String,float[]> attrMinMax=new Hashtable<String, float[]>();
    TapasDataReader.Readers.readExplanations(path,steps,flights,attrMinMax);
    /**/
  
    ArrayList<CommonExplanation> exList=new ArrayList<CommonExplanation>(10000);
    ArrayList<Explanation> dataInstances=new ArrayList<Explanation>(100000);
    boolean actionsDiffer=false;
    
    for (Map.Entry<String,Flight> entry:flights.entrySet()) {
      Flight f=entry.getValue();
      if (f.expl!=null)
        for (int i=0; i<f.expl.length; i++)
          if (f.expl[i]!=null) {
            CommonExplanation.addExplanation(exList, f.expl[i],
                false, attrMinMax, true);
            dataInstances.add(f.expl[i]);
            actionsDiffer=actionsDiffer ||
                              (dataInstances.size()>1 && f.expl[i].action!=dataInstances.get(0).action);
          }
    }
    if (exList.isEmpty()) {
      System.out.println("Failed to reconstruct the list of common explanations!");
      System.exit(1);
    }
    else
      System.out.println("Made a list of "+exList.size()+" common explanations!");
    
    //MainBody(attrMinMax,exList);
    ShowRules showRules=new ShowRules(exList,attrMinMax);
    //showRules.setDataInstances(dataInstances,actionsDiffer);
    showRules.countRightAndWrongRuleApplications();

    showRules.showRulesInTable();
    JFrame fr=showRules.getMainFrame();
    if (fr==null) {
      System.out.println("Failed to visualize the rules!");
      System.exit(1);
    }
  }

  public static void mainSingleFile (String fname) {
    System.out.println("* loading rules and data from "+fname);
    Hashtable<String,float[]> attrMinMax=new Hashtable<String, float[]>();
    Vector<String> attrs=new Vector<>();
    Vector<Explanation> vex=new Vector<>();
    ArrayList<CommonExplanation> exList=new ArrayList<CommonExplanation>(10000);
    boolean bAllInts=true;
    try {
      BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(fname))));
      String strLine=br.readLine();
      int line=0;
      String s[]=strLine.split(",");
      for (int i=0; i<s.length; i++) {
        float minmax[]=new float[2];
        minmax[0]=Integer.MAX_VALUE;
        minmax[1]=Integer.MIN_VALUE;
        attrMinMax.put(s[i],minmax);
        attrs.add(s[i]);
      }
      while ((strLine = br.readLine()) != null) {
        line++;
        s=strLine.split(",");
        String srule[]=s[1].split("&");
        Explanation ex=new Explanation();
        ex.eItems=new ExplanationItem[srule.length];
        ex.FlightID=""+(line-1);
        ex.step=0;
        ex.Q=Float.valueOf(s[2]);
        bAllInts&=ex.Q==Math.round(ex.Q);
        //ex.action=(int)ex.Q; // ToDo
        vex.add(ex);
        for (int i=0; i<srule.length; i++) {
          int p=srule[i].indexOf("=");
          int attrIdx=-1;
          try {
            attrIdx=Integer.valueOf(srule[i].substring(0,p)).intValue();
          } catch (NumberFormatException nfe) {
            System.out.println("Error in line "+line+": extracting attr name from rule item # "+i+" "+srule[i]);
          }
          ExplanationItem ei=new ExplanationItem();
          ei.attr=attrs.elementAt(attrIdx);
          float minmax[]=attrMinMax.get(ei.attr);
          String ss=srule[i].substring(p+1);
          int p1=ss.indexOf("<="), p2=ss.indexOf(">");
          ei.value=Float.MIN_VALUE;
          try {
            ei.value=Float.valueOf(ss.substring(0,Math.max(p1,p2))).floatValue();
          } catch (NumberFormatException nfe) {
            System.out.println("Error in line "+line+": extracting attr value from rule item # "+i+" "+srule[i]);
          }
          boolean changed=false;
          if (ei.value<minmax[0]) {
            minmax[0]=ei.value;
            changed=true;
          }
          if (ei.value>minmax[1]) {
            minmax[1]=ei.value;
            changed=true;
          }
          if (changed)
            attrMinMax.put(ei.attr,minmax);
          String sss=ss.substring((p1>=0)?p1+2:p2+1);
          double d=Double.NaN;
          try {
            d=Double.valueOf(sss).doubleValue();
          } catch (NumberFormatException nfe) {
            System.out.println("Error in line "+line+": extracting condition from rule item # "+i+" "+srule[i]);
          }
          if (p1>=0) // condition <=
            ei.interval=new double[]{Double.NEGATIVE_INFINITY,d};
          else  // condition >
            ei.interval=new double[]{d,Double.POSITIVE_INFINITY};
          ei.attr_core=ei.attr;
          ei.sector="None";
          ex.eItems[i]=ei;
        }
      }
      br.close();
    } catch (IOException io) {
      System.out.println(io);
    }
    if (bAllInts)
      for (Explanation ex:vex)
        ex.action=(int)ex.Q;
    for (Explanation ex:vex)
      CommonExplanation.addExplanation(exList,ex,false,attrMinMax,true);
    //MainBody(attrMinMax,exList);
  
    boolean actionsDiffer=false;
    for (int i=1; i<vex.size() && !actionsDiffer; i++)
      actionsDiffer=vex.elementAt(i).action!=vex.elementAt(i-1).action;
    
    ShowRules showRules=new ShowRules(exList,attrMinMax);
    showRules.setDataInstances(vex,actionsDiffer);
    showRules.countRightAndWrongRuleApplications();
    showRules.showRulesInTable();
    JFrame fr=showRules.getMainFrame();
    if (fr==null) {
      System.out.println("Failed to visualize the rules!");
      System.exit(1);
    }
    //givenAGraph_whenPrimRuns_thenPrintMST();
  }

  public static List<Vertex> createGraph() {
    List<Vertex> graph = new ArrayList<>();
    Vertex a = new Vertex("A");
    Vertex b = new Vertex("B");
    Vertex c = new Vertex("C");
    Vertex d = new Vertex("D");
    Vertex e = new Vertex("E");
    Edge ab = new Edge(2d);
    a.addEdge(b, ab);
    b.addEdge(a, ab);
    Edge ac = new Edge(3d);
    a.addEdge(c, ac);
    c.addEdge(a, ac);
    Edge bc = new Edge(2d);
    b.addEdge(c, bc);
    c.addEdge(b, bc);
    Edge be = new Edge(5d);
    b.addEdge(e, be);
    e.addEdge(b, be);
    Edge cd = new Edge(1d);
    c.addEdge(d, cd);
    d.addEdge(c, cd);
    Edge ce = new Edge(1d);
    c.addEdge(e, ce);
    e.addEdge(c, ce);
    graph.add(a);
    graph.add(b);
    graph.add(c);
    graph.add(d);
    graph.add(e);
    return graph;
  }
  public static void givenAGraph_whenPrimRuns_thenPrintMST() {
    Prim prim = new Prim(createGraph());
    System.out.println(prim.originalGraphToString());
    System.out.println("----------------");
    prim.run();
    System.out.println();
    prim.resetPrintHistory();
    System.out.println(prim.minimumSpanningTreeToString());
  }

}
