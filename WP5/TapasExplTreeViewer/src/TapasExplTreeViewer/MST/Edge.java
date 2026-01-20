package TapasExplTreeViewer.MST;

public class Edge {

  private double weight;
  private boolean isIncluded = false;
  private boolean isPrinted = false;

  public Edge(double weight) {
    this.weight = weight;
  }

  public double getWeight() {
    return weight;
  }

  public void setWeight(int weight) {
    this.weight = weight;
  }

  public boolean isIncluded() {
    return isIncluded;
  }

  public void setIncluded(boolean included) {
    isIncluded = included;
  }

  public boolean isPrinted() {
    return isPrinted;
  }

  public void setPrinted(boolean printed) {
    isPrinted = printed;
  }
}
