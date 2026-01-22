using UnityEngine;

public class InfoObject : MonoBehaviour
{
    [Header("Dialog Text for This Object")]
    [SerializeField] private string header = "My Header";
    [SerializeField, TextArea] private string body = "Description or details...";

    public void ShowInfo()
    {
        if (SimplePOIInformer.Instance == null) return;
        SimplePOIInformer.Instance.ShowInfoPanel();
        SimplePOIInformer.Instance.SetInfoText(header, body);
    }

    public void UpdateMyText(string newBodyText)
    {
        if (SimplePOIInformer.Instance == null) return;
        SimplePOIInformer.Instance.UpdateMainText(newBodyText);
    }
}
