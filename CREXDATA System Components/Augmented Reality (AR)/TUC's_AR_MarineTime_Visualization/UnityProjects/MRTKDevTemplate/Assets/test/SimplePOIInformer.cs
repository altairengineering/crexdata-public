using UnityEngine;
using TMPro;

public class SimplePOIInformer : MonoBehaviour
{
    // Singleton instance
    public static SimplePOIInformer Instance { get; private set; }

    [Header("References")]
    [SerializeField] private TextMeshProUGUI headerText;
    [SerializeField] private TextMeshProUGUI mainText;

    [SerializeField] private GameObject infoPanel;

    private Animator animator;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        infoPanel.SetActive(false);  // optional

        // Grab the Animator component
        animator = GetComponent<Animator>();
    }

    public void ShowInfoPanel()
    {
        if (animator != null)
        {
            animator.SetBool("IsVisible", true);
        }
        infoPanel.SetActive(true);
    }

    public void HideInfoPanel()
    {
        if (animator != null)
        {
            animator.SetBool("IsVisible", false);
        }
        //infoPanel.SetActive(false);
    }

    public void SetInfoText(string newHeader, string newBody)
    {
        if (!string.IsNullOrEmpty(newHeader))
            headerText.text = newHeader;
        if (!string.IsNullOrEmpty(newBody))
            mainText.text = newBody;
    }

    public void UpdateMainText(string newText)
    {
        if (!string.IsNullOrEmpty(newText))
        {
            mainText.text = newText;
        }
    }

    public void OnHideComplete()
    {
        infoPanel.SetActive(false);
    }
}
