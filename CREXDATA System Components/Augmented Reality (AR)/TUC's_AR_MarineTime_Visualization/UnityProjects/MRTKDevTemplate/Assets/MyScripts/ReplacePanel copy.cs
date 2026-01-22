using UnityEngine;
using MixedReality.Toolkit.UX;

public class ReplacePanel : MonoBehaviour
{
    public GameObject initialPanel;  // The initial panel to be moved
    public GameObject button;        // The button to trigger the replacement
    public GameObject replacementPanelPrefab;  // The prefab for the replacement panel

    private void Start()
    {
        // Register the button's press event
        if (button != null)
        {
            PressableButton pressableButton = button.GetComponent<PressableButton>();
            if (pressableButton != null)
            {
                pressableButton.OnClicked.AddListener(OnButtonClicked);
            }
        }
    }

    private void OnButtonClicked()
    {
        if (initialPanel != null && replacementPanelPrefab != null)
        {
            // Get the position, rotation, and scale of the initial panel
            Vector3 position = initialPanel.transform.position;
            Quaternion rotation = initialPanel.transform.rotation;
            Vector3 scale = initialPanel.transform.localScale;

            // Calculate the actual size of the initial panel in world units
            Vector3 actualSize = new Vector3(
                initialPanel.GetComponent<Renderer>().bounds.size.x / initialPanel.transform.localScale.x,
                initialPanel.GetComponent<Renderer>().bounds.size.y / initialPanel.transform.localScale.y,
                initialPanel.GetComponent<Renderer>().bounds.size.z / initialPanel.transform.localScale.z
            );

            // Destroy the initial panel
            Destroy(initialPanel);

            // Instantiate the replacement panel at the same position, rotation, and scale
            GameObject replacementPanelInstance = Instantiate(replacementPanelPrefab, position, rotation);
            replacementPanelInstance.transform.localScale = actualSize;  // Ensure the same actual size
        }
        else
        {
            Debug.LogError("Initial panel or replacement panel prefab are not assigned.");
        }
    }
}
