using UnityEngine;

using MixedReality.Toolkit.UX;

public class CreatePanelBetweenCubes : MonoBehaviour
{
    public GameObject cube1;
    public GameObject cube2;
    public GameObject button;
    public GameObject panelPrefab;
    public GameObject lookAtTarget; // Reference to the sphere

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
        if (cube1 != null && cube2 != null && panelPrefab != null && lookAtTarget != null)
        {
            Vector3 position1 = cube1.transform.position;
            Vector3 position2 = cube2.transform.position;

            // Calculate the center point between the two cubes
            Vector3 centerPosition = (position1 + position2) / 2;

            // Calculate the distance between the two cubes
            float distance = Vector3.Distance(position1, position2);

            // Instantiate the panel
            GameObject panelInstance = Instantiate(panelPrefab, centerPosition, Quaternion.identity);

            // Calculate the scale needed to stretch the panel between the cubes
            Vector3 newScale = panelInstance.transform.localScale;
            newScale.x = distance;
            panelInstance.transform.localScale = newScale;

            // Make the panel look at the sphere
            panelInstance.transform.LookAt(lookAtTarget.transform.position);

            // Correct the orientation so it faces the right direction
            panelInstance.transform.Rotate(0, 0, 0);
        }
        else
        {
            Debug.LogError("Cubes, panel prefab, or look-at target are not assigned.");
        }
    }
}
