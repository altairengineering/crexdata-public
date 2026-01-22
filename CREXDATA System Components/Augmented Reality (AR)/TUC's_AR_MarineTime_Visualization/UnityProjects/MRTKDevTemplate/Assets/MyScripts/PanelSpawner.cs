using UnityEngine;

public class PanelSpawner : MonoBehaviour
{
    public GameObject[] panels; // References to the existing panels
    public float distanceFromUser = 2.0f; // Distance in front of the user
    private int currentPanelIndex = 0; // Index to keep track of the current panel to be activated
    void Start()
    {
        //deactiva
        foreach (GameObject panel in panels)
        {
            Debug.Log("Stopped scaling Y.");
            panel.SetActive(false);
            Debug.LogError("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy");
        }
    }

        public void SpawnPanel()
    {
        if (panels == null || panels.Length == 0)
        {
            Debug.LogError("Panels are not assigned or the array is empty.");
            return;
        }

        // Get the user's position and orientation
        Vector3 userPosition = Camera.main.transform.position;
        Vector3 userForward = Camera.main.transform.forward;

        // Calculate the spawn position in front of the user
        Vector3 spawnPosition = userPosition + userForward * distanceFromUser;

        // Get the next panel to activate and move
        GameObject panel = panels[currentPanelIndex];

        // Activate the panel if it's not already active
        if (!panel.activeSelf)
        {
            panel.SetActive(true);
        }

        // Move the panel to the calculated position
        panel.transform.position = spawnPosition;

        // Optionally, face the panel towards the user
        panel.transform.LookAt(userPosition);
        panel.transform.Rotate(0, 180, 0); // Adjust rotation if necessary

        // Update the index for the next panel
        currentPanelIndex = (currentPanelIndex + 1) % panels.Length;
    }
}
