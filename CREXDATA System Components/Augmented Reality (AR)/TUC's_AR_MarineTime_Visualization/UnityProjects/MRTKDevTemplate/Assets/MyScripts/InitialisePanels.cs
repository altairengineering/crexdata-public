using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InitialisePanels : MonoBehaviour
{
    public GameObject panelPrefab;

    private GameObject[] panels = new GameObject[4];
    private float horizonLevel;

    public void PlacePanelsAroundUser()
    {
        // Load the saved horizon level
        horizonLevel = PlayerPrefs.GetFloat("HorizonLevel", 1.5f); // Default to 1.5 if not set

        // Calculate the panel height
        float panelHeight = panelPrefab.GetComponent<Renderer>().bounds.size.y;

        // Calculate the required position to place the horizon level at 2/3 of the panel height
        float yOffset = (2.0f / 3.0f) * panelHeight - horizonLevel;

        Vector3 userPosition = Camera.main.transform.position;
        Vector3 userForward = Camera.main.transform.forward;
        Vector3 userRight = Camera.main.transform.right;

        float distance = 1.0f; // Distance from the user

        // Calculate positions
        Vector3[] positions = new Vector3[4];
        positions[0] = new Vector3(userPosition.x + userForward.x * distance, horizonLevel + yOffset, userPosition.z + userForward.z * distance);
        positions[1] = new Vector3(userPosition.x - userForward.x * distance, horizonLevel + yOffset, userPosition.z - userForward.z * distance);
        positions[2] = new Vector3(userPosition.x + userRight.x * distance, horizonLevel + yOffset, userPosition.z + userRight.z * distance);
        positions[3] = new Vector3(userPosition.x - userRight.x * distance, horizonLevel + yOffset, userPosition.z - userRight.z * distance);

        // Instantiate or move panels
        for (int i = 0; i < 4; i++)
        {
            if (panels[i] == null)
            {
                panels[i] = Instantiate(panelPrefab, positions[i], Quaternion.identity);
            }
            else
            {
                panels[i].transform.position = positions[i];
            }

            // Make panels face the user
            panels[i].transform.LookAt(new Vector3(userPosition.x, panels[i].transform.position.y, userPosition.z));
            panels[i].transform.Rotate(0, 180, 0); // Rotate 180 degrees to face the user
        }
    }
}
