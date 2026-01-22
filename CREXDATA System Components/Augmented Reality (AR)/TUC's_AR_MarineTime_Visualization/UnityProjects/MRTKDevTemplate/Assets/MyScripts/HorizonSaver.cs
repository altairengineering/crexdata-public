using UnityEngine;
using UnityEngine.UI;

public class HorizonSaver : MonoBehaviour
{
    public Transform horizonLine; // Reference to the horizon line cube transform

    public void SaveHorizonLevel()
    {
        // Save the Y position of the horizon line
        float horizonLevel = horizonLine.position.y;
        PlayerPrefs.SetFloat("HorizonLevel", horizonLevel);
        PlayerPrefs.Save();

        // Notify the user
        Debug.Log("Horizon level saved: " + horizonLevel);
    }
}
