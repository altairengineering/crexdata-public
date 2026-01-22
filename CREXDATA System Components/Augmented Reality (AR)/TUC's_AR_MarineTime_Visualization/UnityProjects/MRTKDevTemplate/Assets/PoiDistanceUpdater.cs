using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UIElements;

public class PoiDistanceUpdater : MonoBehaviour
{
    private Transform _userTransform;
    public TextMeshProUGUI poiDistanceText;
    public float refresh_rate = 1;
    public static float POIScale = 120f; // Factor to adjust the perceived size of the object

    public static float POIHeight = 25f;
    // Start is called before the first frame update
    public SimplePOIInformer infoPanelController;

    void Start()
    {
        if (infoPanelController == null)
        {
            infoPanelController = FindObjectOfType<SimplePOIInformer>();
            if (infoPanelController == null)
            {
                Debug.LogWarning("SimplePOIInformer not found in the scene.");
                return;
            }
        }
        _userTransform = Camera.main.transform;
        try
        {
            StartCoroutine(UpdateDistance());
        }
        catch (System.Exception)
        {
            Debug.LogWarning("InfoPanelController not found");
            throw;
        }

        InvokeRepeating("MaintainSize", 0f, 5f); // Call MaintainSize every 0.1 seconds
    }

    IEnumerator UpdateDistance()
    {
        while (true)
        {
            Vector3 objectPositionXZ = new Vector3(transform.position.x, 0, transform.position.z);
            Vector3 userPositionXZ = new Vector3(_userTransform.position.x, 0, _userTransform.position.z);
            float distance = Vector3.Distance(objectPositionXZ, userPositionXZ);
            if (poiDistanceText != null)
                poiDistanceText.text = $"{distance:F2} m";
            yield return new WaitForSeconds(refresh_rate); // Wait for 1 second
        }
    }

    //i will attach this scrip to the POI object and i want a funtion to maintain the same percetive size no mater how far the camera is from the object
    public void MaintainSize()
    {
        if (Camera.main != null)
        {
            float distance = Vector3.Distance(Camera.main.transform.position, transform.position);
            float scaleFactor = distance / POIScale; // Adjust the divisor to change the perceived size
            transform.localScale = new Vector3(scaleFactor, scaleFactor, scaleFactor);

            //adjust also the y possition of the object, if it will be infront of me it will be in the same height as the camera and as it gets further away it will be higher
            Vector3 cameraPosition = Camera.main.transform.position;
            Vector3 newPosition = transform.position;
            newPosition.y = cameraPosition.y + (distance / POIHeight); // Adjust the height based on distance
            transform.position = newPosition;

        }
    }

    public void PoiInfoPanel()
    {
        if (infoPanelController != null)
        {
            infoPanelController.ShowInfoPanel();
            string poiName = gameObject.name; // Get the name of the POI GameObject

            Vector3 worldCoordinates = transform.position;

            string infoText = $"<b>{poiName}</b>\n";
            infoText += $"Distance: {poiDistanceText.text}\n";
            infoText += $"Coordinates: {worldCoordinates.x:F2}, {worldCoordinates.y:F2}, {worldCoordinates.z:F2}";

            infoPanelController.SetInfoText("POI Info", infoText);
        }
        else
        {
            Debug.LogWarning("InfoPanelController is not set.");
        }
    }
}
