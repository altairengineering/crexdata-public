using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Mapbox.Unity.Map;
using Mapbox.Utils;

public class MapInitializer : MonoBehaviour
{
    public AbstractMap map;
    [SerializeField] private GameObject button;

    public void InitializeMapAtLocation()
    {
        Vector2d latLong = new Vector2d(47.26886867779381, 11.390172374646944);
        map.SetCenterLatitudeLongitude(latLong);
        int zoomLevel = (int)map.Zoom;
        map.Initialize(latLong, zoomLevel);
        button.gameObject.SetActive(false);
    }
}
