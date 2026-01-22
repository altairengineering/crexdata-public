using System.Collections.Generic;
using UnityEngine;

public class GPSConverter : MonoBehaviour
{
    public double myLatitude = 0.0;
    public double myLongitude = 0.0;
    public List<Vector2> objectCoordinates = new List<Vector2>();
    public GameObject objectPrefab;

    private List<GameObject> spawnedObjects = new List<GameObject>();
    private LineRenderer lineRenderer;
    private int lastCoordinateCount;

    void Start()
    {
        lineRenderer = GetComponent<LineRenderer>();
        lastCoordinateCount = objectCoordinates.Count;
        SpawnObjects();
    }

    void Update()
    {
        if (objectCoordinates.Count != lastCoordinateCount)
        {
            // Update line renderer position count
            lineRenderer.positionCount = objectCoordinates.Count;

            for (int i = lastCoordinateCount; i < objectCoordinates.Count; i++)
            {
                Vector2 newCoordinate = objectCoordinates[i];
                Vector3 newObjectPosition = CalculateObjectLocalPosition(newCoordinate.x, newCoordinate.y);
                GameObject newSpawnedObject = Instantiate(objectPrefab, newObjectPosition, Quaternion.identity);
                spawnedObjects.Add(newSpawnedObject);
            }

            lastCoordinateCount = objectCoordinates.Count;  // Update the count
        }

        // Always update line positions to reflect current object positions
        UpdateLineRendererPositions();
    }

    private void SpawnObjects()
    {
        foreach (Vector2 coordinate in objectCoordinates)
        {
            Vector3 objectPosition = CalculateObjectLocalPosition(coordinate.x, coordinate.y);
            GameObject spawnedObject = Instantiate(objectPrefab, objectPosition, Quaternion.identity);
            spawnedObjects.Add(spawnedObject);
        }
        // Set initial line positions
        lineRenderer.positionCount = objectCoordinates.Count;
        UpdateLineRendererPositions();
    }

    private void UpdateLineRendererPositions()
    {
        for (int i = 0; i < spawnedObjects.Count; i++)
        {
            if (i < lineRenderer.positionCount)  // Ensure the index is within bounds
            {
                lineRenderer.SetPosition(i, spawnedObjects[i].transform.position);
            }
        }
    }

    Vector3 CalculateObjectLocalPosition(double objectLatitude, double objectLongitude)
    {
        double latOffset = (objectLatitude - myLatitude) * 111000.0;
        double lonOffset = (objectLongitude - myLongitude) * (111000.0 * Mathf.Cos((float)(myLatitude * Mathf.PI / 180.0)));
        return new Vector3((float)lonOffset, 0, (float)latOffset);
    }
}
