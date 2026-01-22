using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEngine.XR.ARFoundation;

public class WaterLevelIndicator : MonoBehaviour
{
    public GameObject waterLevelIndicator; // Assign in inspector
    public float waterLevelHeight = 1.0f; // Example height in meters

    private void Start()
    {
        // Example: Directly position the water level indicator on start.
        // In practice, you'd adjust this based on detected wall positions.
        PositionWaterLevelIndicator();
    }

    void PositionWaterLevelIndicator()
    {
        if (!waterLevelIndicator) return;

        // Assuming we've detected a wall and know its bounds, we'll just simulate it here.
        Vector3 wallStartPoint = new Vector3(-5, waterLevelHeight, 10);
        Vector3 wallEndPoint = new Vector3(5, waterLevelHeight, 10);

        LineRenderer lineRenderer = waterLevelIndicator.GetComponent<LineRenderer>();
        if (lineRenderer)
        {
            lineRenderer.positionCount = 2;
            lineRenderer.SetPosition(0, wallStartPoint);
            lineRenderer.SetPosition(1, wallEndPoint);
        }
    }

    // Placeholder for wall detection logic

    void DetectWalls()
    {
        // Get the ARMeshManager
        ARMeshManager meshManager = FindObjectOfType<ARMeshManager>();

        if (meshManager == null)
        {
            Debug.LogError("Failed to get ARMeshManager");
            return;
        }

        // Get the latest spatial mesh data
        var spatialMeshData = meshManager.meshes;

        // Find the lowest point in the spatial mesh data
        float lowestPoint = float.PositiveInfinity;
        foreach (var mesh in spatialMeshData)
        {
            MeshFilter meshFilter = mesh.GetComponent<MeshFilter>();
            if (meshFilter != null)
            {
                foreach (var vertex in meshFilter.sharedMesh.vertices)
                {
                    if (vertex.y < lowestPoint)
                    {
                        lowestPoint = vertex.y;
                    }
                }
            }
        }

        // Calculate the height of the HoloLens from the ground
        float height = Camera.main.transform.position.y - lowestPoint;

        // Once you have the wall positions and the height, call PositionWaterLevelIndicator() to update the indicator
        PositionWaterLevelIndicator();
    }
}
