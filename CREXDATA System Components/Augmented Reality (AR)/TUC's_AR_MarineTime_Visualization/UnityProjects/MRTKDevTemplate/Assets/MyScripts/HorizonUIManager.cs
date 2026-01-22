using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

public class HorizonManager : MonoBehaviour
{
    public float userHeight = 1.75f; // Default height in meters
    public float skyBandHeight = 5.0f;
    public float horizonBandHeight = 1.0f;
    public float waterSurfaceHeight = 5.0f;
    public float bandWidthMultiplier = 5.0f; // Multiplier for the width
    public Material transparentMaterial; // Assign a transparent material in the inspector

    void Start()
    {
        InputTracking.Recenter();
        Vector3 horizonPosition = CalculateHorizonPosition(userHeight);
        SetupUIAreas(horizonPosition);
    }

    void Update()
    {
        Vector3 horizonPosition = CalculateHorizonPosition(userHeight);
        SetupUIAreas(horizonPosition);
    }

    float CalculateHorizonDistance(float height)
    {
        float earthRadius = 6371000f; // Radius of the Earth in meters
        return Mathf.Sqrt(2 * earthRadius * height);
    }

    float CalculateHorizonAngle(float height)
    {
        float earthRadius = 6371000f; // Radius of the Earth in meters
        return Mathf.Acos(earthRadius / (earthRadius + height));
    }

    Vector3 CalculateHorizonPosition(float height)
    {
        float horizonDistance = CalculateHorizonDistance(height);
        float horizonAngle = CalculateHorizonAngle(height);
        float horizonYPosition = Mathf.Tan(horizonAngle) * horizonDistance;
        return new Vector3(0, horizonYPosition, horizonDistance);
    }

    void SetupUIAreas(Vector3 horizonPosition)
    {
        Vector3 skyBandPosition = horizonPosition + new Vector3(0, skyBandHeight / 2 + horizonBandHeight / 2, 0);
        CreateUIBand("SkyBand", skyBandPosition, skyBandHeight);

        Vector3 horizonBandPosition = horizonPosition;
        CreateUIBand("HorizonBand", horizonBandPosition, horizonBandHeight);

        Vector3 waterSurfacePosition = horizonPosition - new Vector3(0, waterSurfaceHeight / 2 + horizonBandHeight / 2, 0);
        CreateUIBand("WaterSurface", waterSurfacePosition, waterSurfaceHeight);
    }

    void CreateUIBand(string name, Vector3 position, float height)
    {
        GameObject band = GameObject.Find(name);
        if (band == null)
        {
            band = new GameObject(name);
            MeshRenderer renderer = band.AddComponent<MeshRenderer>();
            renderer.material = transparentMaterial;
            MeshFilter filter = band.AddComponent<MeshFilter>();
            filter.mesh = CreateQuadMesh(height, bandWidthMultiplier);

            // Add a transparent material and ensure it doesn't block raycasts
            MeshCollider collider = band.AddComponent<MeshCollider>();
            collider.convex = true;
            collider.isTrigger = true; // Ensure it doesn't block clicks
        }
        band.transform.position = position;
    }

    Mesh CreateQuadMesh(float height, float widthMultiplier)
    {
        float width = 2 * widthMultiplier; // Make the quad 5 times wider
        Mesh mesh = new Mesh();
        mesh.vertices = new Vector3[] {
            new Vector3(-width / 2, -height / 2, 0),
            new Vector3(width / 2, -height / 2, 0),
            new Vector3(width / 2, height / 2, 0),
            new Vector3(-width / 2, height / 2, 0)
        };
        mesh.uv = new Vector2[] {
            new Vector2(0, 0),
            new Vector2(1, 0),
            new Vector2(1, 1),
            new Vector2(0, 1)
        };
        mesh.triangles = new int[] { 0, 1, 2, 2, 3, 0 };
        mesh.RecalculateNormals();
        return mesh;
    }
}
