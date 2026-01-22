using System.Collections.Generic;
using UnityEngine;
using Mapbox.Unity.Map;
using Mapbox.Utils;
using MixedReality.Toolkit.UX;

public class MapAndPlayerManager : MonoBehaviour
{
    public GameObject MapGameobject;
    public AbstractMap map;
    public GameObject Player;
    [SerializeField] private GameObject button;
    public PressableButton toggleButton;

    [SerializeField]
    private Vector2d coordinates;

    public Dictionary<GameObject, Vector2d> spawnedPOIs = new();

    public Dictionary<GameObject, List<Vector2d>> spawnedAreas = new Dictionary<GameObject, List<Vector2d>>();



    // New LineRenderers that won't get wider
    public LineRenderer myRouteMapLineRenderer; // Add this in the inspector
    public LineRenderer otherRouteMapLineRenderer; // Add this in the inspector

    // Existing LineRenderers that will be renamed
    public LineRenderer myRouteUncertainMapLineRenderer; // Rename the existing one in the inspector
    public LineRenderer otherRouteUncertainMapLineRenderer;

    public float myRouteUncertainStartWidth = 0.001f;
    public float myRouteUncertainEndWidth = 0.01f;
    public float otherRouteUncertainStartWidth = 0.001f;
    public float otherRouteUncertainEndWidth = 0.01f;

    // Public variables for certain line renderers
    public float myRouteLineWidth = 0.001f;
    public float otherRouteLineWidth = 0.001f;

    private bool mapInitialized = false;

    public void Start()
    {
        //Used for manually providing user coordinates if not connected to the server
        //UpdateMapAndPlayer(new Vector2d(51.54874817192535, 7.373871377682607));
    }

    private void InitializeMapAtLocation()
    {
        map.SetCenterLatitudeLongitude(coordinates);
        int zoomLevel = (int)map.Zoom;
        map.Initialize(coordinates, zoomLevel);
        mapInitialized = true;
    }

    public void UpdatePlayerLocation(Vector2d coordinates)
    {
        if (!mapInitialized) return;

        try
        {
            if (MapGameobject.activeSelf)
            {
                map.SetCenterLatitudeLongitude(coordinates);
                map.UpdateMap();
                UpdateSpawnedPOIs();
                Player.transform.position = map.GeoToWorldPosition(coordinates);
                Debug.Log("Player location:" + Player.transform.position);
            }
        }
        catch (System.NullReferenceException)
        {
            Debug.LogWarning("Map is not initialized yet");
        }
    }

    public void UpdateSpawnedPOIs()
    {
        if (!mapInitialized || !MapGameobject.activeSelf) return;

        foreach (KeyValuePair<GameObject, Vector2d> poi in spawnedPOIs)
        {
            poi.Key.transform.position = map.GeoToWorldPosition(poi.Value);
            poi.Key.transform.position = new Vector3(poi.Key.transform.position.x, poi.Key.transform.position.y + 0.01f, poi.Key.transform.position.z);
        }
        // Update polygons, too
        UpdateSpawnedAreas();
    }

    public void UpdateSpawnedAreas()
    {
        if (!mapInitialized || !MapGameobject.activeSelf)
            return;

        foreach (var kvp in spawnedAreas)
        {
            GameObject areaObj = kvp.Key;
            List<Vector2d> latLonVertices = kvp.Value;

            MeshFilter meshFilter = areaObj.GetComponent<MeshFilter>();
            if (meshFilter == null)
                continue;

            // We'll need the parent's transform
            Transform parentT = areaObj.transform.parent;
            if (!parentT)
                continue; // Just in case

            Vector3[] vertices = new Vector3[latLonVertices.Count];
            for (int i = 0; i < latLonVertices.Count; i++)
            {
                // 1) Get the world position from Mapbox
                Vector3 worldPos = map.GeoToWorldPosition(latLonVertices[i]);
                worldPos.y += 0.01f;  // small lift above the map

                // 2) Convert it to local space relative to the mapGameObject
                Vector3 localPos = parentT.InverseTransformPoint(worldPos);

                // 3) Store in the mesh array
                vertices[i] = localPos;
            }

            // Triangulate, assign to mesh, etc.
            int[] triangles = Triangulate(vertices.Length);

            Mesh mesh = meshFilter.sharedMesh;
            if (mesh == null)
            {
                mesh = new Mesh();
                meshFilter.sharedMesh = mesh;
            }
            else
            {
                mesh.Clear();
            }

            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
        }
    }


    // Simple fan triangulation for convex polygons
    private int[] Triangulate(int vertexCount)
    {
        List<int> triangles = new List<int>();
        for (int i = 1; i < vertexCount - 1; i++)
        {
            triangles.Add(0);
            triangles.Add(i);
            triangles.Add(i + 1);
        }
        return triangles.ToArray();
    }



    public void SetCoordinates(Vector2d coordinates)
    {
        this.coordinates = coordinates;
    }

    public void UpdateMapAndPlayer(Vector2d coordinates)
    {
        this.coordinates = coordinates;
        UpdatePlayerLocation(coordinates);
    }

    // Function to toggle and untoggle the visibility of the map gameobject with the button
    public void ToggleMapVisibility()
    {
        MapGameobject.SetActive(!MapGameobject.activeSelf);
        if (MapGameobject.activeSelf)
        {
            if (!mapInitialized)
            {
                InitializeMapAtLocation();
            }
            UpdateSpawnedPOIs();
            ConnectPOIs();
        }
    }


    public void test()
    {
        UpdatePlayerLocation(new Vector2d(51.54874817192535, 7.373871377682607));

    }

    public void ConnectPOIs()
    {
        if (spawnedPOIs.Count < 2) return;

        UpdateLineRenderers();
    }

    private void Update()
    {
        // Continuously update the line renderers as points move
        UpdateLineRenderers();
    }

    private void UpdateLineRenderers()
    {
        // Create lists to hold positions for each type of prefab
        List<Vector3> myRoutePositions = new List<Vector3>();
        List<Vector3> otherRoutePositions = new List<Vector3>();

        // Create lists to hold the corresponding GameObjects for each type of prefab
        List<GameObject> myRouteObjects = new List<GameObject>();
        List<GameObject> otherRouteObjects = new List<GameObject>();

        // Add the player's position to the list of positions
        myRoutePositions.Add(Player.transform.position);
        myRouteObjects.Add(Player);

        // Iterate through the dictionary and filter based on prefab type
        foreach (KeyValuePair<GameObject, Vector2d> poi in spawnedPOIs)
        {
            GameObject currentPOI = poi.Key;

            // Check if the current GameObject is of type MyRoutePointMapPrefab
            if (currentPOI.name == "MyArrow(Clone)")
            {
                myRoutePositions.Add(currentPOI.transform.position);
                myRouteObjects.Add(currentPOI);
            }
            // Check if the current GameObject is of type OtherRoutePointMapPrefab
            else if (currentPOI.name == "OtherArrow(Clone)")
            {
                otherRoutePositions.Add(currentPOI.transform.position);
                otherRouteObjects.Add(currentPOI);
            }
            // Check if the current GameObject is of type OtherShipMapPrefab and place it first
            else if (currentPOI.name == "OtherShipMap(Clone)")
            {
                // Insert at the beginning of the list, pushing other elements down
                otherRoutePositions.Insert(0, currentPOI.transform.position);
                otherRouteObjects.Insert(0, currentPOI);
            }

        }

        // Update the position count for each LineRenderer
        myRouteMapLineRenderer.positionCount = myRoutePositions.Count;
        otherRouteMapLineRenderer.positionCount = otherRoutePositions.Count;
        myRouteUncertainMapLineRenderer.positionCount = myRoutePositions.Count;
        otherRouteUncertainMapLineRenderer.positionCount = otherRoutePositions.Count;

        // Update the positions in the LineRenderers
        myRouteMapLineRenderer.SetPositions(myRoutePositions.ToArray());
        otherRouteMapLineRenderer.SetPositions(otherRoutePositions.ToArray());

        // Offset the positions for the uncertain line renderers
        List<Vector3> myRouteUncertainPositions = OffsetPositions(myRoutePositions, -0.001f); // Adjust -0.1f as needed
        List<Vector3> otherRouteUncertainPositions = OffsetPositions(otherRoutePositions, -0.001f); // Adjust -0.1f as needed

        myRouteUncertainMapLineRenderer.SetPositions(myRouteUncertainPositions.ToArray());
        otherRouteUncertainMapLineRenderer.SetPositions(otherRouteUncertainPositions.ToArray());

        // Make the points face each other
        OrientPointsTowardsNext(myRouteObjects);
        OrientPointsTowardsNext(otherRouteObjects);

        // Adjust the width of the uncertain lines (these get wider)
        AdjustLineRendererWidth(myRouteUncertainMapLineRenderer, myRouteUncertainStartWidth, myRouteUncertainEndWidth);
        AdjustLineRendererWidth(otherRouteUncertainMapLineRenderer, otherRouteUncertainStartWidth, otherRouteUncertainEndWidth);

        // Set the width of the certain lines (these won't get wider)
        SetLineRendererWidth(myRouteMapLineRenderer, myRouteLineWidth);
        SetLineRendererWidth(otherRouteMapLineRenderer, otherRouteLineWidth);
    }


    private List<Vector3> OffsetPositions(List<Vector3> originalPositions, float yOffset)
    {
        List<Vector3> offsetPositions = new List<Vector3>();
        foreach (var position in originalPositions)
        {
            offsetPositions.Add(new Vector3(position.x, position.y + yOffset, position.z));
        }
        return offsetPositions;
    }

    private void OrientPointsTowardsNext(List<GameObject> pointsList)
    {
        // Make each point look at the next point in the list
        for (int i = 0; i < pointsList.Count - 1; i++)
        {
            Transform currentPoint = pointsList[i].transform;
            Transform nextPoint = pointsList[i + 1].transform;

            Vector3 direction = nextPoint.localPosition - currentPoint.localPosition;
            if (direction != Vector3.zero)
            {
                currentPoint.localRotation = Quaternion.LookRotation(direction, Vector3.up);
            }
        }

        // Handle the last object separately
        if (pointsList.Count > 1)
        {
            Transform lastPoint = pointsList[pointsList.Count - 1].transform;
            Transform secondLastPoint = pointsList[pointsList.Count - 2].transform;

            // Copy the rotation of the second last point to the last point
            lastPoint.localRotation = secondLastPoint.localRotation;
        }
    }



    private void AdjustLineRendererWidth(LineRenderer lineRenderer, float startWidth, float endWidth)
    {
        // Set the start and end width of the LineRenderer
        lineRenderer.startWidth = startWidth;
        lineRenderer.endWidth = endWidth;

        // Create a width curve to make the line progressively wider
        AnimationCurve widthCurve = new AnimationCurve();
        widthCurve.AddKey(0.0f, startWidth);
        widthCurve.AddKey(1.0f, endWidth);

        // Apply the width curve to the LineRenderer
        lineRenderer.widthCurve = widthCurve;
    }

    private void SetLineRendererWidth(LineRenderer lineRenderer, float width)
    {
        // Set the start and end width of the LineRenderer to be constant
        lineRenderer.startWidth = width;
        lineRenderer.endWidth = width;

        // Reset the width curve to a straight line
        AnimationCurve widthCurve = new AnimationCurve();
        widthCurve.AddKey(0.0f, width);
        widthCurve.AddKey(1.0f, width);

        // Apply the width curve to the LineRenderer
        lineRenderer.widthCurve = widthCurve;
    }




}
