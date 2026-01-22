using Mapbox.Utils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

public enum POIType
{
    CollisionPoint,
    Ship,
    OtherShip,
    MyRoutePoint,
    OtherRoutePoint,
    General,
    ShipPassingBy,
    RedLighthouse,
    GreenLighthouse,
    Reef,
    Wreck,
    UnknownDanger,
    MooringBuoy,
    SpecialPurposeBuoy
}


[System.Serializable]
public class GPSCoordinate
{
    [Tooltip("Input format: 'Latitude,Longitude' e.g., '34.0522,-118.2437'")]
    public string coordinates;
    public POIType POIType;

    public double Latitude
    {
        get
        {
            return double.Parse(coordinates.Split(',')[0].Trim());
        }
    }

    public double Longitude
    {
        get
        {
            return double.Parse(coordinates.Split(',')[1].Trim());
        }
    }
}

public class GPSMulti : MonoBehaviour
{
    [Header("POIs Coordinates")]
    [Tooltip("Add all object coordinates here")]
    public List<GPSCoordinate> staticCoordinates = new List<GPSCoordinate>();

    [Header("Route Coordinates")]
    [Tooltip("Add all object coordinates here")]
    public List<GPSCoordinate> RoutecCoordinates = new List<GPSCoordinate>();

    [Header("Area Coordinates")]
    public List<string> areaCoordinates;

    [Header("Area Material")]
    public Material areaMaterial; // Material to be provided through the Inspector

    public Material areaMapMaterial;

    // Horizon calibration data
    private Vector3 horizonOrigin; // The point on the horizon
    private Vector3 horizonNormal; // The normal vector of the horizon plane
    private bool isHorizonCalibrated = false;

    [Header("POI Prefabs")]
    public GameObject CollisionPointPrefab;
    public GameObject ShipPrefab;
    public GameObject MyRoutePointPrefab;
    public GameObject OtherRoutePointPrefab;
    public GameObject GeneralPrefab;
    public GameObject ShipPassingByPrefab;
    public GameObject RedLighthousePrefab;
    public GameObject GreenLighthousePrefab;
    public GameObject ReefPrefab;
    public GameObject WreckPrefab;
    public GameObject UnknownDangerPrefab;
    public GameObject MooringBuoyPrefab;
    public GameObject SpecialPurposeBuoyPrefab;



    [Header("Map Prefabs")]
    public GameObject CollisionPointMapPrefab;
    public GameObject MyShipMapPrefab;
    public GameObject OtherShipMapPrefab;
    public GameObject MyRoutePointMapPreafab;
    public GameObject OtherRoutePointMapPrefab;
    public GameObject generalPointMapPrefab;
    public GameObject ShipPassingByMapPrefab;
    public GameObject RedLighthouseMapPrefab;
    public GameObject GreenLighthouseMapPrefab;
    public GameObject ReefMapPrefab;
    public GameObject WreckMapPrefab;
    public GameObject UnknownDangerMapPrefab;
    public GameObject MooringBuoyMapPrefab;
    public GameObject SpecialPurposeBuoyMapPrefab;



    [Header("Reference Coordinate Values")]
    [SerializeField] private bool useFakeCoordinates = false;
    public string referenceCoordinates;
    public Vector3 referenceUnityPosition = Vector3.zero;
    public double referenceLatitude = 0.0;
    public double referenceLongitude = 0.0;
    public bool referenceCoordinatesSet = false;
    public GameObject mapGameObject;

    private Dictionary<POIType, GameObject> poiPrefabs;

    private Dictionary<POIType, GameObject> mapPrefabs;

    public Dictionary<POIType, List<GameObject>> spawnedObjects = new Dictionary<POIType, List<GameObject>>();

    private Camera mainCamera;

    private void Awake()
    {
        poiPrefabs = new Dictionary<POIType, GameObject>
    {
        { POIType.CollisionPoint, CollisionPointPrefab },
        { POIType.Ship, ShipPrefab },
        { POIType.OtherShip, ShipPrefab },
        { POIType.MyRoutePoint, MyRoutePointPrefab },
        { POIType.OtherRoutePoint, OtherRoutePointPrefab },
        { POIType.General, GeneralPrefab },
        { POIType.ShipPassingBy, ShipPassingByPrefab },
        { POIType.RedLighthouse, RedLighthousePrefab },
        { POIType.GreenLighthouse, GreenLighthousePrefab },
        { POIType.Reef, ReefPrefab },
        { POIType.Wreck, WreckPrefab },
        { POIType.UnknownDanger, UnknownDangerPrefab },
        { POIType.MooringBuoy, MooringBuoyPrefab },
        { POIType.SpecialPurposeBuoy, SpecialPurposeBuoyPrefab }
    };

        mapPrefabs = new Dictionary<POIType, GameObject>
    {
        { POIType.CollisionPoint, CollisionPointMapPrefab },
        { POIType.Ship, MyShipMapPrefab },
        { POIType.OtherShip, OtherShipMapPrefab },
        { POIType.MyRoutePoint, MyRoutePointMapPreafab },
        { POIType.OtherRoutePoint, OtherRoutePointMapPrefab },
        { POIType.General, generalPointMapPrefab },
        { POIType.ShipPassingBy, ShipPassingByMapPrefab },
        { POIType.RedLighthouse, RedLighthouseMapPrefab },
        { POIType.GreenLighthouse, GreenLighthouseMapPrefab },
        { POIType.Reef, ReefMapPrefab },
        { POIType.Wreck, WreckMapPrefab },
        { POIType.UnknownDanger, UnknownDangerMapPrefab },
        { POIType.MooringBuoy, MooringBuoyMapPrefab },
        { POIType.SpecialPurposeBuoy, SpecialPurposeBuoyMapPrefab }
    };

        /*         // Initialize the spawnedObjects dictionary with empty lists for all POI types
                foreach (POIType poiType in Enum.GetValues(typeof(POIType)))
                {
                    if (!spawnedObjects.ContainsKey(poiType))
                    {
                        spawnedObjects[poiType] = new List<GameObject>();
                    }
                } */
    }


    [Header("Line Renderer")]
    public LineRenderer myRouteLineRenderer;   // Assign this via the Inspector
    public LineRenderer otherRouteLineRenderer; // Assign this via the Inspector

    public LineRenderer myRouteUncertainLineRenderer;
    public LineRenderer otherRouteUncertainLineRenderer;

    // Public variables for uncertain line renderers
    public float myRouteUncertainStartWidth = 0.5f;
    public float myRouteUncertainEndWidth = 10f;
    public float otherRouteUncertainStartWidth = 0.5f;
    public float otherRouteUncertainEndWidth = 10f;

    // Public variables for offset
    public float myRouteUncertainYOffset = -0.1f;
    public float otherRouteUncertainYOffset = -0.1f;

    private void Start()
    {
        StartCoroutine(InitializeGPSMulti());
        foreach (string area in areaCoordinates)
        {
            List<GPSCoordinate> myAreaCoordinates = ParseCoordinates(area);
            // Create the real-world area in the scene
            CreateArea(myAreaCoordinates);

            // Create the equivalent polygon on the map
            CreateAreaOnMap(myAreaCoordinates);
        }

        //#if UNITY_EDITOR
        //if (useFakeCoordinates)
        //testReferenceCoordinatesUpdate(); //FOR TESTING PURPOSES ONLY
        //#endif

        mainCamera = Camera.main;
    }

    private IEnumerator InitializeGPSMulti()
    {
        // Wait until the end of the frame to ensure DataDeserialization has finished initializing
        while (!referenceCoordinatesSet)
        {
            yield return null;
        }

        // mapGameObject.GetComponentInParent<MapAndPlayerManager>().SetCoordinates(new Vector2d(referenceLatitude, referenceLongitude));

        // Create LineRenderers for the uncertain routes if not assigned
        if (myRouteUncertainLineRenderer == null)
        {
            // Create a LineRenderer for MyRouteUncertain if not assigned
            GameObject myRouteUncertainLineObj = new GameObject("MyRouteUncertainLineObject");
            myRouteUncertainLineRenderer = myRouteUncertainLineObj.AddComponent<LineRenderer>();
            myRouteUncertainLineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        }

        if (otherRouteUncertainLineRenderer == null)
        {
            // Create a LineRenderer for OtherRouteUncertain if not assigned
            GameObject otherRouteUncertainLineObj = new GameObject("OtherRouteUncertainLineObject");
            otherRouteUncertainLineRenderer = otherRouteUncertainLineObj.AddComponent<LineRenderer>();
            otherRouteUncertainLineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        }

        // Create new LineRenderers for the routes (without width change)
        if (myRouteLineRenderer == null)
        {
            // Create a LineRenderer for MyRoute if not assigned
            GameObject myRouteLineObj = new GameObject("MyRouteLineObject");
            myRouteLineRenderer = myRouteLineObj.AddComponent<LineRenderer>();
            myRouteLineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        }

        if (otherRouteLineRenderer == null)
        {
            // Create a LineRenderer for OtherRoute if not assigned
            GameObject otherRouteLineObj = new GameObject("OtherRouteLineObject");
            otherRouteLineRenderer = otherRouteLineObj.AddComponent<LineRenderer>();
            otherRouteLineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        }

        // Spawn static objects
        foreach (var coord in staticCoordinates)
        {
            SpawnObjectAtLocation(coord);
            //spawnedObjects[coord.POIType].Add(pointObject); // Store the GameObject
        }
        //Debug.Log(" Static" + spawnedObjects[POIType.General].Count);

        // Spawn route objects
        foreach (var coord in RoutecCoordinates)
        {
            SpawnObjectAtLocation(coord);
            //spawnedObjects[coord.POIType].Add(pointObject);
        }

        StartHorizonCalibration();
        AdjustOtherRoutePointsY(adjustmentAmount);
        //print("RoutecCoordinates count: " + RoutecCoordinates.Count);
        /*         if (DataDeserialization.Instance != null)
                {
                    ShipData shipData = DataDeserialization.Instance.ShipData;

                    if (shipData.ship1_traj_lon != null && shipData.ship1_traj_lat != null)
                    {
                        for (int i = 0; i < shipData.ship1_traj_lon.Length; i++)
                        {
                            GPSCoordinate coordinates = new GPSCoordinate
                            {
                                coordinates = shipData.ship1_traj_lat[i] + "," + shipData.ship1_traj_lon[i],
                                POIType = POIType.MyRoutePoint
                            };
                            GameObject pointObject = SpawnObjectAtLocation(coordinates);
                            spawnedObjects[coordinates.POIType].Add(pointObject);
                            Debug.Log("Ship 1111111: " + shipData.ship1_traj_lat[i] + shipData.ship1_traj_lon[i]);
                        }

                        // Initialize the MyRouteLineRenderer position count
                        myRouteLineRenderer.positionCount = spawnedObjects[POIType.MyRoutePoint].Count;
                        myRouteUncertainLineRenderer.positionCount = spawnedObjects[POIType.MyRoutePoint].Count;
                    }

                    if (shipData.ship2_traj_lon != null && shipData.ship2_traj_lat != null)
                    {
                        for (int i = 0; i < shipData.ship2_traj_lon.Length; i++)
                        {
                            GPSCoordinate coordinates = new GPSCoordinate
                            {
                                coordinates = shipData.ship2_traj_lat[i] + "," + shipData.ship2_traj_lon[i],
                                POIType = POIType.OtherRoutePoint
                            };
                            GameObject pointObject = SpawnObjectAtLocation(coordinates);
                            spawnedObjects[coordinates.POIType].Add(pointObject);
                            Debug.Log("Ship 222222: " + shipData.ship2_traj_lat[i] + shipData.ship2_traj_lon[i]);
                        }

                        // Initialize the OtherRouteLineRenderer position count
                        otherRouteLineRenderer.positionCount = spawnedObjects[POIType.OtherRoutePoint].Count;
                        otherRouteUncertainLineRenderer.positionCount = spawnedObjects[POIType.OtherRoutePoint].Count;
                    }
                }
                else
                {
                    Debug.LogError("DataDeserialization instance is not available.");
                } */
    }

    private void Update()
    {
        if (spawnedObjects.ContainsKey(POIType.MyRoutePoint))
        {
            int myRoutePointCount = spawnedObjects[POIType.MyRoutePoint].Count;
            if (myRoutePointCount > 0)
            {
                // Update certain line renderer positions (no width change)
                myRouteLineRenderer.positionCount = myRoutePointCount;
                for (int i = 0; i < myRoutePointCount; i++)
                {
                    if (spawnedObjects[POIType.MyRoutePoint][i] != null)
                    {
                        myRouteLineRenderer.SetPosition(i, spawnedObjects[POIType.MyRoutePoint][i].transform.position);
                    }
                }

                // Update uncertain line renderer positions (with progressive width change)
                myRouteUncertainLineRenderer.positionCount = myRoutePointCount;
                SetLineRendererWidthProgressive(myRouteUncertainLineRenderer, myRouteUncertainStartWidth, myRouteUncertainEndWidth);
                for (int i = 0; i < myRoutePointCount; i++)
                {
                    if (spawnedObjects[POIType.MyRoutePoint][i] != null)
                    {
                        Vector3 offsetPosition = spawnedObjects[POIType.MyRoutePoint][i].transform.position;
                        offsetPosition.y += myRouteUncertainYOffset;
                        myRouteUncertainLineRenderer.SetPosition(i, offsetPosition);
                    }
                }
            }
            AdjustDirectionsForPoints(POIType.MyRoutePoint);
        }
        if (spawnedObjects.ContainsKey(POIType.OtherRoutePoint))
        {
            int otherRoutePointCount = spawnedObjects[POIType.OtherRoutePoint].Count;
            if (otherRoutePointCount > 0)
            {
                // Update certain line renderer positions (no width change)
                otherRouteLineRenderer.positionCount = otherRoutePointCount;
                for (int i = 0; i < otherRoutePointCount; i++)
                {
                    if (spawnedObjects[POIType.OtherRoutePoint][i] != null)
                    {
                        otherRouteLineRenderer.SetPosition(i, spawnedObjects[POIType.OtherRoutePoint][i].transform.position);
                    }
                }

                // Update uncertain line renderer positions (with progressive width change)
                otherRouteUncertainLineRenderer.positionCount = otherRoutePointCount;
                SetLineRendererWidthProgressive(otherRouteUncertainLineRenderer, otherRouteUncertainStartWidth, otherRouteUncertainEndWidth);
                for (int i = 0; i < otherRoutePointCount; i++)
                {
                    if (spawnedObjects[POIType.OtherRoutePoint][i] != null)
                    {
                        Vector3 offsetPosition = spawnedObjects[POIType.OtherRoutePoint][i].transform.position;
                        offsetPosition.y += otherRouteUncertainYOffset;
                        otherRouteUncertainLineRenderer.SetPosition(i, offsetPosition);
                    }
                }
            }
            AdjustDirectionsForPoints(POIType.OtherRoutePoint);
        }
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

    private void SetLineRendererWidthProgressive(LineRenderer lineRenderer, float startWidth, float endWidth)
    {
        // Set the start and end width of the LineRenderer to different values for a progressive effect
        lineRenderer.startWidth = startWidth;
        lineRenderer.endWidth = endWidth;

        // Create a width curve to make the line progressively wider
        AnimationCurve widthCurve = new AnimationCurve();
        widthCurve.AddKey(0.0f, startWidth);
        widthCurve.AddKey(1.0f, endWidth);

        // Apply the width curve to the LineRenderer
        lineRenderer.widthCurve = widthCurve;
    }

    public void UpdateReferenceCoordinates(double latitude, double longitude)
    {
        referenceLatitude = latitude;
        referenceLongitude = longitude;
        referenceCoordinatesSet = true;
    }

    public void SpawnPOIForMap(GPSCoordinate coord)
    {
        if (referenceCoordinatesSet)
        {
            try
            {
                GameObject prefab = mapPrefabs[coord.POIType];
                //spawn the object as child of mapGameObject
                GameObject createdObj = Instantiate(prefab, mapGameObject.transform);
                // Set created object's position to 0,0.1,0
                createdObj.transform.localPosition = new Vector3(0, 0, 0);
                // Add to mapObjects list
                mapGameObject.GetComponentInParent<MapAndPlayerManager>().spawnedPOIs.Add(createdObj, new Vector2d(coord.Latitude, coord.Longitude));
                mapGameObject.GetComponentInParent<MapAndPlayerManager>().UpdateSpawnedPOIs();
                // Track instances for MyRoutePointMapPrefab and OtherRoutePointMapPrefab
            }
            catch (Exception ex)
            {
                Debug.LogError("Failed to spawn map point object: " + ex.ToString());
            }
        }
        else
        {
            Debug.LogError("Reference coordinates not set");
        }


    }


    public GameObject SpawnObjectAtLocation(GPSCoordinate coord)
    {
        Vector3 position = CalculateObjectLocalPosition(coord.Latitude, coord.Longitude);
        float distance = GetFlatDistanceOfTwoPoints(position, Camera.main.transform.position);
        Debug.Log("Distance: " + distance);
        GameObject prefab = poiPrefabs[coord.POIType];
        GameObject createdObj = Instantiate(prefab, position, Quaternion.identity);

        if (!spawnedObjects.ContainsKey(coord.POIType))
        {
            spawnedObjects[coord.POIType] = new List<GameObject>();
        }
        spawnedObjects[coord.POIType].Add(createdObj);
        SpawnPOIForMap(coord);

        return createdObj;
    }

    float GetFlatDistanceOfTwoPoints(Vector3 point1, Vector3 point2)
    {
        Vector3 difference = point2 - point1;
        difference.y = 0; // Ignore the Y-axis
        return difference.magnitude;
    }
    // POI types you want to adjust
    private readonly HashSet<POIType> poiTypesToAdjust = new HashSet<POIType>
    {
        POIType.CollisionPoint,
        POIType.OtherShip,
        POIType.General,
        POIType.ShipPassingBy,
        POIType.RedLighthouse,
        POIType.GreenLighthouse,
        POIType.Reef,
        POIType.Wreck,
        POIType.UnknownDanger,
        POIType.MooringBuoy,
        POIType.SpecialPurposeBuoy
    };

    private GameObject furthestPOI;
    private float furthestDistance;
    private Vector3 userPos;
    private float userY;
    public GameObject HorizonLine;
    public float otherPOIsStartY = -30f; // starting Y for non-adjusted POIs

    // Reference distances (adjust if needed)
    public float userReferenceDistance = 1f;      // Distance at user (starting point)
    public float furthestReferenceDistance = 250f; // Distance at farthest POI

    // Horizon line scales at these distances
    public Vector3 horizonLineScaleAtUser = new Vector3(10000, 5, 1);
    public Vector3 horizonLineScaleAtFurthest = new Vector3(185430, 235.735f, 1);

    // POI scales at these distances
    public Vector3 poiScaleAtUser = new Vector3(0.1f, 0.1f, 0.1f);
    public Vector3 poiScaleAtFurthest = new Vector3(1.8f, 1.8f, 1.8f);

    public void ScaleObjectByDistance(Transform obj, float distance, Vector3 scaleAtUser, Vector3 scaleAtFurthest)
    {
        float t = (distance - userReferenceDistance) / (furthestReferenceDistance - userReferenceDistance);
        t = Mathf.Max(0, t); // Prevents scale shrinking below your "close" value
        obj.localScale = Vector3.LerpUnclamped(scaleAtUser, scaleAtFurthest, t);
    }


    public void UpdateHorizonLineScale(Vector3 horizonLinePosition)
    {
        float distance = Vector3.Distance(Camera.main.transform.position, horizonLinePosition);

        // Use the proportional scaling with LerpUnclamped, just like with the POIs
        float t = (distance - userReferenceDistance) / (furthestReferenceDistance - userReferenceDistance);
        t = Mathf.Max(0, t);
        HorizonLine.transform.localScale = Vector3.LerpUnclamped(horizonLineScaleAtUser, horizonLineScaleAtFurthest, t);
    }

    /// <summary>
    /// Scales a line object horizontally to match the field of view at its current distance
    /// </summary>
    /// <param name="lineObject">The line object to scale</param>
    /// <param name="fieldOfViewWidthDegrees">HoloLens horizontal FOV (approximately 35 degrees)</param>
    /// <param name="originalWidth">Original width of the line (optional)</param>
    /// <param name="widthMultiplier">Optional multiplier to adjust the calculated width</param>
    public void ScaleLineToMatchFieldOfView(
        GameObject lineObject,
        float fieldOfViewWidthDegrees = 52f,
        float? originalWidth = null,
        float widthMultiplier = 1.0f)
    {
        // Get reference to the camera
        Camera viewCamera = Camera.main;

        if (viewCamera == null || lineObject == null)
            return;

        // Get distance from the camera to the line object
        float distance = Vector3.Distance(viewCamera.transform.position, lineObject.transform.position);

        // Calculate width needed at this distance to fill the field of view
        float width = 2f * Mathf.Tan(fieldOfViewWidthDegrees * 0.5f * Mathf.Deg2Rad) * distance;

        // Apply width multiplier
        width *= widthMultiplier;

        // Get original scale
        Vector3 currentScale = lineObject.transform.localScale;
        float actualOriginalWidth = originalWidth.HasValue ? originalWidth.Value : currentScale.x;

        // Calculate new scale for X dimension only
        float scaleFactor = width / actualOriginalWidth;
        Vector3 newScale = currentScale;
        newScale.x = actualOriginalWidth * scaleFactor;

        // Apply the new scale
        lineObject.transform.localScale = newScale;
    }


    public void UpdateOtherPOIScales()
    {
        Vector3 userPosition = Camera.main.transform.position;

        foreach (var poi in spawnedObjects[POIType.MyRoutePoint])
        {
            // Scale MyRoutePoints based on distance from user
            float distance = Vector3.Distance(userPosition, poi.transform.position);
            ScaleObjectByDistance(poi.transform, distance, poiScaleAtUser, poiScaleAtFurthest);
        }

        /*  foreach (var poi in spawnedObjects[POIType.OtherRoutePoint])
         {
             // Scale OtherRoutePoints based on distance from user
             float distance = Vector3.Distance(userPosition, poi.transform.position);
             ScaleObjectByDistance(poi.transform, distance, poiScaleAtUser, poiScaleAtFurthest);
         } */

    }

    //function that will chage the y possition of the otheroutepoints by a certain amount. find the average distance of the pois and for each meter reduce the y by 0.1f 
    public float adjustmentAmount = 0.04f;
    public void AdjustOtherRoutePointsY(float adjustmentAmount)
    {
        if (!spawnedObjects.ContainsKey(POIType.OtherRoutePoint))
        {
            Debug.LogWarning("No OtherRoutePoints found to adjust.");
            return;
        }

        Vector3 userPosition = Camera.main.transform.position;
        float totalDistance = 0f;
        int poiCount = spawnedObjects[POIType.OtherRoutePoint].Count;

        // Calculate total distance from user to each OtherRoutePoint
        foreach (var poi in spawnedObjects[POIType.OtherRoutePoint])
        {
            float distance = Vector3.Distance(userPosition, poi.transform.position);
            ScaleObjectByDistance(poi.transform, distance, poiScaleAtUser, poiScaleAtFurthest);
            totalDistance += distance;
        }

        // Calculate average distance
        float averageDistance = poiCount > 0 ? totalDistance / poiCount : 0f;

        // Adjust Y position of each OtherRoutePoint
        foreach (var poi in spawnedObjects[POIType.OtherRoutePoint])
        {
            Vector3 poiPosition = poi.transform.position;
            poiPosition.y -= averageDistance * adjustmentAmount; // Adjust Y by average distance * adjustment amount
            poi.transform.position = poiPosition;
        }

        Debug.Log($"Adjusted {poiCount} OtherRoutePoints by {adjustmentAmount} per meter based on average distance of {averageDistance}.");
    }

    // === STATE 1: Find furthest point (from ALL POIs), move line there ===
    public void StartHorizonCalibration()
    {
        userPos = Camera.main.transform.position;
        userY = userPos.y;
        furthestDistance = -1f;
        furthestPOI = null;

        // Look for the furthest POI (regardless of type)
        if (spawnedObjects.ContainsKey(POIType.MyRoutePoint))
        {
            print("Checking MyRoutePoint for furthest distance...");
            foreach (var poi in spawnedObjects[POIType.MyRoutePoint])
            {
                print("Checking MyRoutePoint: " + poi.name);
                Vector2 userXZ = new Vector2(userPos.x, userPos.z);
                Vector2 poiXZ = new Vector2(poi.transform.position.x, poi.transform.position.z);
                float dist = Vector2.Distance(userXZ, poiXZ);
                if (dist > furthestDistance)
                {
                    furthestDistance = dist;
                    furthestReferenceDistance = dist; // Update reference distance for furthest POI
                    furthestPOI = poi;
                }
            }
        }

        if (furthestDistance < 0f || furthestPOI == null)
        {
            Debug.LogWarning("No POIs found to align.");
            return;
        }

        // Place the horizon line in front of the user at the furthest distance (in the look direction)
        Vector3 forward = Camera.main.transform.forward;
        //forward.y = 0; forward.Normalize();
        Vector3 targetPos = userPos + forward * furthestDistance;

        HorizonLine.transform.position = targetPos;
        //HorizonLine.transform.rotation = Quaternion.LookRotation(-forward, Vector3.up);
        UpdateHorizonLineScale(HorizonLine.transform.position);
        //ScaleLineToMatchFieldOfView(HorizonLine);
        Debug.Log("Horizon line moved for calibration.");
    }

    // === STATE 2: After user confirms ===
    public void ConfirmHorizonCalibration()
    {
        if (furthestPOI == null)
        {
            Debug.LogWarning("You must call StartHorizonCalibration first.");
            return;
        }

        float horizonYAtFurthest = HorizonLine.transform.position.y;
        print("Horizon Y at furthest: " + horizonYAtFurthest);
        foreach (var poi in spawnedObjects[POIType.MyRoutePoint])
        {
            Vector3 poiPos = poi.transform.position;
            float poiDistance = Vector2.Distance(
                new Vector2(userPos.x, userPos.z),
                new Vector2(poiPos.x, poiPos.z)
            );
            float newY;
            /*if (isAdjustedType)
            {
                // General POIs: Clamp to horizon
                float t = furthestDistance > 0f ? Mathf.Clamp01(poiDistance / furthestDistance) : 0f;
                newY = Mathf.Lerp(userY, horizonYAtFurthest, t);
                poi.transform.position = new Vector3(poiPos.x, newY, poiPos.z);
            } */

            // Route/other POIs: Ramp up, don't clamp top, so points beyond reference keep going
            float t = (poiDistance - userReferenceDistance) / (furthestReferenceDistance - userReferenceDistance);
            t = Mathf.Max(0, t);
            newY = Mathf.LerpUnclamped(otherPOIsStartY, horizonYAtFurthest, t);
            print("POI: " + poi.name + " Distance: " + poiDistance + " t: " + t + " New Y: " + newY);

            // Place the POI
            poi.transform.position = new Vector3(poiPos.x, newY, poiPos.z);

            // --- Add rotation to follow the ramp angle ---
            // Get the ramp direction (in XZ) from the ramp's start to end (horizon)
            Vector3 rampDirection = (HorizonLine.transform.position - new Vector3(userPos.x, horizonYAtFurthest, userPos.z)).normalized;
            poi.transform.rotation = Quaternion.LookRotation(rampDirection, Vector3.up);
        }
        UpdateOtherPOIScales();
        Debug.Log("All POIs aligned (general types on horizon, others ramping up).");
    }




    Vector3 CalculateObjectLocalPosition(double latitude, double longitude)
    {
        double latOffset = (latitude - referenceLatitude) * 111000.0;  // meters per latitude degree
        double lonOffset = (longitude - referenceLongitude) * (111000.0 * Mathf.Cos((float)(referenceLatitude * Mathf.PI / 180.0)));
        return new Vector3((float)lonOffset, 0, (float)latOffset); // Changed Y to 0 for visualization on a flat plane
    }


    void AdjustDirectionsForPoints(POIType pointType)
    {
        var pointsList = spawnedObjects[pointType];

        for (int i = 0; i < pointsList.Count - 1; i++)
        {
            Transform currentPoint = pointsList[i].transform;
            Transform nextPoint = pointsList[i + 1].transform;

            Vector3 direction = nextPoint.position - currentPoint.position;
            if (direction != Vector3.zero)
            {
                currentPoint.LookAt(nextPoint);
            }
        }

        // Handle the last object separately
        if (pointsList.Count > 1)
        {
            Transform lastPoint = pointsList[pointsList.Count - 1].transform;
            Transform secondLastPoint = pointsList[pointsList.Count - 2].transform;

            Vector3 lastDirection = secondLastPoint.position - lastPoint.position;
            if (lastDirection != Vector3.zero)
            {
                lastPoint.rotation = secondLastPoint.rotation;
            }
        }
    }

    public void CreateArea(List<GPSCoordinate> areaCoordinates)
    {
        GameObject areaGameObject = new GameObject("DynamicArea");
        MeshFilter meshFilter = areaGameObject.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = areaGameObject.AddComponent<MeshRenderer>();
        MeshCollider meshCollider = areaGameObject.AddComponent<MeshCollider>();
        meshCollider.convex = false;
        meshCollider.isTrigger = false;

        if (areaMaterial != null)
        {
            meshRenderer.material = areaMaterial; // Use the material provided through the Inspector
        }
        else
        {
            Debug.LogWarning("Area material not assigned. Using default white material.");
            meshRenderer.material = new Material(Shader.Find("Standard"));
            meshRenderer.material.color = Color.white;
        }

        AreaScript areaScript = areaGameObject.AddComponent<AreaScript>();

        // Convert GPS coordinates to local positions and create vertices
        Vector3[] vertices = new Vector3[areaCoordinates.Count];
        for (int i = 0; i < areaCoordinates.Count; i++)
        {
            vertices[i] = CalculateObjectLocalPosition(areaCoordinates[i].Latitude, areaCoordinates[i].Longitude);
        }

        // Assuming the area is flat on the ground, you only need to define vertices in the XZ plane
        int[] triangles = Triangulate(vertices.Length); // You need to implement this based on your vertices

        // Create the mesh
        Mesh mesh = new Mesh();
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();

        meshFilter.mesh = mesh;
        meshCollider.sharedMesh = mesh;
    }
    public void CreateAreaOnMap(List<GPSCoordinate> areaCoords)
    {
        // Make sure we have a MapAndPlayerManager
        MapAndPlayerManager mapManager = mapGameObject.GetComponentInParent<MapAndPlayerManager>();
        if (mapManager == null)
        {
            Debug.LogError("MapAndPlayerManager not found in parent. Cannot register area on map.");
            return;
        }

        // 1. Create a new GameObject under mapGameObject
        GameObject areaMapObj = new GameObject("DynamicAreaMap");
        // Put it in the root (or a neutral empty parent that isn’t rotated):
        areaMapObj.transform.SetParent(mapGameObject.transform, false);

        // 2. Add MeshFilter and MeshRenderer
        MeshFilter meshFilter = areaMapObj.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = areaMapObj.AddComponent<MeshRenderer>();

        // 3. Assign your map-specific material
        if (areaMapMaterial != null)
        {
            // If you want the same as the real world
            meshRenderer.material = areaMapMaterial;
        }
        else if (areaMapMaterial != null)
        {
            // Or if you want a separate material
            meshRenderer.material = areaMapMaterial;
        }
        else
        {
            // Fallback
            meshRenderer.material = new Material(Shader.Find("Standard"));
            meshRenderer.material.color = Color.yellow;
        }

        // 4. Convert the GPSCoordinates into lat/lon pairs
        List<Vector2d> latLonList = new List<Vector2d>();
        foreach (var coord in areaCoords)
        {
            latLonList.Add(new Vector2d(coord.Latitude, coord.Longitude));
        }

        // 5. Store this new object + lat/lon in the map manager
        mapManager.spawnedAreas.Add(areaMapObj, latLonList);

        // 6. Immediately update so we can see it if the map is active
        mapManager.UpdateSpawnedAreas();
    }




    // Simple triangulation function for convex shapes. For complex shapes, use a proper triangulation library.
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

    public List<GPSCoordinate> ParseCoordinates(string coordinatesString)
    {
        List<GPSCoordinate> coordinatesList = new List<GPSCoordinate>();
        string[] coordinatePairs = coordinatesString.Split(new char[] { ' ', '[', ']', ',' }, StringSplitOptions.RemoveEmptyEntries);

        for (int i = 0; i < coordinatePairs.Length; i += 2)
        {
            double latitude = double.Parse(coordinatePairs[i], CultureInfo.InvariantCulture);
            double longitude = double.Parse(coordinatePairs[i + 1], CultureInfo.InvariantCulture);

            // Add the parsed coordinates to the list
            coordinatesList.Add(new GPSCoordinate
            {
                coordinates = $"{latitude},{longitude}",
                POIType = POIType.General
            });
        }

        return coordinatesList;
    }

    public void DeactivatePOIsByType(POIType poiType)
    {
        if (spawnedObjects.ContainsKey(poiType))
        {
            foreach (GameObject poi in spawnedObjects[poiType])
            {
                if (poi != null)
                {
                    poi.SetActive(false);
                }
            }
        }
        else
        {
            Debug.LogWarning($"No POIs found for type {poiType}.");
        }
    }

    // Function to activate all POIs of a specific type
    public void ActivatePOIsByType(POIType poiType)
    {
        if (spawnedObjects.ContainsKey(poiType))
        {
            foreach (GameObject poi in spawnedObjects[poiType])
            {
                if (poi != null)
                {
                    poi.SetActive(true);
                }
            }
        }
        else
        {
            Debug.LogWarning($"No POIs found for type {poiType}.");
        }
    }
    // Method to handle toggle state changes for POIs
    public void TogglePOIType(bool isActive, POIType poiType)
    {
        if (isActive)
        {
            ActivatePOIsByType(poiType);
        }
        else
        {
            DeactivatePOIsByType(poiType);
        }
    }

    // individual toggle methods for specific POI types
    public void ToggleCollisionPoints(bool isActive)
    {
        TogglePOIType(isActive, POIType.CollisionPoint);
    }

    public void ToggleGreenLighthouse(bool isActive)
    {
        TogglePOIType(isActive, POIType.GreenLighthouse);
    }

    public void ToggleRedLighthouse(bool isActive)
    {
        TogglePOIType(isActive, POIType.RedLighthouse);
    }

    public void ToggleUnknownDanger(bool isActive)
    {
        TogglePOIType(isActive, POIType.UnknownDanger);
    }

    public void ToggleMooringBuoy(bool isActive)
    {
        TogglePOIType(isActive, POIType.MooringBuoy);
    }

    public void ToggleWreck(bool isActive)
    {
        TogglePOIType(isActive, POIType.Wreck);
    }

    public void ToggleSpecialPurposeBuoy(bool isActive)
    {
        TogglePOIType(isActive, POIType.SpecialPurposeBuoy);
    }

    public void ToggleOtherShip(bool isActive)
    {
        TogglePOIType(isActive, POIType.OtherShip);
    }

    public void ToggleRocks(bool isActive)
    {
        TogglePOIType(isActive, POIType.Reef);
    }

    // Toggle Suggested Route Points
    public void ToggleSuggestedRoutePoints(bool isActive)
    {
        TogglePOIType(isActive, POIType.MyRoutePoint);
    }

    // Toggle Other Ship's Suggested Route Points
    public void ToggleOtherShipsSuggestedRoutePoints(bool isActive)
    {
        TogglePOIType(isActive, POIType.OtherRoutePoint);
    }

    // Toggle Suggested Route Line
    public void ToggleSuggestedRouteLine(bool isActive)
    {
        if (myRouteLineRenderer != null)
        {
            myRouteLineRenderer.gameObject.SetActive(isActive);
        }
    }

    // Toggle Uncertainty Route Line
    public void ToggleUncertaintyRouteLine(bool isActive)
    {
        if (myRouteUncertainLineRenderer != null)
        {
            myRouteUncertainLineRenderer.gameObject.SetActive(isActive);
        }
    }

    // Toggle Other Ship's Suggested Route Line
    public void ToggleOtherShipsSuggestedRouteLine(bool isActive)
    {
        if (otherRouteLineRenderer != null)
        {
            otherRouteLineRenderer.gameObject.SetActive(isActive);
        }
    }

    // Toggle Other Ship's Uncertainty Route Line
    public void ToggleOtherShipsUncertaintyRouteLine(bool isActive)
    {
        if (otherRouteUncertainLineRenderer != null)
        {
            otherRouteUncertainLineRenderer.gameObject.SetActive(isActive);
        }
    }

    // Toggle Suggested Route Line on Map
    public void ToggleSuggestedRouteLineOnMap(bool isActive)
    {
        if (mapGameObject != null)
        {
            foreach (Transform child in mapGameObject.transform)
            {
                if (child.name.Contains("SuggestedRouteLine"))
                {
                    child.gameObject.SetActive(isActive);
                }
            }
        }
    }

    // Toggle Uncertainty Route Line on Map
    public void ToggleUncertaintyRouteLineOnMap(bool isActive)
    {
        if (mapGameObject != null)
        {
            foreach (Transform child in mapGameObject.transform)
            {
                if (child.name.Contains("UncertaintyRouteLine"))
                {
                    child.gameObject.SetActive(isActive);
                }
            }
        }
    }

    // Toggle Other Ship's Suggested Route Line on Map
    public void ToggleOtherShipsSuggestedRouteLineOnMap(bool isActive)
    {
        if (mapGameObject != null)
        {
            foreach (Transform child in mapGameObject.transform)
            {
                if (child.name.Contains("OtherShipsSuggestedRouteLine"))
                {
                    child.gameObject.SetActive(isActive);
                }
            }
        }
    }

    // Toggle Other Ship's Uncertainty Route Line on Map
    public void ToggleOtherShipsUncertaintyRouteLineOnMap(bool isActive)
    {
        if (mapGameObject != null)
        {
            foreach (Transform child in mapGameObject.transform)
            {
                if (child.name.Contains("OtherShipsUncertaintyRouteLine"))
                {
                    child.gameObject.SetActive(isActive);
                }
            }
        }
    }

    // Disable All Route Elements
    public void DisableAllRouteElements()
    {
        ToggleSuggestedRoutePoints(false);
        ToggleOtherShipsSuggestedRoutePoints(false);
        ToggleSuggestedRouteLine(false);
        ToggleUncertaintyRouteLine(false);
        ToggleOtherShipsSuggestedRouteLine(false);
        ToggleOtherShipsUncertaintyRouteLine(false);
        ToggleOtherShipsSuggestedRouteLineOnMap(false);
        ToggleOtherShipsUncertaintyRouteLineOnMap(false);
        ToggleSuggestedRouteLineOnMap(false);
        ToggleUncertaintyRouteLineOnMap(false);
    }

    // Enable All Route Elements
    public void EnableAllRouteElements()
    {
        ToggleSuggestedRoutePoints(true);
        ToggleOtherShipsSuggestedRoutePoints(true);
        ToggleSuggestedRouteLine(true);
        ToggleUncertaintyRouteLine(true);
        ToggleOtherShipsSuggestedRouteLine(true);
        ToggleOtherShipsUncertaintyRouteLine(true);
        ToggleOtherShipsSuggestedRouteLineOnMap(true);
        ToggleOtherShipsUncertaintyRouteLineOnMap(true);
        ToggleSuggestedRouteLineOnMap(true);
        ToggleUncertaintyRouteLineOnMap(true);
    }



    public void clearSpawnedObjectsByType(POIType poiType)
    {
        if (spawnedObjects.ContainsKey(poiType))
        {
            foreach (GameObject poi in spawnedObjects[poiType])
            {
                if (poi != null)
                {
                    Destroy(poi);
                }
            }
            spawnedObjects[poiType].Clear();
        }
        else
        {
            Debug.LogWarning($"No POIs found for type {poiType} to clear.");
        }
    }

    void testReferenceCoordinatesUpdate()
    {
        referenceLatitude = double.Parse(referenceCoordinates.Split(',')[0].Trim());
        referenceLongitude = double.Parse(referenceCoordinates.Split(',')[1].Trim());
        referenceCoordinatesSet = true;

        mapGameObject.GetComponentInParent<MapAndPlayerManager>().SetCoordinates(new Vector2d(referenceLatitude, referenceLongitude));
    }


}
