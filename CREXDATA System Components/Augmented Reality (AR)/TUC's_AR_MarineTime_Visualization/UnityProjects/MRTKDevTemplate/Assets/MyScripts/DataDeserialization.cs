using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class ShipData
{
    public int ship1_mmsi;
    public int ship2_mmsi;
    public double ship1_speed;
    public double ship2_speed;
    public double ship1_course;
    public double ship2_course;
    public double[] ship1_traj_lon;
    public double[] ship1_traj_lat;
    public double[] ship2_traj_lon;
    public double[] ship2_traj_lat;
    public string LINESTRING1;
    public string LINESTRING2;
    public double collision_lon;
    public double collision_lat;
    public long timestamp;
    public string collision_id;
}

public class DataDeserialization : MonoBehaviour
{
    public static DataDeserialization Instance { get; private set; }
    public ShipData ShipData { get; private set; }

    public TextAsset jsonData; // Drag your JSON file here in the Inspector

    void Awake()
    {
        // Ensure there's only one instance of this script in the scene.
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject); // Keep this object alive when loading new scenes
        }
        else if (Instance != this)
        {
            Destroy(gameObject);
        }
    }


    void Start()
    {
        ShipData = JsonUtility.FromJson<ShipData>(jsonData.text);
        // Logging all parameters
        Debug.Log("Ship 1 MMSI: " + ShipData.ship1_mmsi);
        Debug.Log("Ship 2 MMSI: " + ShipData.ship2_mmsi);
        Debug.Log("Ship 1 Speed: " + ShipData.ship1_speed);
        Debug.Log("Ship 2 Speed: " + ShipData.ship2_speed);
        Debug.Log("Ship 1 Course: " + ShipData.ship1_course);
        Debug.Log("Ship 2 Course: " + ShipData.ship2_course);

        // Assuming you want to log trajectory arrays in some meaningful way
        Debug.Log("Ship 1 Trajectory Longitude: " + (ShipData.ship1_traj_lon.Length > 0 ? string.Join(", ", ShipData.ship1_traj_lon) : "No data"));
        Debug.Log("Ship 1 Trajectory Latitude: " + (ShipData.ship1_traj_lat.Length > 0 ? string.Join(", ", ShipData.ship1_traj_lat) : "No data"));
        Debug.Log("Ship 2 Trajectory Longitude: " + (ShipData.ship2_traj_lon.Length > 0 ? string.Join(", ", ShipData.ship2_traj_lon) : "No data"));
        Debug.Log("Ship 2 Trajectory Latitude: " + (ShipData.ship2_traj_lat.Length > 0 ? string.Join(", ", ShipData.ship2_traj_lat) : "No data"));

        Debug.Log("LINESTRING1: " + ShipData.LINESTRING1);
        Debug.Log("LINESTRING2: " + ShipData.LINESTRING2);
        Debug.Log("Collision Longitude: " + ShipData.collision_lon);
        Debug.Log("Collision Latitude: " + ShipData.collision_lat);
        Debug.Log("Timestamp: " + ShipData.timestamp);
        Debug.Log("Collision ID: " + ShipData.collision_id);

    }

    // Call this method to get LINESTRING1 or LINESTRING2 coordinates as Vector2
    public Vector2[] GetLinestringCoordinates(string linestring)
    {
        // Trim the extraneous characters ("LINESTRING (" and ")") and split the string into lon/lat pairs.
        string trimmedString = linestring.Replace("\"LINESTRING (", "").Replace(")\"", "");
        string[] coordStrings = trimmedString.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);

        List<Vector2> coordinates = new List<Vector2>();

        for (int i = 0; i < coordStrings.Length; i += 2)
        {
            double lon, lat;
            if (double.TryParse(coordStrings[i], out lon) && double.TryParse(coordStrings[i + 1], out lat))
            {
                coordinates.Add(new Vector2((float)lon, (float)lat));
            }
        }

        return coordinates.ToArray();
    }

    public int GetShip1MMSI() => ShipData.ship1_mmsi;
    public int GetShip2MMSI() => ShipData.ship2_mmsi;
    public double GetShip1Speed() => ShipData.ship1_speed;
    public double GetShip2Speed() => ShipData.ship2_speed;
    public double GetShip1Course() => ShipData.ship1_course;
    public double GetShip2Course() => ShipData.ship2_course;
    public double[] GetShip1TrajLon() => ShipData.ship1_traj_lon;
    public double[] GetShip1TrajLat() => ShipData.ship1_traj_lat;
    public double[] GetShip2TrajLon() => ShipData.ship2_traj_lon;
    public double[] GetShip2TrajLat() => ShipData.ship2_traj_lat;
    public string GetLINESTRING1() => ShipData.LINESTRING1;
    public string GetLINESTRING2() => ShipData.LINESTRING2;
    public double GetCollisionLon() => ShipData.collision_lon;
    public double GetCollisionLat() => ShipData.collision_lat;
    public long GetTimestamp() => ShipData.timestamp;
    public string GetCollisionId() => ShipData.collision_id;

}
