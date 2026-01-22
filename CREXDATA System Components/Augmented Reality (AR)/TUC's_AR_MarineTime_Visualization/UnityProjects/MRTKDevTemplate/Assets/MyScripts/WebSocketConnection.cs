using System.Collections;
using WebSocketSharp;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Mapbox.Unity.Location;
using Mapbox.Unity.Map;
using Mapbox.Utils;
using System.Collections.Generic;

public class WebSocketConnection : MonoBehaviour
{
    WebSocket ws;
    public TextMeshProUGUI jsonText;
    public string ip;
    public bool isApplicationQuitting = false;
    public int attempt = 0;
    public string uniqueID;
    NetworkReachability previousReachability;
    [SerializeField] private MapAndPlayerManager mapManager;
    //serialized field game object to pass the game object to the script
    public GPSMulti spawner;
    public PairingManager pairingManager;
    void Start()
    {
        spawner = GetComponent<GPSMulti>();
        //locationProvider = stick.GetComponent<LocationArrayEditorLocationProvider>();

        previousReachability = Application.internetReachability;
        uniqueID = SystemInfo.deviceUniqueIdentifier;

        // Attach the SendInputMessage method to the button's click event.
        // sendButton.onClick.AddListener(SendInputMessage);

        // ChiefButton.onClick.AddListener(() => LocationManager.ChangeStickPosition(stick, "35.512083, 24.019154"));
        // Operator1Button.onClick.AddListener(() => SendMessageToServer("2"));
        // Operator2Button.onClick.AddListener(() => SendMessageToServer("3"));
        // Operator3Button.onClick.AddListener(() => SendMessageToServer("4"));

        // saveButton.onClick.AddListener(SaveLastMessageInfo);
        // loadButton.onClick.AddListener(LoadLastMessageInfo);

        // Connect to the WebSocket server.
        ConnectToServer();
    }

    // This method is responsible for connecting to the WebSocket server.
    void ConnectToServer()
    {
        // Initialize the WebSocket connection to the specified URL.
        ws = new WebSocket("ws://" + ip + ":8080");
        Debug.Log("Connecting to server...");
        // Subscribe to the OnOpen event. This event is triggered when the connection is opened.
        ws.OnOpen += OnConnectionOpened;

        // Subscribe to the OnMessage event. This event is triggered when a message is received from the server.
        ws.OnMessage += OnMessageReceived;

        // Subscribe to the OnClose event. This event is triggered when the connection is closed.
        ws.OnClose += OnConnectionClosed;

        // Connect to the server asynchronously.
        ws.ConnectAsync();
    }

    public event EventHandler OnConnectionOpenedEvent;
    public event EventHandler<CloseEventArgs> OnConnectionClosedEvent;
    public event EventHandler OnMessageReceivedEvent;
    public event Action<string> OnUniqueIDReceivedEvent;

    public event Action<string, string> OnPairingStatusReceived; // (status, phoneId)
    public event Action<string> OnPhoneLostReceived; // (phoneId)
    public event Action<string> OnPairedReceived; // (phoneId)

    // This method is called when the connection is opened.
    void OnConnectionOpened(object sender, EventArgs e)
    {
        // Create a registration message
        var registrationMessage = new
        {
            type = "registration",
            deviceType = "unity",
            useCase = "marine",
            clientId = uniqueID
        };

        // Convert to JSON and send
        string jsonMessage = JsonConvert.SerializeObject(registrationMessage);
        SendMessageToServer(jsonMessage);

        if (attempt > 0)
        {
            Debug.Log("Reconnected to the server.");
            SendMessageToServer("Recome");
        }
        else
        {
            Debug.Log("Connected to the server");
            SendMessageToServer("Come");
        }

        attempt = 0;

        UnityMainThreadDispatcher.Instance.Enqueue(() =>
{
    OnConnectionOpenedEvent?.Invoke(this, e);
});
    }

    // This method is called when a message is received from the server.
    void OnMessageReceived(object sender, MessageEventArgs e)
    {
        UnityMainThreadDispatcher.Instance.Enqueue(() =>
        {
            HandleMessage(e.Data);
        });
    }

    async void OnConnectionClosed(object sender, CloseEventArgs e)
    {
        Debug.Log($"Connection closed. Attempting to reconnect. Total attempts: {++attempt}");

        UnityMainThreadDispatcher.Instance.Enqueue(() =>
        {
            OnConnectionClosedEvent?.Invoke(this, e);
        });

        // Wait for a delay before attempting to reconnect. The delay increases with each failed attempt.
        int delay = 3000;
        await Task.Delay(delay);

        // Check if the application is quitting before attempting to reconnect.
        if (!isApplicationQuitting)
            ConnectToServer();
    }
    void HandleMessage(string message)
    {
        try
        {
            // Try to parse as JSON first
            JObject jsonObject = JObject.Parse(message);
            string messageType = jsonObject["type"]?.ToString();

            if (messageType != null)
            {
                // Handle JSON messages by type
                switch (messageType)
                {
                    case "connectionAck":
                        HandleAknowledgmentMessage(jsonObject);
                        break;

                    case "registrationResult":
                        string registrationStatus = jsonObject["status"].ToString();
                        uniqueID = jsonObject["clientId"].ToString();
                        OnUniqueIDReceivedEvent?.Invoke(uniqueID);
                        string registrationMessage = jsonObject["message"].ToString();

                        Debug.Log($"Registration result: {registrationStatus}, " +
                                  $"Message: {registrationMessage}");

                        RequestPairing();
                        break;

                    case "gps":
                        HandleGpsMessage(jsonObject);
                        break;

                    case "spawnPOI":
                        //HandleSpawnMessage(jsonObject);
                        break;

                    case "MarineTraffic_route":
                        // Handle MarineTraffic route messages
                        //Debug.Log("Received MarineTraffic route message: " + message);
                        HandleRouteMessage(jsonObject);
                        break;

                    case "pairingStatus":
                        string status = jsonObject["status"].ToString();
                        string statusPhoneId = jsonObject["phoneId"].ToString();
                        OnPairingStatusReceived?.Invoke(status, statusPhoneId);
                        break;

                    case "paired":
                        string pairedWith = jsonObject["pairedWith"].ToString();
                        OnPairedReceived?.Invoke(pairedWith);
                        break;

                    case "phoneLost":
                        string lostPhoneId = jsonObject["phoneId"].ToString();
                        OnPhoneLostReceived?.Invoke(lostPhoneId);
                        break;

                    case "staticPoisResult":
                        Debug.Log("Static POI result: " + message);
                        break;

                    case "kafkaStatus":
                        string kafkaStatus = jsonObject["status"].ToString();
                        string kafkaConfig = jsonObject["configKey"].ToString();
                        string reason = kafkaStatus == "error" ? jsonObject["reason"].ToString() : "No reason";
                        string topic = kafkaStatus == "ready" ? jsonObject["topic"].ToString() : "No topic";

                        Debug.Log($"Kafka status: {kafkaStatus}, " +
                                  $"Config Key: {kafkaConfig}, " +
                                  $"Reason: {reason}, " +
                                  $"Topic: {topic}");
                        break;

                    default:
                        Debug.Log($"Received unknown JSON message type: {messageType}");
                        break;
                }

                // Display the JSON for debugging
                if (jsonText != null)
                {
                    //DisplayJsonData(jsonObject);
                }

                return; // Successfully handled as JSON
            }
        }
        catch (JsonReaderException)
        {
            Debug.Log($"Not a JSON message, using legacy handling: {message.Substring(0, Math.Min(50, message.Length))}...");
            // Not a valid JSON, continue with legacy handling
        }

        HandleLegacyMessage(message);
    }

    void HandleAknowledgmentMessage(JObject jsonObject)
    {
        try
        {
            // Extract all fields with null checking
            string message = jsonObject["message"]?.ToString() ?? "No message";
            string connectionId = jsonObject["connectionId"]?.ToString() ?? "Unknown";
            string userAgent = jsonObject["userAgent"]?.ToString() ?? "Unknown";
            string ipAddress = jsonObject["ipAddress"]?.ToString() ?? "Unknown";
            string connectionTime = jsonObject["connectionTime"]?.ToString() ?? "Unknown";

            Debug.Log($"Connection acknowledgment received:\n" +
                      $"Message: {message}\n" +
                      $"Connection ID: {connectionId}\n" +
                      $"User Agent: {userAgent}\n" +
                      $"IP Address: {ipAddress}\n" +
                      $"Connection Time: {connectionTime}");

            // Formated string for UI if needed later***
            if (jsonText != null)
            {
                string formattedInfo = $"Connection Details:\n" +
                                       $"• Message: {message}\n" +
                                       $"• Connection ID: {connectionId}\n" +
                                       $"• User Agent: {userAgent}\n" +
                                       $"• IP Address: {ipAddress}\n" +
                                       $"• Connection Time: {connectionTime}";

                jsonText.text = formattedInfo;
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("Error handling acknowledgment message: " + ex.Message);
        }
    }

    void HandleLegacyMessage(string message)
    {

        //format the message
        string rawMessage;
        try
        {
            rawMessage = RawMessage(message);
        }
        catch (Exception ex)
        {
            Debug.LogWarning("An error occurred when formatting the message: " + ex.Message);
            rawMessage = message;
        }
        Debug.Log(rawMessage.Contains("spawn"));

        if (message == "Readys")
        {
            Debug.Log("Ready!");
        }

        if (message.Contains("info"))
        {
            if (rawMessage.Contains("spawn"))
            {
                Debug.Log("Spawn section handling: " + rawMessage);
                // This assumes that the latitude and longitude are part of the 'raw_message'
                string[] spawnParts = rawMessage.Split(',');
                double latitude = double.Parse(spawnParts[0]);
                double longitude = double.Parse(spawnParts[1]);
                //print spawn parts
                Debug.Log("Latitude: " + latitude + " Longitude: " + longitude);
                if (spawner != null)
                {
                    Debug.Log("Spawning object at: " + latitude + ", " + longitude);
                    //spawner.SpawnObjectAtLocation(new GPSCoordinate { Latitude = latitude, Longitude = longitude });
                }
                else
                    Debug.Log("Spawner is null");
            }

            if (rawMessage.Contains("gps"))
            {
                Debug.Log("GPS Phone");

            }

            Debug.Log("Info section handling: " + message);
            //SaveLoadManager.AddMessage(int.Parse(message.Split(',')[1]), rawMessage, long.Parse(message.Split(',')[3]));
        }
    }
    // This method is called when the connection is closed.
    void HandleGpsMessage(JObject jsonObject)
    {
        try
        {
            double latitude = jsonObject["lat"].Value<double>();
            double longitude = jsonObject["lon"].Value<double>();

            Debug.Log($"Received GPS: Lat {latitude}, Long {longitude}");

            try
            {
                if (spawner != null && !spawner.referenceCoordinatesSet)
                {
                    spawner.UpdateReferenceCoordinates(latitude, longitude);
                }
            }
            catch (Exception e)
            {
                Debug.LogError("Error updating reference coordinates: " + e.Message);
            }

            mapManager.SetCoordinates(new Vector2d(latitude, longitude));
            mapManager.UpdatePlayerLocation(new Vector2d(latitude, longitude));
        }
        catch (Exception ex)
        {
            Debug.LogError("Error handling GPS message: " + ex.Message);
        }
    }

    public void SendJsonToServer(object data)
    {
        string jsonMessage = JsonConvert.SerializeObject(data);
        SendMessageToServer(jsonMessage);
    }

    void RequestPairing()
    {
        if (pairingManager != null)
        {
            pairingManager.StartPairing();
        }
        else
        {
            Debug.LogWarning("PairingManager not found in the scene.");
        }
    }
    public void RequestPairing(string phoneId)
    {
        var pairRequest = new
        {
            type = "pair",
            unityId = uniqueID,
            phoneId = phoneId
        };

        SendJsonToServer(pairRequest);
    }

    void SendMessageToServer(string message)
    {
        try
        {
            if (ws.ReadyState == WebSocketState.Open)
            {
                ws.Send(message);
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("An error occurred when trying to send a message: " + ex.Message);
        }
    }

    public static string RawMessage(string input)
    {
        // Split the message by commas
        string[] parts = input.Split(',');

        // Known positions
        string prefix = parts[0] + "," + parts[1]; // "safran4,4"
        string suffix = parts[parts.Length - 2] + "," + parts[parts.Length - 1]; // "5,info"

        // Extract raw_message starting from index 2 to parts.Length - 3
        string rawMessage = string.Join(",", parts, 2, parts.Length - 4);

        // Reformat into the desired structure
        return rawMessage;
    }

    public void HandleRouteMessage(JObject jsonObject)
    {
        try
        {
            spawner.clearSpawnedObjectsByType(POIType.MyRoutePoint);
            spawner.clearSpawnedObjectsByType(POIType.OtherRoutePoint);
            spawner.clearSpawnedObjectsByType(POIType.CollisionPoint);

            JObject payload = (JObject)jsonObject["payload"];
            if (payload == null)
            {
                Debug.LogError("Spawn POI message has no payload");
                return;
            }

            string ship1Mmsi = payload["ship1_mmsi"]?.ToString();
            string ship2Mmsi = payload["ship2_mmsi"]?.ToString();

            // Extracting other scalar values
            double? ship1Speed = payload["ship1_speed"]?.Value<double>();
            double? ship2Speed = payload["ship2_speed"]?.Value<double>();
            double? ship1Course = payload["ship1_course"]?.Value<double>();
            double? ship2Course = payload["ship2_course"]?.Value<double>();
            double? collisionLon = payload["collision_lon"]?.Value<double>();
            double? collisionLat = payload["collision_lat"]?.Value<double>();
            long? timestamp = payload["timestamp"]?.Value<long>();
            string collisionId = payload["collision_id"]?.ToString();
            string linestring1 = payload["LINESTRING1"]?.ToString();
            string linestring2 = payload["LINESTRING2"]?.ToString();

            // Extracting arrays
            List<double> ship1TrajLon = payload["ship1_traj_lon"]?.ToObject<List<double>>() ?? new List<double>();
            List<double> ship1TrajLat = payload["ship1_traj_lat"]?.ToObject<List<double>>() ?? new List<double>();
            List<double> ship2TrajLon = payload["ship2_traj_lon"]?.ToObject<List<double>>() ?? new List<double>();
            List<double> ship2TrajLat = payload["ship2_traj_lat"]?.ToObject<List<double>>() ?? new List<double>();

            // You can now use these variables. For example, to log them:
            Debug.Log($"Ship 1 MMSI: {ship1Mmsi}, Speed: {ship1Speed}, Course: {ship1Course}");
            Debug.Log($"Ship 2 MMSI: {ship2Mmsi}, Speed: {ship2Speed}, Course: {ship2Course}");
            Debug.Log($"Collision Lon: {collisionLon}, Lat: {collisionLat}, Timestamp: {timestamp}, ID: {collisionId}");
            Debug.Log($"Linestring 1: {linestring1}");
            Debug.Log($"Linestring 2: {linestring2}");

            Debug.Log($"Ship 1 Trajectory Longitudes Count: {ship1TrajLon.Count}");
            if (ship1TrajLon.Count > 0)
            {
                Debug.Log($"First Ship 1 Traj Lon: {ship1TrajLon[0]}");
            }
            Debug.Log($"Ship 2 Trajectory Longitudes Count: {ship2TrajLon.Count}");
            if (ship2TrajLon.Count > 0)
            {
                Debug.Log($"First Ship 2 Traj Lon: {ship2TrajLon[0]}");
            }

            //LOGIC


            if (ship1TrajLon != null && ship1TrajLat != null)
            {
                for (int i = 0; i < ship1TrajLon.Count; i++)
                {
                    GPSCoordinate coordinates = new GPSCoordinate
                    {
                        coordinates = ship1TrajLat[i] + "," + ship1TrajLon[i],
                        POIType = POIType.MyRoutePoint
                    };
                    GameObject pointObject = spawner.SpawnObjectAtLocation(coordinates);
                    //spawner.spawnedObjects[coordinates.POIType].Add(pointObject);
                    //Debug.Log("Ship 1111111: " + ship1TrajLat[i] + ship1TrajLon[i]);
                }

                // Initialize the MyRouteLineRenderer position count
                spawner.myRouteLineRenderer.positionCount = spawner.spawnedObjects[POIType.MyRoutePoint].Count;
                spawner.myRouteUncertainLineRenderer.positionCount = spawner.spawnedObjects[POIType.MyRoutePoint].Count;
            }

            if (ship2TrajLon != null && ship2TrajLat != null)
            {
                for (int i = 0; i < ship2TrajLon.Count; i++)
                {
                    GPSCoordinate coordinates = new GPSCoordinate
                    {
                        coordinates = ship2TrajLat[i] + "," + ship2TrajLon[i],
                        POIType = POIType.OtherRoutePoint
                    };
                    GameObject pointObject = spawner.SpawnObjectAtLocation(coordinates);
                    //spawner.spawnedObjects[coordinates.POIType].Add(pointObject);
                    //Debug.Log("Ship 222222: " + ship2TrajLat[i] + ship2TrajLon[i]);
                }

                // Initialize the OtherRouteLineRenderer position count
                spawner.otherRouteLineRenderer.positionCount = spawner.spawnedObjects[POIType.OtherRoutePoint].Count;
                spawner.otherRouteUncertainLineRenderer.positionCount = spawner.spawnedObjects[POIType.OtherRoutePoint].Count;
            }

            if (collisionLon != null && collisionLat != null)
            {
                GPSCoordinate coordinates = new GPSCoordinate
                {

                    coordinates = collisionLat + "," + collisionLon,
                    POIType = POIType.CollisionPoint
                };
                GameObject pointObject = spawner.SpawnObjectAtLocation(coordinates);
                //spawner.spawnedObjects[coordinates.POIType].Add(pointObject);
                //Debug.Log("Ship 222222: " + ship2TrajLat[i] + ship2TrajLon[i]);
            }

            if (linestring1 != null && linestring1 != "null")
            {

            }


        }
        catch (Exception ex)
        {
            Debug.LogError("Error handling MarineTraffic route message: " + ex.Message);
        }
    }
    // This method is called when the application is quitting.
    void OnApplicationQuit()
    {
        // Set the flag to indicate that the application is quitting.
        isApplicationQuitting = true;
        if (ws != null && ws.IsAlive)
        {
            // Unsubscribe from the OnMessage event.
            ws.OnMessage -= OnMessageReceived;
            // Unsubscribe from the OnClose event.
            ws.OnClose -= OnConnectionClosed;
            ws.OnOpen -= OnConnectionOpened;
            ws.CloseAsync();
            ws = null;
            Debug.Log("WebSocket connection closed on application quit.");
        }

        Debug.Log("CYA");
    }
}
