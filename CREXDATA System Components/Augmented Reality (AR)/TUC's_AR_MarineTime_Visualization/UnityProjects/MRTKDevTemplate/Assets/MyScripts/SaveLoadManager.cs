using UnityEngine;
using System.Collections.Generic;

public static class SaveLoadManager
{
    private static Dictionary<int, (long offset, string message)> lastMessages = new Dictionary<int, (long offset, string message)>();

    public static void SaveLastMessages()
    {
        if (lastMessages.Count == 0)
        {
            return;
        }

        string keys = PlayerPrefs.GetString("Keys");
        foreach (var entry in lastMessages)
        {
            int partition = entry.Key;
            (long offset, string message) = entry.Value;

            string key = $"Partition{partition}";
            PlayerPrefs.SetString(key, $"{offset},{message}");

            // Only add the key to the keys string if it's not already there
            if (!keys.Contains(key))
            {
                keys += key + ",";
            }
        }

        PlayerPrefs.SetString("Keys", keys);
        PlayerPrefs.Save();
    }
    //Load last messages from the PlayerPrefs
    public static string LoadLastMessages()
    {
        string lastState = "";
        string keys = PlayerPrefs.GetString("Keys");
        foreach (var key in keys.Split(','))
        {
            if (!string.IsNullOrEmpty(key) && key.Contains("Partition"))
            {
                // Remove the "Partition" part of the key to get the partition number
                string partition = key.Replace("Partition", "");
                lastState += partition + "," + PlayerPrefs.GetString(key) + "\n";
            }
        }
        Debug.Log("Last state: \n" + lastState);
        return lastState;
    }
    public static void AddMessage(int partition, string message, long offset)
    {
        if (lastMessages.ContainsKey(partition))
        {
            // If a message from this partition already exists, update it
            lastMessages[partition] = (offset, message);
        }
        else
        {
            // If no message from this partition exists, add a new one
            lastMessages.Add(partition, (offset, message));
        }
    }
    public static string GetLastData()
    {
        string lastData = "";
        foreach (var entry in lastMessages)
        {
            int partition = entry.Key;
            (long offset, string message) = entry.Value;

            lastData += $"Partition: {partition}, Offset: {offset}, Message: {message}\n";
        }

        return lastData;
    }
    public static void ClearSaveFile()
    {
        PlayerPrefs.DeleteAll();
    }
}