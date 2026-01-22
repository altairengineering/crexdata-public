using System;
using UnityEngine;

public class PairingManager : MonoBehaviour
{
    public WebSocketConnection webSocketConnection;
    public string targetPhoneId;

    private enum PairingState
    {
        NotPaired,
        WaitingForPhone,
        Paired,
        PhoneDisconnected
    }

    [SerializeField] private PairingState currentState = PairingState.NotPaired;

    void Start()
    {
        // Message handlers
        webSocketConnection.OnPairingStatusReceived += HandlePairingStatus;
        webSocketConnection.OnPhoneLostReceived += HandlePhoneLost;
        webSocketConnection.OnPairedReceived += HandlePaired;

    }

    public void StartPairing()
    {
        currentState = PairingState.NotPaired;
        RequestPairing();
    }

    private void RequestPairing()
    {
        if (string.IsNullOrEmpty(targetPhoneId))
        {
            Debug.LogWarning("No target phone ID specified for pairing, setting same as Unity client ID.");
            targetPhoneId = webSocketConnection.uniqueID;
        }
        Debug.Log($"Requesting pairing with phone: {targetPhoneId}");
        webSocketConnection.RequestPairing(targetPhoneId);
    }

    private void HandlePairingStatus(string status, string phoneId)
    {
        if (status == "waiting" && phoneId == targetPhoneId)
        {
            Debug.Log($"Waiting for phone {phoneId} to connect...");
            currentState = PairingState.WaitingForPhone;
        }
    }

    private void HandlePaired(string phoneId)
    {
        if (phoneId == targetPhoneId)
        {
            Debug.Log($"Successfully paired with phone {phoneId}!");
            currentState = PairingState.Paired;
        }
    }

    private void HandlePhoneLost(string phoneId)
    {
        if (phoneId == targetPhoneId)
        {
            Debug.Log($"Phone {phoneId} disconnected. Waiting for reconnection...");
            currentState = PairingState.PhoneDisconnected;
            RequestPairing();
        }
    }

    void OnDestroy()
    {
        // Unregister message handlers
        if (webSocketConnection != null)
        {
            webSocketConnection.OnPairingStatusReceived -= HandlePairingStatus;
            webSocketConnection.OnPhoneLostReceived -= HandlePhoneLost;
            webSocketConnection.OnPairedReceived -= HandlePaired;
        }
    }
}