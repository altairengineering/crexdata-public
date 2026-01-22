using UnityEngine;
using Microsoft.MixedReality.QR;  // Provided by Microsoft.MixedReality.QR NuGet package
using System;                    // For EventHandler, etc.

public class QRSimpleTest : MonoBehaviour
{
    private QRCodeWatcher qrWatcher;

    private async void Start()
    {
        // 1. Request access to the QR code detection capability
        var accessStatus = await QRCodeWatcher.RequestAccessAsync();
        if (accessStatus == QRCodeWatcherAccessStatus.Allowed)
        {
            // 2. Create and start the watcher
            qrWatcher = new QRCodeWatcher();
            qrWatcher.Added += OnQRCodeAdded;
            qrWatcher.Updated += OnQRCodeUpdated;
            qrWatcher.Removed += OnQRCodeRemoved;

            qrWatcher.Start();
            Debug.LogError("[QRSimpleTest] QRCodeWatcher started. Waiting for codes...");
        }
        else
        {
            Debug.LogError("[QRSimpleTest] Access to QR codes not allowed.");
        }
    }

    private void OnQRCodeAdded(object sender, QRCodeAddedEventArgs e)
    {
        // Jump back to Unity main thread before using Unity APIs:
        UnityEngine.WSA.Application.InvokeOnAppThread(() =>
        {
            Debug.LogError($"[QRSimpleTest] QR code ADDED! Data = '{e.Code.Data}' (ID: {e.Code.Id})");
        }, false);
    }

    private void OnQRCodeUpdated(object sender, QRCodeUpdatedEventArgs e)
    {
        UnityEngine.WSA.Application.InvokeOnAppThread(() =>
        {
            Debug.LogError($"[QRSimpleTest] QR code UPDATED! Data = '{e.Code.Data}' (ID: {e.Code.Id})");
        }, false);
    }

    private void OnQRCodeRemoved(object sender, QRCodeRemovedEventArgs e)
    {
        UnityEngine.WSA.Application.InvokeOnAppThread(() =>
        {
            Debug.LogError($"[QRSimpleTest] QR code REMOVED! Data = '{e.Code.Data}' (ID: {e.Code.Id})");
        }, false);
    }
}
