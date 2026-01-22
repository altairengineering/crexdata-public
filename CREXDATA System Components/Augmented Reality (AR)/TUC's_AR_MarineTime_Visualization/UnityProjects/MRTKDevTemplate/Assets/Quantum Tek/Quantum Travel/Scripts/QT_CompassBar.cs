using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace QuantumTek.QuantumTravel
{
    /// <summary>
    /// QT_CompassBar is used as a compass bar, showing travel direction in 3D space and any important markers.
    /// </summary>
    [AddComponentMenu("Quantum Tek/Quantum Travel/Compass Bar")]
    [DisallowMultipleComponent]
    public class QT_CompassBar : MonoBehaviour
    {
        [SerializeField] private RectTransform rectTransform = null;
        [SerializeField] private RectTransform barBackground = null;
        [SerializeField] private RectTransform markersTransform = null;
        [SerializeField] private RawImage image = null;

        [Header("Object References")]
        public QT_MapObject ReferenceObject;
        public List<QT_MapObject> Objects = new List<QT_MapObject>();
        public QT_MapMarker MarkerPrefab;
        public List<QT_MapMarker> Markers { get; set; } = new List<QT_MapMarker>();

        [Header("Compass Bar Variables")]
        public Vector2 CompassSize = new Vector2(200, 25);
        public Vector2 ShownCompassSize = new Vector2(100, 25);
        public float MaxRenderDistance = 5;
        public float MarkerSize = 20;
        public float MinScale = 0.5f;
        public float MaxScale = 2f;

        private void Awake()
        {
            foreach (var obj in Objects)
                if (obj.Data.ShowOnCompass)
                    AddMarker(obj);

            rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, ShownCompassSize.x);
            rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, ShownCompassSize.y);
            barBackground.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, CompassSize.x);
            barBackground.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, CompassSize.y);
        }

        private void Update()
        {
            image.uvRect = new Rect(ReferenceObject.transform.localEulerAngles.y / 360, 0, 1, 1);

            foreach (var marker in Markers)
            {
                marker.SetPosition(CalculatePosition(marker));
                marker.SetScale(CalculateScale(marker));
            }
        }

        private Vector2 CalculatePosition(QT_MapMarker marker)
        {
            float compassDegree = CompassSize.x / 360;
            Vector2 referencePosition = ReferenceObject.Position(QT_MapType.Map3D);
            Vector2 referenceForward = new Vector2(ReferenceObject.transform.forward.x, ReferenceObject.transform.forward.z);
            float angle = Vector2.SignedAngle(marker.Object.Position(QT_MapType.Map3D) - referencePosition, referenceForward);

            return new Vector2(compassDegree * angle, 0);
        }

        private Vector2 CalculateScale(QT_MapMarker marker)
        {
            float distance = Vector2.Distance(ReferenceObject.Position(QT_MapType.Map3D), marker.Object.Position(QT_MapType.Map3D));
            float scale = 0;

            if (distance < MaxRenderDistance)
                scale = Mathf.Clamp(1 - distance / MaxRenderDistance, MinScale, MaxScale);

            return new Vector2(scale, scale);
        }

        /// <summary>
        /// Creates a new marker on the compass bar, based on the given object.
        /// </summary>
        /// <param name="obj">The GameObject with a QT_MapObject on it.</param>
        public void AddMarker(QT_MapObject obj)
        {
            QT_MapMarker marker = Instantiate(MarkerPrefab, markersTransform);
            marker.Initialize(obj, MarkerSize);
            Markers.Add(marker);
        }
    }
}