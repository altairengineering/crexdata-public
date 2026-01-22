namespace Mapbox.Examples
{
	using Mapbox.Unity.Location;
	using Mapbox.Unity.Map;
	using Mapbox.Utils;
	using UnityEngine;

	public class ImmediatePositionWithLocationProvider : MonoBehaviour
	{

		bool _isInitialized;
		[SerializeField]
		AbstractMap _map;
		ILocationProvider _locationProvider;
		ILocationProvider LocationProvider
		{
			get
			{
				if (_locationProvider == null)
				{
					_locationProvider = LocationProviderFactory.Instance.DefaultLocationProvider;
				}

				return _locationProvider;
			}
		}

		Vector3 _targetPosition;

		void Start()
		{
			//LocationProviderFactory.Instance.mapManager.OnInitialized += () => _isInitialized = true;
			_map.OnInitialized += () => _isInitialized = true;
		}

		void LateUpdate()
		{
			if (_isInitialized)
			{
				//var map = LocationProviderFactory.Instance.mapManager;
				transform.position = _map.GeoToWorldPosition(new Vector2d(35.53115282806701, 24.067864777926356));
			}
		}
	}
}
