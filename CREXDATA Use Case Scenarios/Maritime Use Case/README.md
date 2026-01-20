# Maritime Use Case 

The [Crexdata Maritime Use Case](https://crexdata.eu/maritime/) is advancing maritime safety and operational efficiency through innovative technologies, by leveraging real-time data, IoT devices, and predictive analytics, the project addresses critical challenges in:
- Collision Forecasting and Rerouting
- Hazardous Weather Rerouting

The [workflows](./Workflows/) folder contains  the following workflow specifications:
  -  `crexdata-MMDF-Maritime Use Case-FULL.rmp` : The main maritime use case workflow that takes advantage of the MMDF toolbox to combine ASV telemetry with Automatic Indentification System (AIS) data and static files through their kafka metatdata records. 
  - `crexdata-maritime - sea-trials-sent-mission.rmp` : Api worklow wrapping up decicated REST-API calls for assigning mission to ASVs on field
  - `crexdata-maritime - sea-trials-start-vessels.rmp` : Api worklow wrapping up decicated REST-API calls for sending "start mission" command to ASVs on field
  - `crexdata-maritime - sea-trials-stop-vessels.rmp` : Api worklow wrapping up decicated REST-API calls for sending "stop mission" command to ASVs on field


For details regarding the Multi-Modal data fusion toolboox look on the relevant [MMDF toolbox](../../CREXDATA%20System%20Components/Multi-Modal%20Data%20Fusion%20Toolbox/) section in System Components.
