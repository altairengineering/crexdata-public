<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" classificationMin="0" classificationMax="2" opacity="1">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">
          <!-- Values are water depth in meters. NoData is expected to be -9999.0 -->
          <item alpha="255" value="0.0" label="0.0 m" color="#f7fbff"/>
          <item alpha="255" value="0.1" label="0.1 m" color="#deebf7"/>
          <item alpha="255" value="0.3" label="0.3 m" color="#c6dbef"/>
          <item alpha="255" value="0.5" label="0.5 m" color="#9ecae1"/>
          <item alpha="255" value="1.0" label="1.0 m" color="#6baed6"/>
          <item alpha="255" value="2.0" label="2.0 m" color="#2171b5"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation grayscaleMode="0" colorizeOn="0" saturation="0" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
</qgis>

