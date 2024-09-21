addCommandAlias("build", ";compile;test;assembly")
addCommandAlias("rebuild", ";clean;build")

lazy val root = project.in(file("."))
  .enablePlugins(AssemblyPlugin)
  .settings(logLevel in Test := Level.Info)
  .settings(logLevel in Compile := Level.Error)
  .settings(libraryDependencies ++= Dependencies.Logging)
  .settings(libraryDependencies ++= Dependencies.Testing)
  .settings(libraryDependencies ++= Dependencies.Data)
  .settings(libraryDependencies ++= Dependencies.Math)
  .settings(libraryDependencies ++= Dependencies.Tools)
  .settings(libraryDependencies ++= Dependencies.Kafka)

