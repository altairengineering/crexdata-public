#!/bin/bash
CONF_FILE=$1

TOPICS=$(grep "^topic\." $CONF_FILE | sed 's/topic\.[a-zA-Z]* = \(.*\)/\1/g')

for TOPIC in $TOPICS;
do
 echo "$TOPIC"
 docker compose exec broker kafka-topics --create \
    --topic "$TOPIC" \
    --bootstrap-server localhost:9092 \
    --replication-factor 1 \
    --partitions 1
 AVRO_FILE_VALUE="../schemas/${TOPIC}-value.avsc"
 AVRO_FILE_KEY="../schemas/${TOPIC}-key.avsc"
 if [[ -f "$AVRO_FILE_VALUE" ]]; then
  AVSCHEMA=$(jq tostring "$AVRO_FILE_VALUE")
  curl -XPOST -H "Content-Type:application/json" -d"{\"schema\":$AVSCHEMA}" http://localhost:8081/subjects/"$TOPIC"-value/versions -o /dev/null --silent
 fi

 if [[ -f "$AVRO_FILE_KEY" ]]; then
  AVSCHEMA=$(jq tostring "$AVRO_FILE_KEY")
  curl -XPOST -H "Content-Type:application/json" -d"{\"schema\":$AVSCHEMA}" http://localhost:8081/subjects/"$TOPIC"-key/versions -o /dev/null --silent
 fi
done
