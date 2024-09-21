#!/bin/bash
CONF_FILE=$1

TOPICS=$(grep "^topic\." $CONF_FILE | sed 's/topic\.[a-zA-Z]* = \(.*\)/\1/g')

for TOPIC in $TOPICS;
do
 echo "$TOPIC"
 docker compose exec broker kafka-topics --if-exists --delete \
    --topic "$TOPIC" \
    --bootstrap-server localhost:9092
 AVRO_FILE="../schemas/${TOPIC}.avsc"
 if [[ -f "$AVRO_FILE" ]]; then
  AVRO_FILE="../schemas/${TOPIC}.avsc"
  curl -X DELETE http://localhost:8081/subjects/"${TOPIC}"-value?permanent=true -o /dev/null --silent
 fi
done
