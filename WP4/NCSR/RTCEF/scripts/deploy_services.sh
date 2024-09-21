#!/bin/bash
CONF=../configs/port/conf_0.ini

PROJECT_PATH=$(cat ${CONF} | awk '!/^;/'| grep "ProjectPath = "| sed 's/.* = \(.*\)/\1/g' |sed 's/\"//g')
WAYEB_JAR=$(cat ${CONF} | awk '!/^;/'| grep "model.wayebJar = "| sed 's/.* = \(.*\)/\1/g' | sed 's/${.*}\(.*\)/\1/g'| sed 's/\"//g')
WAYEB_PATH=$(echo $PROJECT_PATH$WAYEB_JAR)
PIDS_FILE=./pids/pid
PRINT_LOGS=./execution_logs/

if [[ $1 == "start_offline" ]]; then

  java -jar ${PROJECT_PATH}${WAYEB_JAR} service --pythonPort 25335 --javaPort 25336 > ${PRINT_LOGS}wayeb_service_print_log.log 2>&1 & PID_WSE=("$!")
  echo ${PID_WSE} > ${PIDS_FILE}_wse.txt

  java -jar ${PROJECT_PATH}${WAYEB_JAR} server > ${PRINT_LOGS}wayeb_server_print_log.log 2>&1 & PID_WSS=("$!")
  echo ${PID_WSS} > ${PIDS_FILE}_wss.txt

#  python ../main.py -s observer -c ${CONF}  > ${PRINT_LOGS}observer_print_log.log 2>&1 & PID_OB=("$!")
#  echo ${PID_OB} > ${PIDS_FILE}_ob.txt

  python ../main.py -s optimiser -c ${CONF} > ${PRINT_LOGS}optimiser_print_log.log 2>&1 & PID_OPT=("$!")
  echo ${PID_OPT} > ${PIDS_FILE}_opt.txt

  python ../main.py -s factory -c ${CONF} > ${PRINT_LOGS}factory_print_log.log 2>&1 & PID_FAC=("$!")
  echo ${PID_FAC} > ${PIDS_FILE}_fac.txt

  python ../main.py -s collector -c ${CONF} > ${PRINT_LOGS}collector_print_log.log 2>&1 & PID_COL+=("$!")
  echo ${PID_COL} > ${PIDS_FILE}_col.txt

  python ../main.py -s engine -c ${CONF} > ${PRINT_LOGS}engine_print_log.log 2>&1 & PID_EN+=("$!")
  echo ${PID_EN} > ${PIDS_FILE}_en.txt

  python ../main.py -s reader -c ${CONF} > ${PRINT_LOGS}reader_print_log.log 2>&1 & PID_READ+=("$!")
  echo ${PID_READ} > ${PIDS_FILE}_read.txt

  python ../main.py -s replayer -c ${CONF} > ${PRINT_LOGS}replayer_print_log.log 2>&1 & PID_REP+=("$!")
  echo ${PID_REP} > ${PIDS_FILE}_rep.txt

  echo "Wayeb engine side: "${PID_WSE}
  echo "Wayeb server side: "${PID_WSS}
  echo "Optimiser service: "${PID_OPT}
  echo "Factory service: "${PID_FAC}
  echo "Collector service: "${PID_COL}
  echo "Engine service: "${PID_EN}
  echo "Reader service: "${PID_READ}
  echo "Replayer service: "${PID_REP}
fi

if [[ $1 == "start" ]]; then

  java -jar ${PROJECT_PATH}${WAYEB_JAR} service --pythonPort 25335 --javaPort 25336 > ${PRINT_LOGS}wayeb_service_print_log.log 2>&1 & PID_WSE=("$!")
  echo ${PID_WSE} > ${PIDS_FILE}_wse.txt

  java -jar ${PROJECT_PATH}${WAYEB_JAR} server > ${PRINT_LOGS}wayeb_server_print_log.log 2>&1 & PID_WSS=("$!")
  echo ${PID_WSS} > ${PIDS_FILE}_wss.txt

  python ../main.py -s observer -c ${CONF}  > ${PRINT_LOGS}observer_print_log.log 2>&1 & PID_OB=("$!")
  echo ${PID_OB} > ${PIDS_FILE}_ob.txt

  python ../main.py -s optimiser -c ${CONF} > ${PRINT_LOGS}optimiser_print_log.log 2>&1 & PID_OPT=("$!")
  echo ${PID_OPT} > ${PIDS_FILE}_opt.txt

  python ../main.py -s factory -c ${CONF} > ${PRINT_LOGS}factory_print_log.log 2>&1 & PID_FAC=("$!")
  echo ${PID_FAC} > ${PIDS_FILE}_fac.txt

  python ../main.py -s collector -c ${CONF} > ${PRINT_LOGS}collector_print_log.log 2>&1 & PID_COL+=("$!")
  echo ${PID_COL} > ${PIDS_FILE}_col.txt

  python ../main.py -s engine -c ${CONF} > ${PRINT_LOGS}engine_print_log.log 2>&1 & PID_EN+=("$!")
  echo ${PID_EN} > ${PIDS_FILE}_en.txt

  python ../main.py -s reader -c ${CONF} > ${PRINT_LOGS}reader_print_log.log 2>&1 & PID_READ+=("$!")
  echo ${PID_READ} > ${PIDS_FILE}_read.txt

  python ../main.py -s replayer -c ${CONF} > ${PRINT_LOGS}replayer_print_log.log 2>&1 & PID_REP+=("$!")
  echo ${PID_REP} > ${PIDS_FILE}_rep.txt

  echo "Wayeb engine side: "${PID_WSE}
  echo "Wayeb server side: "${PID_WSS}
  echo "Observer service: "${PID_OB}
  echo "Optimiser service: "${PID_OPT}
  echo "Factory service: "${PID_FAC}
  echo "Collector service: "${PID_COL}
  echo "Engine service: "${PID_EN}
  echo "Reader service: "${PID_READ}
  echo "Replayer service: "${PID_REP}
fi



if [[ $1 == "status" ]]; then
  PID_WSE=$(cat ${PIDS_FILE}_wse.txt 2> /dev/null)
  PID_WSS=$(cat ${PIDS_FILE}_wss.txt 2> /dev/null)
  PID_OB=$(cat ${PIDS_FILE}_ob.txt 2> /dev/null)
  PID_OPT=$(cat ${PIDS_FILE}_opt.txt 2> /dev/null)
  PID_FAC=$(cat ${PIDS_FILE}_fac.txt 2> /dev/null)
  PID_COL=$(cat ${PIDS_FILE}_col.txt 2> /dev/null)
  PID_EN=$(cat ${PIDS_FILE}_en.txt 2> /dev/null)
  PID_READ=$(cat ${PIDS_FILE}_read.txt 2> /dev/null)
  PID_REP=$(cat ${PIDS_FILE}_rep.txt 2> /dev/null)

  if  ps -p ${PID_WSE} > /dev/null 2>&1 ; then
    echo "Wayeb engine side: on"
  else
    echo "Wayeb engine side: off"
  fi
  if ps -p ${PID_WSS} > /dev/null 2>&1 ; then
    echo "Wayeb server side: on"
  else
    echo "Wayeb server side: off"
  fi
  if  ps -p ${PID_OB} > /dev/null 2>&1  ; then
    echo "Service observer: on"
  else
    echo "Service observer: off"
  fi
  if  ps -p ${PID_OPT} > /dev/null 2>&1 ; then
    echo "Service optimiser: on"
  else
    echo "Service optimiser: off"
  fi
  if  ps -p ${PID_FAC} > /dev/null 2>&1 ; then
      echo "Service factory: on"
  else
      echo "Service factory: off"
  fi
  if  ps -p ${PID_COL} > /dev/null 2>&1 ; then
      echo "Service collector: on"
  else
      echo "Service collector: off"
  fi
  if  ps -p ${PID_EN} > /dev/null 2>&1 ; then
        echo "Service engine: on"
  else
        echo "Service engine: off"
  fi
  if  ps -p ${PID_READ} > /dev/null 2>&1 ; then
          echo "Service reader: on"
  else
          echo "Service reader: off"
  fi
  if  ps -p ${PID_REP} > /dev/null 2>&1 ; then
            echo "Service replayer: on"
  else
            echo "Service replayer: off"
  fi
fi

if [[ $1 == "kill" ]]; then
  for PID in $(cat ${PIDS_FILE}_*); do
    echo "Killing ${PID}..."
    kill ${PID}
  done
fi

if [[ $1 == "clean" ]]; then
  ./clean_topics.sh ${CONF}
  rm -f ${PIDS_FILE}_*
  rm -f ../data/assembled/assembled_*
  rm -f ../data/collected/dt_bucket_*
  rm -f ../data/reader/reader_*.txt
  rm -f ../data/saved_models/forecasting/wayeb_*.model
  rm -f ../data/saved_models/forecasting/wayeb_*.model.wtd
  rm -f ./print_log.log
  rm -f ${PRINT_LOGS}/*.log
fi

if [[ $1 == "init" ]]; then
  ./create_topics.sh ${CONF}
fi