#!/bin/sh

job_id=`printf "%02d" ${SLURM_PROCID}`

if [ -z "$1" ]
then
    echo ./COVID19 ./output_R${job_id}/config.xml 1> ./output_R${job_id}/stdout.log 2> ./output_R${job_id}/stderr.log
    exec ./COVID19 ./output_R${job_id}/config.xml 1> ./output_R${job_id}/stdout.log 2> ./output_R${job_id}/stderr.log
else
    echo ./COVID19 ./$1/output_R${job_id}/config.xml 1> ./$1/output_R${job_id}/stdout.log 2> ./$1/output_R${job_id}/stderr.log
    exec ./COVID19 ./$1/output_R${job_id}/config.xml 1> ./$1/output_R${job_id}/stdout.log 2> ./$1/output_R${job_id}/stderr.log
fi
