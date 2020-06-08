#!/usr/bin/env bash

cd ~/spark
./sbin/start-master.sh
./sbin/start-slave.sh spark://192.168.1.30:7077 # ip of master