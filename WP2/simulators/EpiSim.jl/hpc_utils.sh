#!/bin/bash


# BSC login nodes

# WIFI-enabled login node
WIFI_NODE="glogin4"

LOGIN_NODES=("glogin1" "glogin2" "glogin3")
LOGIN_NODES+=($WIFI_NODE)

in_hpc_bsc() {
	for node in "${LOGIN_NODES[@]}"; do
		if [[ $(hostname) == "$node" ]]; then
			return 0
		fi
	done
	return 1
}

# useful to know if we can install dependencies in this env...
in_hpc_wifi() {
	[[ $(hostname) == "$WIFI_NODE" ]]
	return $?
}

