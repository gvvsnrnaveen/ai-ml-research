#!/bin/sh
linux_dist=`lsb_release -a | grep "Distributor" | awk '{print tolower($3)}'`

if [ "$linux_dist" != "ubuntu" ];
then
	echo "This script is only applicable on Ubuntu OS"
	echo "Please install dependencies manually listed below"
	echo "1. armadillo"
	echo "2. libmlpack-dev"
	exit
fi

sudo apt install -y libarmadillo-dev
sudo apt install -y libmlpack-dev

