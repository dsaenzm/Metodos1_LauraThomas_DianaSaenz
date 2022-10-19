function help(){

	echo "---Debe incluir tres parametros---"
}

if ! [ $# -eq 3 ]; then
	echo "Corriendo programa"
	help
	exit 1
fi
