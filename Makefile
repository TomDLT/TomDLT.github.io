
# does not work, must be done manually
grip:
	export GRIPURL=$PWD
	echo "CACHE_DIRECTORY = '$PWD/blog/css'" >> ~/.grip/settings.py
	for f in blog/*.md; do grip --export $f --no-inline; done
	for f in blog/*.html; do sed -i $f -e 's@'"$GRIPURL"'/@@g'; done
