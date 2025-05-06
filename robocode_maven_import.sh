#!/bin/bash

LIBS_DIR="./robocode_libs"

GROUP_ID="robocode.local"

VERSION="1.0.0"

for JAR in "$LIBS_DIR"/*.jar; do
    FILENAME=$(basename -- "$JAR")
    ARTIFACT_ID="${FILENAME%.jar}"

    echo "Installing $FILENAME as $GROUP_ID:$ARTIFACT_ID:$VERSION"

    mvn install:install-file \
        -Dfile="$JAR" \
        -DgroupId="$GROUP_ID" \
        -DartifactId="$ARTIFACT_ID" \
        -Dversion="$VERSION" \
        -Dpackaging=jar
done