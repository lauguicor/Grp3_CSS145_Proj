wget https://download.bell-sw.com/java/11.0.16+8/bellsoft-jre11.0.16+8-linux-amd64.tar.gz
tar -xzf bellsoft-jre11.0.16+8-linux-amd64.tar.gz

# Set JAVA_HOME to point to the extracted Java directory
export JAVA_HOME=$PWD/jre11.0.16
export PATH=$JAVA_HOME/bin:$PATH