echo 'Starting Minio server...'
minio server minio/ &
mc alias set mys3 http://127.0.0.1:9000 minioadmin minioadmin

echo 'Creating bucket...'
mc mb mys3/mlops-hw

echo "Initializing dvc..."
git init
dvc init -f
# dvc remote remove hatefuls3
dvc remote add -d mys3 s3://mlops-hw -f
dvc remote modify mys3 endpointurl http://127.0.0.1:9000
dvc remote modify mys3 access_key_id minioadmin
dvc remote modify mys3 secret_access_key minioadmin

echo "Running api..."
python src/api.py &