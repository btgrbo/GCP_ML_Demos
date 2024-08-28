ENDPOINT_ID="4034561161100787712"
PROJECT_ID="738673379845"
INPUT_DATA_FILE="test_instances.json"


curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
"https://europe-west3-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/europe-west3/endpoints/${ENDPOINT_ID}:predict" \
-d "@${INPUT_DATA_FILE}"