
from inference_sdk import InferenceHTTPClient

# 2. Connect to your local server
client = InferenceHTTPClient(
    api_url="http://localhost:9001", # Local server address
    api_key="rf_7ah84tjCbPuEE1Oj3beo"
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="fiboxvision",
    workflow_id="sam-2",
    images={
        "image": "screenshots/screenshot_20251108_210827.jpg" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)
