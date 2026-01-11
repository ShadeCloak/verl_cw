
import os
os.environ['WANDB_API_KEY'] = 'b1f6482b72c77da12006e8e4dc84a1464d2fc34a'


import wandb
wandb.init(project="test", mode="online")
wandb.log({"test": 1})
wandb.finish()
print("✅ W&B login successful!")


# import requests
# import os
# key = os.environ['WANDB_API_KEY']
# r = requests.get('https://api.wandb.ai/graphql', 
#                  headers={'Authorization': f'Bearer {key}'},
#                  json={'query': '{viewer {username email}}'})
# print('Status:', r.status_code)
# print('Response:', r.json())
