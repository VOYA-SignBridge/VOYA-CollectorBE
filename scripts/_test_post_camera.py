import json
import urllib.request

url = 'http://localhost:8000/upload/camera'
payload = {
    'user': 'testuser',
    'label': 'testlabel',
    'dialect': '',
    'session_id': 'sess123',
    'frames': [
        {
            'timestamp': 0,
            'landmarks': {
                'pose': [
                    {'x': 0.1, 'y': 0.2, 'z': 0.0, 'visibility': 1.0}
                ],
                'face': [],
                'left_hand': [],
                'right_hand': []
            }
        }
    ]
}

data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        resp = r.read().decode('utf-8')
        print('STATUS', r.getcode())
        print(resp)
except Exception as e:
    print('ERROR', e)
