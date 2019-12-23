import json
with open('../datasets/markable/annotations/markable_test.json') as json_file:
    data = json.load(json_file)

import ipdb; ipdb.set_trace()
print(len(data))