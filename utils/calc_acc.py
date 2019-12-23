import pickle
import os 
import sys
import numpy as np

db, context = sys.argv[1], sys.argv[2]


# import pdb; pdb.set_trace()
with open(os.path.join('./output/' + db, context, 'predictions.pkl'), 'rb') as f:
    predictions = pickle.load(f)

with open(os.path.join('./output/' + db, context, 'target.pkl'), 'rb') as f:
    targets = pickle.load(f)


acc = np.mean(np.array(predictions[0])==np.array(targets[0]))
print('accuracy is {}'.format(acc))



# # import pdb; pdb.set_trace()
# with open(os.path.join('./output_context/' + db, context, 'predictions.pkl'), 'rb') as f:
#     predictions = pickle.load(f)

# with open(os.path.join('./output_context/' + db, context, 'target.pkl'), 'rb') as f:
#     targets = pickle.load(f)


# acc = np.mean(np.array(predictions[0])==np.array(targets[0]))
# print('accuracy is {}'.format(acc))
