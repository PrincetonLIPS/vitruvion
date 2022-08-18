import json
import numpy as np
from matplotlib import pyplot as plt


def read_json(path):
    with open(path, 'r') as fh:
        data = json.load(fh)
    return data

no_noise_path = '../results/week_0322/constraints_w_noise/loss_no_noise.json'
with_noise_path = '../results/week_0322/constraints_w_noise/loss_with_noise.json'

no_noise_data = read_json(no_noise_path)
with_noise_data = read_json(with_noise_path)

def add_plot(data, label):
    if label == 'no_noise':
        jump = 6
    else:
        jump = 1
    step = [d[1] for d in data[::jump]]
    loss = [d[2] for d in data[::jump]]
    plt.plot(step, loss, label=label)

add_plot(no_noise_data, 'no_noise')
add_plot(with_noise_data, 'with_noise')

eval_x = 3840000
eval_y = 1.3
plt.scatter([eval_x], [eval_y], c='m')
plt.annotate("no_noise eval. on noisy", (eval_x, eval_y),
    xytext=(eval_x+10000, eval_y+0.04))

eval_noisy_y = 0.3
plt.scatter([eval_x], [eval_noisy_y], c='g')
plt.annotate("noisy eval. on no_noise", (eval_x, eval_noisy_y),
    xytext=(eval_x+10000, eval_noisy_y+0.04))

plt.ylim(0, 2)
plt.xlim(0, no_noise_data[-1][1])
plt.legend()
plt.xlabel('Number of examples')
plt.ylabel('Loss')
plt.savefig('constraint_curves.pdf')