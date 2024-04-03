import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

line_titles = ['teacher_index', 'loss', 'generalisation_error_0', 'log_generalisation_error_0', 'generalisation_error_1', 'log_generalisation_error_1', 'student_head_0_weight_0', 'student_head_0_weight_1', 'ambient_student_self_overlap_0_0', 'ambient_student_self_overlap_0_1', 'ambient_student_self_overlap_1_0', 'ambient_student_self_overlap_1_1', 'aggregate_student_self_overlap_0_0_0', 'aggregate_student_self_overlap_0_0_1', 'aggregate_student_self_overlap_0_1_0', 'aggregate_student_self_overlap_0_1_1', 'aggregate_student_self_overlap_1_0_0', 'aggregate_student_self_overlap_1_0_1', 'aggregate_student_self_overlap_1_1_0', 'aggregate_student_self_overlap_1_1_1', 'latent_student_self_overlap_0_0_0', 'latent_student_self_overlap_0_0_1', 'latent_student_self_overlap_0_1_0', 'latent_student_self_overlap_0_1_1', 'latent_student_self_overlap_1_0_0', 'latent_student_self_overlap_1_0_1', 'latent_student_self_overlap_1_1_0', 'latent_student_self_overlap_1_1_1', 'student_teacher_0_overlap_0_0', 'student_teacher_0_overlap_0_1', 'student_teacher_0_overlap_1_0', 'student_teacher_0_overlap_1_1', 'student_teacher_1_overlap_0_0', 'student_teacher_1_overlap_0_1', 'student_teacher_1_overlap_1_0', 'student_teacher_1_overlap_1_1', 'rotated_student_teacher_0_overlap_0_0', 'rotated_student_teacher_0_overlap_0_1', 'rotated_student_teacher_0_overlap_1_0', 'rotated_student_teacher_0_overlap_1_1', 'rotated_student_teacher_1_overlap_0_0', 'rotated_student_teacher_1_overlap_0_1', 'rotated_student_teacher_1_overlap_1_0', 'rotated_student_teacher_1_overlap_1_1']

overlaps = np.arange(0,1.05,0.05)
runs_data = np.loadtxt('grouped_runs.txt').reshape(201, 126, 21) #.reshape(1001, 122, 20) # time, metric, runs
num_runs = runs_data.shape[2]
chosen_metric = 5 # 2 for gen error of task 1, 3 for log of 2, 4 for gen error of task 2, 5 for log of 5
log_x = False

colors = plt.cm.viridis(np.linspace(0,1,num_runs))

for run_idx in range(num_runs):
    if log_x:
        plt.plot(np.log(np.arange(1,len(runs_data)+1)), runs_data[:,chosen_metric,run_idx], color=colors[run_idx])
    else:
        plt.plot(np.arange(len(runs_data)), runs_data[:,chosen_metric,run_idx], color=colors[run_idx])
plt.xlabel('time')
plt.ylabel(line_titles[chosen_metric])
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
plt.colorbar(sm)
plt.grid()
plt.savefig('runs.png',dpi=400)
