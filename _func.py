# Public libraries:
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def file_reader(path):
    char_map = {"[": "", " ": "", "]": "", "\n": ""}
    with open(Path(path), "r") as file:
        out = []
        for line in file:
            l = list(filter(None, line.translate(str.maketrans(char_map)).split(";")))
            out.append([list(map(int, list(filter(None, e.split(","))))) for e in l])
    return out


def file_writer(results, path):
    with open(Path(path), "w") as file:
        for result in results:
            line = ""
            for task in result:
                e_str = ""
                for e in task:
                    e_str += str(e) + ","
                line += f"[{e_str}];"
            file.write(line + "\n")

def file_writer2(results, path):
    with open(Path(path), "w") as file:
        for idx , result in enumerate(results):
            if idx == 3:
                continue
            line = ""
            result = result.astype(np.int32)
            e_str = ""
            for e in result: e_str += str(e) + ","
            line += f"[{e_str}];"
            file.write(line)

def save_figs2(examples, results, title, path, time_limit=40):
    i = 0
    y_min, y_max = 5, 45
    T,C,D,R = examples[0]
    result = results
    missed_valuess = results[4]

    N = len(T)
    y_label = []
    D = T
    for j in range(N):
        y_label.append(f"Task {j+1}\n(C = {C[j]})")
    fig, ax = plt.subplots()

    lines = []
    for k in range(N):
        lines.append([[], []])
    for j in range(time_limit + 1):
        for k in range(N):
            if j % T[k] == 0:
                lines[k][0].append(j)
            if j % T[k] == D[k]:
                lines[k][1].append(j)

    y_heights = []
    for k in range(N):
        y_heights.append((y_max - y_min) * (N - k) / (N + 1) + y_min)

    tasks = []
    for k in range(N):
        tasks.append([[], []])
    
    for j in range(time_limit):
        for k in range(N):
            if result[k][j] == 1:
                if result[N][j] == 1:
                    tasks[k][1].append((j, 1))
                else:
                    tasks[k][0].append((j, 1))
    for y in y_heights:
        y_index = y_heights.index(y)
        ax.broken_barh(
            tasks[y_index][0], (y - 2.5, 5), facecolors="tab:green"
        )
        ax.broken_barh(
            tasks[y_index][1], (y - 2.5, 5), facecolors="tab:blue"
        )
    missed_all = []
    for k in range(N):
        missed_all.append([[], []])
    
    for j in range(time_limit):
        for k in range(N):
            if missed_valuess[j] == 1 and results[k][j] == 1:
                missed_all[k][0].append((j, 1))
    
    for y in y_heights:
        y_index = y_heights.index(y)
        ax.broken_barh(
            missed_all[y_index][0], (y - 2.5, 5), facecolors="tab:orange"
        )

    for y in y_heights:
        y_index = y_heights.index(y)
        for line in lines[y_index][0]:
            plt.arrow(line, y, 0, 3, linewidth=3, color="black")
        for line in lines[y_index][1]:
            plt.arrow(line, y, 0, -3, linewidth=3, color="red")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-1, time_limit + 1)
    plt.title(title + f" Scheduling for Example {i+1}", fontsize=28)
    ax.set_xlabel(f"Real-Time clock", fontsize=18)

    ax.set_yticks(y_heights, labels=y_label, fontsize=18, va="center")
    grid_points = range(time_limit + 1)
    ax.xaxis.set_ticks(grid_points)
    ax.grid(color="blue", axis="y", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.grid(
        color="blue",
        axis="x",
        linestyle="-",
        linewidth=0.5,
        alpha=0.8,
    )
    figure = plt.gcf()

    # plt.show()

    figure.set_size_inches(16, 9)

    plt.savefig(path + f"Figure_{i+1}_{title}.jpeg")

    i += 1

def save_figs(examples, results, title, path, time_limit=40):
    # Your code goes here.
    i = 0
    y_min, y_max = 5, 45
    for result in results:
        if title == "RMP":
            result = results
            T, C = examples
            N = len(T)
            y_label = []
            for j in range(N):
                if j != 2:
                    y_label.append(f"Task {j+1}\n(C = {C[j]})")
                else:
                    y_label.append(f"Aperiodic\nTask")
            fig, ax = plt.subplots()

            lines = []
            for k in range(N):
                lines.append([[], []])

            interrupts = result[N + 1]
            D = []
            for j in range(len(interrupts[0])):
                D.append(interrupts[0][j] + interrupts[1][j])

            for j in range(time_limit + 1):
                for k in range(N):
                    if j % T[k] == 0:
                        lines[k][0].append(j)

            y_heights = []
            for k in range(N):
                y_heights.append((y_max - y_min) * (N - k) / (N + 1) + y_min)

            tasks = []
            for k in range(N):
                tasks.append([[], []])

            for j in range(time_limit):
                for k in range(N):
                    if result[k][j] == 1:
                        if result[N][j] == 1:
                            tasks[k][1].append((j, 1))
                        else:
                            tasks[k][0].append((j, 1))

            for y in y_heights:
                y_index = y_heights.index(y)
                if y_index != 2:
                    ax.broken_barh(
                        tasks[y_index][0], (y - 2.5, 5), facecolors="tab:green"
                    )
                    ax.broken_barh(
                        tasks[y_index][1], (y - 2.5, 5), facecolors="tab:orange"
                    )
                else:
                    ax.broken_barh(
                        tasks[y_index][0], (y - 2.5, 5), facecolors="tab:blue"
                    )
                    ax.broken_barh(
                        tasks[y_index][1], (y - 2.5, 5), facecolors="tab:orange"
                    )
            # plt.arrow(x, y, dx, dy)
            for y in y_heights:
                y_index = y_heights.index(y)
                for line in lines[y_index][0]:
                    plt.arrow(line, y, 0, 3, linewidth=3, color="black")
                for line in lines[y_index][1]:
                    plt.arrow(line, y, 0, -3, linewidth=3, color="red")

            for j in range(2):
                for line in interrupts[0]:
                    plt.arrow(line, y_heights[2], 0, 3, linewidth=3, color="blue")
                for line in D:
                    plt.arrow(line, y_heights[2], 0, -3, linewidth=3, color="red")

            ax.set_ylim(y_min, y_max)
            ax.set_xlim(-1, time_limit + 1)
            plt.title(title + f" Scheduling for Example {i+1}", fontsize=28)
            ax.set_xlabel(f"Real-Time clock", fontsize=18)

            ax.set_yticks(y_heights, labels=y_label, fontsize=18, va="center")
            grid_points = range(time_limit + 1)
            ax.xaxis.set_ticks(grid_points)
            ax.grid(color="blue", axis="y", linestyle="-", linewidth=0.8, alpha=0.8)
            ax.grid(
                color="blue",
                axis="x",
                linestyle="-",
                linewidth=0.5,
                alpha=0.8,
            )
            figure = plt.gcf()

            # plt.show()

            figure.set_size_inches(16, 9)
            plt.savefig(path + f"Figure_{title}.jpeg")

            break
        else:
            T, C, D = examples[i]
            N = len(T)
            if title == "AP":
                D = T

                y_label = []
                for j in range(N + 1):
                    if j == 0:
                        y_label.append(f"Aperiodic\nTask")
                    else:
                        y_label.append(f"Task {j}\n(C = {C[j-1]})")
                fig, ax = plt.subplots()

                interrupts = []
                for j in range(time_limit):
                    if result[N][j] == 1:
                        interrupts.append((j, 1))

                lines = []
                for k in range(N):
                    lines.append([[], []])

                for j in range(time_limit + 1):
                    for k in range(N):
                        if j % T[k] == 0:
                            lines[k][0].append(j)
                        if j % T[k] == D[k]:
                            lines[k][1].append(j)

                y_heights = []
                for k in range(N):
                    y_heights.append((y_max - y_min) * (k + 1) / (N + 2) + y_min)
                y_heights.sort(reverse=True)

                tasks = []
                for k in range(N):
                    tasks.append([[], []])

                for j in range(time_limit):
                    for k in range(N):
                        if result[k][j] == 1:
                            if result[N + 1][j] == 1:
                                tasks[k][1].append((j, 1))
                            else:
                                tasks[k][0].append((j, 1))

                for y in y_heights:
                    y_index = y_heights.index(y)
                    ax.broken_barh(
                        tasks[y_index][0], (y - 2.5, 5), facecolors="tab:green"
                    )
                    ax.broken_barh(
                        tasks[y_index][1], (y - 2.5, 5), facecolors="tab:orange"
                    )

                # plt.arrow(x, y, dx, dy)
                for y in y_heights:
                    y_index = y_heights.index(y)
                    for line in lines[y_index][0]:
                        plt.arrow(line, y, 0, 3, linewidth=3, color="black")
                    for line in lines[y_index][1]:
                        plt.arrow(line, y, 0, -3, linewidth=3, color="red")

                y = (y_max - y_min) * (N + 1) / (N + 2) + y_min
                ax.broken_barh(interrupts, (y - 2.5, 5), facecolors="tab:red")
                y_heights.append(y)
                y_heights.sort(reverse=True)
            else:
                if title == "RM":
                    D = T
                y_label = []
                for j in range(N):
                    y_label.append(f"Task {j+1}\n(C = {C[j]})")
                fig, ax = plt.subplots()

                lines = []
                for k in range(N):
                    lines.append([[], []])

                for j in range(time_limit + 1):
                    for k in range(N):
                        if j % T[k] == 0:
                            lines[k][0].append(j)
                        if j % T[k] == D[k]:
                            lines[k][1].append(j)

                y_heights = []
                for k in range(N):
                    y_heights.append((y_max - y_min) * (N - k) / (N + 1) + y_min)

                tasks = []
                for k in range(N):
                    tasks.append([[], []])

                for j in range(time_limit):
                    for k in range(N):
                        if result[k][j] == 1:
                            if result[N][j] == 1:
                                tasks[k][1].append((j, 1))
                            else:
                                tasks[k][0].append((j, 1))

                for y in y_heights:
                    y_index = y_heights.index(y)
                    ax.broken_barh(
                        tasks[y_index][0], (y - 2.5, 5), facecolors="tab:green"
                    )
                    ax.broken_barh(
                        tasks[y_index][1], (y - 2.5, 5), facecolors="tab:orange"
                    )

                # plt.arrow(x, y, dx, dy)
                for y in y_heights:
                    y_index = y_heights.index(y)
                    for line in lines[y_index][0]:
                        plt.arrow(line, y, 0, 3, linewidth=3, color="black")
                    for line in lines[y_index][1]:
                        plt.arrow(line, y, 0, -3, linewidth=3, color="red")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-1, time_limit + 1)
        plt.title(title + f" Scheduling for Example {i+1}", fontsize=28)
        ax.set_xlabel(f"Real-Time clock", fontsize=18)

        ax.set_yticks(y_heights, labels=y_label, fontsize=18, va="center")
        grid_points = range(time_limit + 1)
        ax.xaxis.set_ticks(grid_points)
        ax.grid(color="blue", axis="y", linestyle="-", linewidth=0.8, alpha=0.8)
        ax.grid(
            color="blue",
            axis="x",
            linestyle="-",
            linewidth=0.5,
            alpha=0.8,
        )
        figure = plt.gcf()

        # plt.show()

        figure.set_size_inches(16, 9)
        plt.savefig(path + f"Figure_{i+1}_{title}.jpeg")

        i += 1


def rm_scheduler(examples, time_limit=40):
    results = []
    for exp in examples:
        T, C = exp[0], exp[1]
        # your code goes here :: Start
        idx = 0
        missed = [0] * time_limit
        result = []
        N = len(T)  # N = 3
        arrive_flag = [0] * N
        job_count = [0] * N
        deadlines = [0] * N
        for j in range(N):
            result.append([0] * time_limit)
        for i in range(time_limit):
            T_priority = []
            for j in range(N):
                if i % T[j] == 0:
                    arrive_flag[j] += 1
                    deadlines[j] = T[j]
                if arrive_flag[j] > 0:
                    T_priority.append(T[j])

            T_priority.sort()
            if len(T_priority) > 0:
                for k in range(N):
                    if T[idx] == T_priority[0] and arrive_flag[idx] > 0:
                        j = idx
                        break
                    if T[k] == T_priority[0] and arrive_flag[k] > 0:
                        j = k
                        break
                idx = j
                result[j][i] = 1
                job_count[j] += 1
                if deadlines[j] <= 0 or arrive_flag[j] > 1:
                    missed[i] = 1

            for j in range(N):
                if job_count[j] == C[j]:
                    job_count[j] = 0
                    arrive_flag[j] -= 1
                    if arrive_flag[j] == 0:
                        deadlines[j] = 0
                if arrive_flag[j] > 0:
                    deadlines[j] -= 1

        # your code goes here :: End
        result.append(missed)
        results.append(result)
    return results


def dm_scheduler(examples, time_limit=40):
    results = []
    for exp in examples:
        T, C, D = exp[0], exp[1], exp[2]
        # your code goes here :: Start

        idx = 0
        missed = [0] * time_limit
        result = []
        N = len(T)
        arrive_flag = [0] * N
        job_count = [0] * N
        deadlines = [0] * N
        for j in range(N):
            result.append([0] * time_limit)
        for i in range(time_limit):
            D_prime = []
            for j in range(N):
                if i % T[j] == 0:
                    arrive_flag[j] += 1
                    deadlines[j] = D[j]
                if arrive_flag[j] > 0:
                    D_prime.append(D[j])

            D_prime.sort()
            if len(D_prime) > 0:
                for k in range(N):
                    if D[idx] == D_prime[0] and arrive_flag[idx] > 0:
                        j = idx
                        break
                    if D[k] == D_prime[0] and arrive_flag[k] > 0:
                        j = k
                        break
                idx = j
                result[j][i] = 1
                job_count[j] += 1
                if deadlines[j] <= 0 or arrive_flag[j] > 1:
                    missed[i] = 1

            for j in range(N):
                if job_count[j] == C[j]:
                    job_count[j] = 0
                    arrive_flag[j] -= 1
                    if arrive_flag[j] == 0:
                        deadlines[j] = 0
                if arrive_flag[j] > 0:
                    deadlines[j] -= 1

        # your code goes here :: End
        result.append(missed)
        results.append(result)
    return results


def ed_scheduler(examples, time_limit=40):
    results = []
    for exp in examples:
        T, C, D = exp[0], exp[1], exp[2]

        # your code goes here :: Start
        missed = [0] * time_limit
        result = []
        idx = 0
        N = len(T)
        arrive_flag = [0] * N
        job_count = [0] * N
        deadlines = [0] * N
        for j in range(N):
            result.append([0] * time_limit)
        for i in range(time_limit):
            D_prime = []
            T_priority = []
            for j in range(N):
                if i % T[j] == 0:
                    arrive_flag[j] += 1
                    deadlines[j] = D[j]
                if arrive_flag[j] > 0:
                    D_prime.append(deadlines[j] - D[j] * (arrive_flag[j] - 1))
                    T_priority.append(T[j])

            D_prime.sort()
            T_priority.sort(reverse=True)
            if len(D_prime) > 0:
                if len(D_prime) > 1:
                    if D_prime[0] == D_prime[1]:
                        if (
                            deadlines[idx] - D[idx] * (arrive_flag[idx] - 1)
                            == D_prime[0]
                            and arrive_flag[idx] > 0
                        ):
                            j = idx
                        else:
                            for k in range(N):
                                if k != idx:
                                    if (
                                        deadlines[k] - D[k] * (arrive_flag[k] - 1)
                                        == D_prime[0]
                                        and T[k] == T_priority[0]
                                        and arrive_flag[k] > 0
                                    ):
                                        j = k
                                        break
                    else:
                        for k in range(N):
                            if (
                                deadlines[k] - D[k] * (arrive_flag[k] - 1) == D_prime[0]
                                and arrive_flag[k] > 0
                            ):
                                j = k
                                break
                else:
                    for k in range(N):
                        if (
                            deadlines[k] - D[k] * (arrive_flag[k] - 1) == D_prime[0]
                            and arrive_flag[k] > 0
                        ):
                            j = k
                            break

                idx = j
                result[j][i] = 1
                job_count[j] += 1
                if deadlines[j] <= 0 or arrive_flag[j] > 1:
                    missed[i] = 1

            for j in range(N):
                if job_count[j] == C[j]:
                    job_count[j] = 0
                    arrive_flag[j] -= 1
                    if arrive_flag[j] == 0:
                        deadlines[j] = 0
                if arrive_flag[j] > 0:
                    deadlines[j] -= 1
        # your code goes here :: End
        result.append(missed)
        results.append(result)
    return results


def ap_rm_scheduler(examples, ap_task_time, ap_task_jobs, time_limit=40):
    results = []
    for exp in examples:
        T, C = exp[0], exp[1]
        # your code goes here :: Start
        ap_task_count = 0
        ap_arrive_flag = 0
        idx = 0
        missed = [0] * time_limit
        result = []
        interrupt = [0] * time_limit

        N = len(T)
        arrive_flag = [0] * N
        job_count = [0] * N
        deadlines = [0] * N
        for j in range(N):
            result.append([0] * time_limit)
        for i in range(time_limit):
            T_priority = []
            if i == ap_task_time:
                ap_arrive_flag = 1
            for j in range(N):
                if i % T[j] == 0:
                    arrive_flag[j] += 1
                    deadlines[j] = T[j]
                if arrive_flag[j] > 0:
                    T_priority.append(T[j])

            T_priority.sort()
            if len(T_priority) > 0 or ap_arrive_flag == 1:
                for k in range(N):
                    if T[k] == T_priority[0] and arrive_flag[k] > 0:
                        j = k
                        break
                if T[idx] == T_priority[0] and arrive_flag[idx] > 0:
                    j = idx
                if ap_arrive_flag == 1:
                    j = -1
                if j >= 0:
                    idx = j
                    result[j][i] = 1
                    job_count[j] += 1
                    if arrive_flag[j] > 1:
                        missed[i] = 1
                else:
                    interrupt[i] = 1
                    ap_task_count += 1

            for j in range(N):
                if job_count[j] == C[j]:
                    job_count[j] = 0
                    arrive_flag[j] -= 1
                    deadlines[j] = 0
                if arrive_flag[j] > 0:
                    deadlines[j] -= 1
            if ap_task_count == ap_task_jobs:
                ap_arrive_flag = 0
                ap_task_count = 0
        # your code goes here :: End
        result.append(interrupt)
        result.append(missed)
        results.append(result)
    return results


def resourced_rm_without_inheritance(examples, time_limit=40):
    results = []
    inputi = []
    for exp in examples:
        all_tasks = np.array(exp, dtype=np.int32).T
        task_number = all_tasks.shape[0]
        results = np.zeros((task_number, time_limit))
        valid_misses = [0] * time_limit
        resources = [0] * time_limit
        inputi = exp.copy()

        task_priorities = [[all_tasks[i, 0], i] for i in range(task_number)]
        task_priorities.sort(key=lambda x: x[0])
        for i, _ in enumerate(task_priorities):
            task_priorities[i][0] = i + 1
        inputi = np.array(inputi)
        periods_sort = [[j * all_tasks[i, 0] for j in range(1, time_limit // all_tasks[i, 0] + 1)] for i in range(task_number)]
        periods_sort = [[0] + period_list for period_list in periods_sort]
        periods_sort = sorted([(period, index) for index, period_list in enumerate(periods_sort) for period in period_list], key=lambda x: x[0])

        las_priority_tasks = []
        for period, index in periods_sort:
            if period < time_limit:
                las_priority_tasks.append([period, all_tasks[index, 1], all_tasks[index, 2] + period, all_tasks[index, 3], index])

        inputi[1] += inputi[3] 
        last_result = rm_scheduler([inputi])
        for i in range(time_limit):
            arrival_deadline = []
            for index, task in enumerate(las_priority_tasks):
                is_resource_free = False
                if i - task[0] >= 0 and (task[1] > 0 or task[3] > 0):
                    if task[1] <= 0 and task[3] > 0:
                        is_resource_free = True
                    arrival_deadline.append((task_priorities[task[4]], task[-1], index, is_resource_free))

            if arrival_deadline:
                number_job_id = min(arrival_deadline, key=lambda x: x[0][0])
                for val in arrival_deadline:
                    if True == val[-1]:
                        number_job_id = val
                results[number_job_id[1], i] = 1
                if number_job_id[3]:
                    las_priority_tasks[number_job_id[2]][3] -= 1
                    resources[i] = 1
                else:
                    las_priority_tasks[number_job_id[2]][1] -= 1
                if i >= las_priority_tasks[number_job_id[2]][2]:
                    valid_misses[i] = 1
    results = np.vstack((results, resources))
    results = np.vstack((results, np.array(last_result[0][3]).reshape(1, -1)))
    return results

def resourced_rm_with_inheritance(exampless, time_limit = 40):
    results = []
    for exp in exampless:
        all_tasks = np.array(exp,dtype=np.int32).T
        task_number = all_tasks.shape[0]
        results = np.zeros((task_number,time_limit))
        valid_misses = [0] * time_limit
        resources = [0] * time_limit
        task_priorities = []
        for index,i in enumerate(all_tasks):
            task_priorities.append([i[0],index])
        task_priorities = sorted(task_priorities,key=lambda x : x[0])
        k = 1
        for i in range(len(task_priorities)):
            task_priorities[i][0] =  k
            k+=1
        del k
        priorities_saved = [0]*all_tasks.shape[0]
        inputii = exampless[0].copy()
        inputii = np.array(inputii)
        inputii[1] += inputii[3] 
        for i in task_priorities:
            priorities_saved[i[1]] = i[0]
        task_priorities=priorities_saved
        periods_sort=[]
        last_result = rm_scheduler([inputii])
        for i in range(task_number):
            periods_sort.append([j*all_tasks[i,0] for j in range(1,math.floor(time_limit/all_tasks[i,0])+1)])
            periods_sort[i].insert(0,0)
        periods_sort  = sorted ([(i,index) for index,j in enumerate(periods_sort) for i in j ],key=lambda x:x[0]) 
        las_priority_tasks = []
        for i in periods_sort:
            if(i[0]<40):
                las_priority_tasks.append([i[0],all_tasks[i[1],1],all_tasks[i[1],2]+i[0],all_tasks[i[1],3],i[1]])
        periods_list2 = []
        resource_is_lock = False
        blocked_task_number = -1
        i = 0
        while(i<time_limit):

            arrive_next_deadline = []
            for index,j in enumerate(las_priority_tasks):
                is_resource_free = False
                if((i - j[0] >= 0) and((j[1]>0)or(j[3]>0))):
                    if(j[1]<=0 and j[3]>0):
                        is_resource_free = True
                    arrive_next_deadline.append((task_priorities[j[4]],j[-1],index,is_resource_free))
            if(arrive_next_deadline == []):
                i+=1
                continue
            job = min(arrive_next_deadline,key=lambda x:x[0])
            if(job[3]):
                if(resource_is_lock and blocked_task_number != job[1]):
                    periods_list2.append(task_priorities[blocked_task_number])
                    task_priorities[blocked_task_number] = task_priorities[job[1]]
                    continue
            results[job[1],i] = 1
            if(i>=las_priority_tasks[job[2]][2]):
                valid_misses[i] = 1
            if(job[3]):
                resource_is_lock=True
                blocked_task_number = job[1]
                las_priority_tasks[job[2]][3] -= 1
                resources[i] = 1 
                if(las_priority_tasks[job[2]][3]<=0):
                    resource_is_lock = False
                    blocked_task_number = -1
                    if(len(periods_list2)!=0):
                        task_priorities[job[1]] = periods_list2.pop()
            else:
                las_priority_tasks[job[2]][1] -= 1
                
            i+=1
    results = np.vstack((results, resources))
    results = np.vstack((results, np.array(last_result[0][3]).reshape(1, -1)))
    return results