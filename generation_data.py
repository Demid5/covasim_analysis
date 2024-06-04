from SALib.sample import saltelli, morris
import numpy as np
import covasim as cv
from math import floor
from tqdm import tqdm
from dask import delayed, compute
import multiprocessing
import location_preprocessor as lp
import pickle


def generate_synth_params():
    params_inv = [
        'enrollment_rates_by_age',
        'employment_rates_by_age',
        'school_size_distribution_by_pk',
        'school_size_distribution_by_es',
        'school_size_distribution_by_uv',
        'household_size_distribution',
        'h_contact_matrix',
        'workplace_size_counts_by_num_personnel',
        'w_contact_matrix',
        'average_student_teacher_ratio',
        'average_class_size',
        'average_teacher_teacher_degree',
        'average_student_all_staff_ratio',
        'average_additional_staff_degree']
    param2ind = dict(zip(params_inv, range(len(params_inv))))
    problem = {
        'num_vars': len(params_inv),
        'names': params_inv,
        'bounds': [
        [0, 6],
        [0, 6],
        [0, 6],
        [0, 6],
        [0, 6],
        [0, 6],
        [0, 11],
        [0, 6],
        [0, 11],
        [10, 29],
        [0, 15],
        [0, 9],
        [1, 10],
        [5, 34]]
    }
    param_values = saltelli.sample(problem, 2048)
    #param_values = morris.sample(problem, 2000, num_levels=4)
    return param_values, param2ind


def single_build_and_run(city_pars):
    try:
        pop_size = 100000
        pars = {"pop_size": pop_size, "pop_type": 'synthpops', 'n_days': 300}
        pop = lp.make_people_from_pars(**city_pars)    
        sim = cv.Sim(pars=pars, rand_seed=0, verbose=False, variants=cv.variant('wild', days=0)).init_people(prepared_pop=pop)
        sim.run()
        #print(f"end; {rand_seed}")
        return sim
    except:
        print(":((()))")
        raise

def single_build_and_run_all(epid_pars, city_pars):
    try:
        pop_size = 100000
        pop = lp.make_people_from_pars(**city_pars)    
        sim = cv.Sim(
            pars={
                "n_days": 300,
                "pop_size": pop_size,
            },
            pop_type="synthpops", variants=cv.variant(
            variant={
                "rel_beta": epid_pars['rel_beta'], 
                "rel_symp_prob": epid_pars['rel_symp_prob'], 
                "dur_asym2rec": dict(dist="lognormal_int", par1=epid_pars['dur_asym2rec'], par2=2.0),
                "dur_mild2rec": dict(dist="lognormal_int", par1=epid_pars['dur_mild2rec'], par2=2.0),
            }, days=0), 
            verbose=False  
            ).init_people(prepared_pop=pop)
        sim.run()
        return sim
    except:
        print(":((()))")
        raise

@delayed
def delayed_single_build_and_run(city_pars):
    return single_build_and_run(city_pars)

def process_using_dask(all_city_pars):
    tasks = []
    for pars in all_city_pars:
        task = delayed_single_build_and_run(pars)
        tasks.append(task)
    # Compute all tasks in parallel
    results = compute(*tasks, scheduler='processes', num_workers=70)  # Use the process scheduler
    return results

def rand_int(low, high, size):
    return np.random.randint(low=low, high=high, size=size)


def get_coefs(cm_size, start_ind, end_ind, kind_of, mult_coef=2.0):
    bounds = np.linspace(start_ind, end_ind, 3)[:-1]
    mult_coefs = np.ones(shape=(cm_size))
    for i in range(cm_size):
        if '1' in kind_of and i <= bounds[0]:
            mult_coefs[i] *= mult_coef
        if '2' in kind_of and i > bounds[0] and i <= bounds[1]:
            mult_coefs[i] *= mult_coef
        if '3' in kind_of and i > bounds[1]:
            mult_coefs[i] *= mult_coef
    return mult_coefs

def make_new_kind_dist(dist, coefs, ind_val, is_dist):
    dist = np.array(dist)
    new_dist = np.copy(dist)
    old_sum_val = 0.0
    new_sum_val = 0.0
    
    for i in range(len(dist)):
        old_sum_val += dist[i][ind_val]
        new_dist[i][ind_val] *= coefs[i]
        new_sum_val += new_dist[i][ind_val]

    # norming
    coef = new_sum_val if is_dist else old_sum_val/new_sum_val
    for i in range(len(dist)):
        new_dist[i][ind_val] /= coef

    return new_dist.tolist() 

def get_size_distribution_vars(dist, ind_val, is_dist, start_ind, end_ind):
    cm_size = len(dist)
    kinds = [
        "1", "2", "3", "12", "13", "23",  "123"
    ]
    size_distribution_vars = dict()
    if ind_val == 0:
        dist = np.array(dist).reshape(-1, 1)
    for kind_of in kinds:
        tmp_res = make_new_kind_dist(dist, get_coefs(cm_size, start_ind, end_ind, kind_of), ind_val, is_dist)
        if ind_val == 0:
            size_distribution_vars[kind_of] = (np.array(tmp_res).flatten()).tolist()
        else:
            size_distribution_vars[kind_of] = tmp_res

    return size_distribution_vars

def get_diff_kinds_contacts(contact_matrix, kind):
    age_categories = ['young', 'middle', 'old']
    age_idx_bounds = None
    if kind == 's':
        age_idx_bounds = [0, 1, 3, 5]
    elif kind == 'w':
        age_idx_bounds = [2, 5, 7, 10]
    else:
        age_idx_bounds = [0, 4, 10, 16]
    mult_coef = 2.0
    diff_kinds_contacts = dict()
    contact_matrix_idx = dict()
    for i in range(len(age_idx_bounds) - 1):
        for j in range(len(age_idx_bounds) - 1):
            contact_matrix_idx[(i, j)] = ((age_idx_bounds[i], age_idx_bounds[j]),
                                           (age_idx_bounds[i+1], age_idx_bounds[j+1]))

    for (idx_from, age_from) in enumerate(age_categories):
        for (idx_to, age_to) in enumerate(age_categories):
            cur_key = age_from + ',' + age_to
            diff_kinds_contacts[cur_key] = np.copy(contact_matrix)

            (start_i, start_j), (end_i, end_j) = contact_matrix_idx[(idx_from, idx_to)]
            #print(f"{cur_key}: (start_i, start_j), (end_i, end_j) = {(start_i, start_j), (end_i, end_j)}")
            for i in range(start_i, end_i):
                for j in range(start_j, end_j):
                    diff_kinds_contacts[cur_key][i, j] *= mult_coef
        cur_key = age_from
        diff_kinds_contacts[cur_key] = np.copy(contact_matrix)
        (start_i, start_j), (end_i, end_j) = ((age_idx_bounds[idx_from], 0), 
                                              (age_idx_bounds[idx_from + 1], age_idx_bounds[-1]))
        #print(f"{cur_key}: (start_i, start_j), (end_i, end_j) = {(start_i, start_j), (end_i, end_j)}")
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                diff_kinds_contacts[cur_key][i, j] *= mult_coef
    return diff_kinds_contacts


def ch_enrollment_rates_by_age_dist_vars(enrollment_rates_by_age_dist_vars):
    for k in enrollment_rates_by_age_dist_vars.keys():
        tmp_res = np.array(enrollment_rates_by_age_dist_vars[k])
        tmp_res[:, 1] = np.clip(tmp_res[:, 1], 0, 0.999)
        enrollment_rates_by_age_dist_vars[k] = tmp_res.tolist()


def generate_synth_data():
    param_values, param2ind = generate_synth_params()
    np.save("params_last_sobol_2.npy", param_values)

    #param_values = np.load("total_param_values_new_synth_100k.npy")

    rand_seed_n = 2
    pop_size = 100000
    # school parameters
    sp_average_student_teacher_ratio = param_values[:, param2ind['average_student_teacher_ratio']]
    sp_average_class_size = sp_average_student_teacher_ratio + param_values[:, param2ind['average_class_size']]
    sp_average_student_all_staff_ratio = sp_average_student_teacher_ratio - param_values[:, param2ind['average_student_all_staff_ratio']] # [5-35]
    
    enrollment_rates_by_age_dist_vars = get_size_distribution_vars([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.439], [4.0, 0.439], [5.0, 0.941], [6.0, 0.941], [7.0, 0.941], [8.0, 0.941], [9.0, 0.941], [10.0, 0.975], [11.0, 0.975], [12.0, 0.975], [13.0, 0.975], [14.0, 0.975], [15.0, 0.97], [16.0, 0.97], [17.0, 0.97], [18.0, 0.71], [19.0, 0.71], [20.0, 0.366], [21.0, 0.366], [22.0, 0.366], [23.0, 0.366], [24.0, 0.366], [25.0, 0.11], [26.0, 0.11], [27.0, 0.11], [28.0, 0.11], [29.0, 0.11], [30.0, 0.11], [31.0, 0.11], [32.0, 0.11], [33.0, 0.11], [34.0, 0.11], [35.0, 0.024], [36.0, 0.024], [37.0, 0.024], [38.0, 0.024], [39.0, 0.024], [40.0, 0.024], [41.0, 0.024], [42.0, 0.024], [43.0, 0.024], [44.0, 0.024], [45.0, 0.024], [46.0, 0.024], [47.0, 0.024], [48.0, 0.024], [49.0, 0.024], [50.0, 0.024], [51.0, 0.0], [52.0, 0.0], [53.0, 0.0], [54.0, 0.0], [55.0, 0.0], [56.0, 0.0], [57.0, 0.0], [58.0, 0.0], [59.0, 0.0], [60.0, 0.0], [61.0, 0.0], [62.0, 0.0], [63.0, 0.0], [64.0, 0.0], [65.0, 0.0], [66.0, 0.0], [67.0, 0.0], [68.0, 0.0], [69.0, 0.0], [70.0, 0.0], [71.0, 0.0], [72.0, 0.0], [73.0, 0.0], [74.0, 0.0], [75.0, 0.0], [76.0, 0.0], [77.0, 0.0], [78.0, 0.0], [79.0, 0.0], [80.0, 0.0], [81.0, 0.0], [82.0, 0.0], [83.0, 0.0], [84.0, 0.0], [85.0, 0.0], [86.0, 0.0], [87.0, 0.0], [88.0, 0.0], [89.0, 0.0], [90.0, 0.0], [91.0, 0.0], [92.0, 0.0], [93.0, 0.0], [94.0, 0.0], [95.0, 0.0], [96.0, 0.0], [97.0, 0.0], [98.0, 0.0], [99.0, 0.0], [100.0, 0.0]], 1, False, 1, 100)
    ch_enrollment_rates_by_age_dist_vars(enrollment_rates_by_age_dist_vars)
    
    school_size_distribution_by_pk_dist_vars = get_size_distribution_vars([0.17567567567567569, 0.06756756756756757, 0.0945945945945946, 0.5135135135135135, 0.14864864864864866, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, True, 0, 5)
    school_size_distribution_by_es_dist_vars = get_size_distribution_vars([0.038461538461538464, 0.038461538461538464, 0.11538461538461539, 0.038461538461538464, 0.038461538461538464, 0.0, 0.038461538461538464, 0.11538461538461539, 0.23076923076923078, 0.23076923076923078, 0.07692307692307693, 0.038461538461538464, 0.0, 0.0], 0, True, 0, 14)
    school_size_distribution_by_uv_dist_vars = get_size_distribution_vars([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.03, 0.05, 0.10, 0.20, 0.60], 0, True, 7, 15)
    #print(school_size_distribution_by_pk_dist_vars)

    possible_keys_enr = np.array(list(enrollment_rates_by_age_dist_vars.keys()))
    possible_keys_pk = np.array(list(school_size_distribution_by_pk_dist_vars.keys()))
    possible_keys_es = np.array(list(school_size_distribution_by_es_dist_vars.keys()))
    possible_keys_uv = np.array(list(school_size_distribution_by_uv_dist_vars.keys()))
    
    input_school_parameters = dict(
        average_student_teacher_ratio=sp_average_student_teacher_ratio, # [10-30]
        average_class_size=sp_average_class_size, # [5-35]
        average_teacher_teacher_degree=param_values[:, param2ind["average_teacher_teacher_degree"]], # [0-10]
        average_student_all_staff_ratio=sp_average_student_all_staff_ratio, # [5-35]
        average_additional_staff_degree=param_values[:, param2ind["average_additional_staff_degree"]], # [5-35]
        indices_enrollment_rates_by_age=param_values[:, param2ind['enrollment_rates_by_age']],
        indices_school_size_distribution_by_pk=param_values[:, param2ind['school_size_distribution_by_pk']],
        indices_school_size_distribution_by_es=param_values[:, param2ind['school_size_distribution_by_es']],
        indices_school_size_distribution_by_uv=param_values[:, param2ind['school_size_distribution_by_uv']],
    )

    # household parameters
    household_size_distribution_vars = get_size_distribution_vars([[1.0, 0.4427288451567194], [2.0, 0.2619681851105684], [3.0, 0.16098223226689992], [4.0, 0.09663208641237447], [5.0, 0.027742045288127005], [6.0, 0.006962228229678328], [7.0, 0.002984377535632439]], 1, True, 1, 8)
    possible_keys_hssd = np.array(list(household_size_distribution_vars.keys()))

    household_matrix = lp.HouseholdParameters.get_default_parameters().contact_matrix
    diff_household_matrices = get_diff_kinds_contacts(household_matrix, 'h')
    possible_keys_hms = np.array(list(diff_household_matrices.keys()))

    input_household_parameters = dict(
        indices_household_size_distribution=param_values[:, param2ind["household_size_distribution"]],
        indices_household_contact_matrices=param_values[:, param2ind["h_contact_matrix"]],
    )

    # work parameters
    work_size_distribution_vars = get_size_distribution_vars([[1, 4, 738.0], [5, 9, 402.0], [10, 19, 417.0], [20, 49, 716.0], [50, 99, 630.0], [100, 249, 413.0], [250, 499, 157.0], [500, 999, 74.0], [1000, 1999, 32.0]], 2, False, 1, 10)
    possible_keys_wd = np.array(list(work_size_distribution_vars.keys()))

    employment_rates_by_age_dist_vars = get_size_distribution_vars([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0], [7.0, 0.0], [8.0, 0.0], [9.0, 0.0], [10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0], [14.0, 0.0], [15.0, 0.0], [16.0, 0.332], [17.0, 0.332], [18.0, 0.332], [19.0, 0.332], [20.0, 0.699], [21.0, 0.699], [22.0, 0.699], [23.0, 0.699], [24.0, 0.699], [25.0, 0.784], [26.0, 0.784], [27.0, 0.784], [28.0, 0.784], [29.0, 0.784], [30.0, 0.782], [31.0, 0.782], [32.0, 0.782], [33.0, 0.782], [34.0, 0.782], [35.0, 0.792], [36.0, 0.792], [37.0, 0.792], [38.0, 0.792], [39.0, 0.792], [40.0, 0.792], [41.0, 0.792], [42.0, 0.792], [43.0, 0.792], [44.0, 0.792], [45.0, 0.801], [46.0, 0.801], [47.0, 0.801], [48.0, 0.801], [49.0, 0.801], [50.0, 0.801], [51.0, 0.801], [52.0, 0.801], [53.0, 0.801], [54.0, 0.801], [55.0, 0.721], [56.0, 0.721], [57.0, 0.721], [58.0, 0.721], [59.0, 0.721], [60.0, 0.567], [61.0, 0.567], [62.0, 0.567], [63.0, 0.567], [64.0, 0.567], [65.0, 0.242], [66.0, 0.242], [67.0, 0.242], [68.0, 0.242], [69.0, 0.242], [70.0, 0.242], [71.0, 0.242], [72.0, 0.242], [73.0, 0.242], [74.0, 0.242], [75.0, 0.064], [76.0, 0.064], [77.0, 0.064], [78.0, 0.064], [79.0, 0.064], [80.0, 0.064], [81.0, 0.064], [82.0, 0.064], [83.0, 0.064], [84.0, 0.064], [85.0, 0.064], [86.0, 0.064], [87.0, 0.064], [88.0, 0.064], [89.0, 0.064], [90.0, 0.064], [91.0, 0.064], [92.0, 0.064], [93.0, 0.064], [94.0, 0.064], [95.0, 0.064], [96.0, 0.064], [97.0, 0.064], [98.0, 0.064], [99.0, 0.064], [100.0, 0.064]], 1, False, 0, 100)
    possible_keys_emr = np.array(list(employment_rates_by_age_dist_vars.keys()))

    work_matrix = lp.WorkParameters.get_default_parameters().contact_matrix
    diff_work_matrices = get_diff_kinds_contacts(work_matrix, 'w')
    possible_keys_ws = np.array(list(diff_work_matrices.keys()))

    input_work_parameters = dict(
        indices_work_size_distribution=param_values[:, param2ind["workplace_size_counts_by_num_personnel"]],
        indices_work_contact_matrices=param_values[:, param2ind["w_contact_matrix"]],
        indices_employment_rates_by_age=param_values[:, param2ind['employment_rates_by_age']]
    )

 
    all_city_pars = []
    for input_id in range(param_values.shape[0]):
        for rand_seed in range(rand_seed_n):
            h_matrix = diff_household_matrices[possible_keys_hms[int(input_household_parameters['indices_household_contact_matrices'][input_id])]]
            w_matrix = diff_work_matrices[possible_keys_ws[int(input_work_parameters['indices_work_contact_matrices'][input_id])]]
            city_pars = dict(
                common_pars=lp.CommonParameters(
                    rand_seed=rand_seed,
                    location=f"Nsk_{input_id}_{rand_seed}",
                    n=pop_size
                ),
                household_pars=lp.HouseholdParameters(
                    household_size_distribution=household_size_distribution_vars[possible_keys_hssd[int(input_household_parameters['indices_household_size_distribution'][input_id])]],
                    contact_matrix=h_matrix
                ),
                school_pars=lp.SchoolParameters(
                    average_student_teacher_ratio=int(input_school_parameters['average_student_teacher_ratio'][input_id]), # [10-30]
                    average_class_size=int(input_school_parameters['average_class_size'][input_id]), # [5-35]
                    average_teacher_teacher_degree=int(input_school_parameters['average_teacher_teacher_degree'][input_id]), # [0-10]
                    average_student_all_staff_ratio=max(int(input_school_parameters['average_student_all_staff_ratio'][input_id]), 1), # [5-35]
                    average_additional_staff_degree=int(input_school_parameters['average_additional_staff_degree'][input_id]), 
                    employment_rates_by_age=employment_rates_by_age_dist_vars[possible_keys_emr[int(input_work_parameters['indices_employment_rates_by_age'][input_id])]],
                    enrollment_rates_by_age=enrollment_rates_by_age_dist_vars[possible_keys_enr[int(input_school_parameters['indices_enrollment_rates_by_age'][input_id])]],
                    school_size_distribution_by_type=[
                        {
                            'school_type': 'pk', 
                            'size_distribution': school_size_distribution_by_pk_dist_vars[possible_keys_pk[int(input_school_parameters['indices_school_size_distribution_by_pk'][input_id])]]
                        }, 
                        {
                            'school_type': 'es', 
                            'size_distribution': school_size_distribution_by_es_dist_vars[possible_keys_es[int(input_school_parameters['indices_school_size_distribution_by_es'][input_id])]]
                        }, 
                        {
                            'school_type': 'uv', 
                            'size_distribution': school_size_distribution_by_uv_dist_vars[possible_keys_uv[int(input_school_parameters['indices_school_size_distribution_by_uv'][input_id])]]
                        }
                    ],
                ),
                work_pars=lp.WorkParameters(
                    workplace_size_counts_by_num_personnel=work_size_distribution_vars[possible_keys_wd[int(input_work_parameters['indices_work_size_distribution'][input_id])]],
                    contact_matrix=w_matrix
                ),
                filename=None
            )
            all_city_pars.append(city_pars)

    step = 100
    all_count = len(all_city_pars)

    for i in tqdm(range(0, all_count, step)):
        with multiprocessing.Pool(processes=step) as pool:
            ress = pool.map(single_build_and_run, all_city_pars[i:i+step])
        msim = cv.MultiSim(list(ress))
        do_results(msim, rand_seed_n, f"peak_morris_2/synth_{i}")


def multi_run_wrapper(args):
   return single_build_and_run_all(*args)

def generate_multi_params():
    params_inv = [
        'rel_beta',
        'rel_symp_prob',
        'dur_asym2rec',
        'dur_mild2rec',
        'enrollment_rates_by_age',
        'employment_rates_by_age', 
        'household_size_distribution',
        'average_student_teacher_ratio',
        'average_class_size',
        #'average_student_all_staff_ratio'
        ]
    param2ind = dict(zip(params_inv, range(len(params_inv))))
    problem = {
        'num_vars': len(params_inv),
        'names': params_inv,
        'bounds': [
            [0.1, 2.0],
            [0.1, 2.0],
            [4.000210396272597, 11.999356371721072],
            [4.0002485424543, 11.998532442496924],
            [0, 6],
            [0, 6],
            [0, 6],
            [10, 29],
            [0, 15],
        ]
    }

    param2ind = dict(zip(params_inv, range(len(params_inv))))
    param_values = morris.sample(problem, 2000, num_levels=4)
    #param_values = saltelli.sample(problem, 2048)
    return param_values, param2ind


def make_multi_data(param_values, param2ind, rand_seed_n):
    pop_size = 100000
    
    # school parameters
    sp_average_student_teacher_ratio = param_values[:, param2ind['average_student_teacher_ratio']]
    sp_average_class_size = sp_average_student_teacher_ratio + param_values[:, param2ind['average_class_size']]
    sp_average_student_all_staff_ratio = sp_average_student_teacher_ratio - 5 # [5-35]

    enrollment_rates_by_age_dist_vars = get_size_distribution_vars([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.439], [4.0, 0.439], [5.0, 0.941], [6.0, 0.941], [7.0, 0.941], [8.0, 0.941], [9.0, 0.941], [10.0, 0.975], [11.0, 0.975], [12.0, 0.975], [13.0, 0.975], [14.0, 0.975], [15.0, 0.97], [16.0, 0.97], [17.0, 0.97], [18.0, 0.71], [19.0, 0.71], [20.0, 0.366], [21.0, 0.366], [22.0, 0.366], [23.0, 0.366], [24.0, 0.366], [25.0, 0.11], [26.0, 0.11], [27.0, 0.11], [28.0, 0.11], [29.0, 0.11], [30.0, 0.11], [31.0, 0.11], [32.0, 0.11], [33.0, 0.11], [34.0, 0.11], [35.0, 0.024], [36.0, 0.024], [37.0, 0.024], [38.0, 0.024], [39.0, 0.024], [40.0, 0.024], [41.0, 0.024], [42.0, 0.024], [43.0, 0.024], [44.0, 0.024], [45.0, 0.024], [46.0, 0.024], [47.0, 0.024], [48.0, 0.024], [49.0, 0.024], [50.0, 0.024], [51.0, 0.0], [52.0, 0.0], [53.0, 0.0], [54.0, 0.0], [55.0, 0.0], [56.0, 0.0], [57.0, 0.0], [58.0, 0.0], [59.0, 0.0], [60.0, 0.0], [61.0, 0.0], [62.0, 0.0], [63.0, 0.0], [64.0, 0.0], [65.0, 0.0], [66.0, 0.0], [67.0, 0.0], [68.0, 0.0], [69.0, 0.0], [70.0, 0.0], [71.0, 0.0], [72.0, 0.0], [73.0, 0.0], [74.0, 0.0], [75.0, 0.0], [76.0, 0.0], [77.0, 0.0], [78.0, 0.0], [79.0, 0.0], [80.0, 0.0], [81.0, 0.0], [82.0, 0.0], [83.0, 0.0], [84.0, 0.0], [85.0, 0.0], [86.0, 0.0], [87.0, 0.0], [88.0, 0.0], [89.0, 0.0], [90.0, 0.0], [91.0, 0.0], [92.0, 0.0], [93.0, 0.0], [94.0, 0.0], [95.0, 0.0], [96.0, 0.0], [97.0, 0.0], [98.0, 0.0], [99.0, 0.0], [100.0, 0.0]], 1, False, 1, 100)
    ch_enrollment_rates_by_age_dist_vars(enrollment_rates_by_age_dist_vars)
    possible_keys_enr = np.array(list(enrollment_rates_by_age_dist_vars.keys()))

    input_school_parameters = dict(
        average_student_teacher_ratio=sp_average_student_teacher_ratio, # [10-30]
        average_class_size=sp_average_class_size, # [5-35]
        average_student_all_staff_ratio=sp_average_student_all_staff_ratio, # [5-35]
        indices_enrollment_rates_by_age=param_values[:, param2ind['enrollment_rates_by_age']],
    )

    # household parameters
    household_size_distribution_vars = get_size_distribution_vars([[1.0, 0.4427288451567194], [2.0, 0.2619681851105684], [3.0, 0.16098223226689992], [4.0, 0.09663208641237447], [5.0, 0.027742045288127005], [6.0, 0.006962228229678328], [7.0, 0.002984377535632439]], 1, True, 1, 8)
    possible_keys_hssd = np.array(list(household_size_distribution_vars.keys()))

    input_household_parameters = dict(
        indices_household_size_distribution=param_values[:, param2ind["household_size_distribution"]],
    )

    # work parameters
    employment_rates_by_age_dist_vars = get_size_distribution_vars([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0], [7.0, 0.0], [8.0, 0.0], [9.0, 0.0], [10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0], [14.0, 0.0], [15.0, 0.0], [16.0, 0.332], [17.0, 0.332], [18.0, 0.332], [19.0, 0.332], [20.0, 0.699], [21.0, 0.699], [22.0, 0.699], [23.0, 0.699], [24.0, 0.699], [25.0, 0.784], [26.0, 0.784], [27.0, 0.784], [28.0, 0.784], [29.0, 0.784], [30.0, 0.782], [31.0, 0.782], [32.0, 0.782], [33.0, 0.782], [34.0, 0.782], [35.0, 0.792], [36.0, 0.792], [37.0, 0.792], [38.0, 0.792], [39.0, 0.792], [40.0, 0.792], [41.0, 0.792], [42.0, 0.792], [43.0, 0.792], [44.0, 0.792], [45.0, 0.801], [46.0, 0.801], [47.0, 0.801], [48.0, 0.801], [49.0, 0.801], [50.0, 0.801], [51.0, 0.801], [52.0, 0.801], [53.0, 0.801], [54.0, 0.801], [55.0, 0.721], [56.0, 0.721], [57.0, 0.721], [58.0, 0.721], [59.0, 0.721], [60.0, 0.567], [61.0, 0.567], [62.0, 0.567], [63.0, 0.567], [64.0, 0.567], [65.0, 0.242], [66.0, 0.242], [67.0, 0.242], [68.0, 0.242], [69.0, 0.242], [70.0, 0.242], [71.0, 0.242], [72.0, 0.242], [73.0, 0.242], [74.0, 0.242], [75.0, 0.064], [76.0, 0.064], [77.0, 0.064], [78.0, 0.064], [79.0, 0.064], [80.0, 0.064], [81.0, 0.064], [82.0, 0.064], [83.0, 0.064], [84.0, 0.064], [85.0, 0.064], [86.0, 0.064], [87.0, 0.064], [88.0, 0.064], [89.0, 0.064], [90.0, 0.064], [91.0, 0.064], [92.0, 0.064], [93.0, 0.064], [94.0, 0.064], [95.0, 0.064], [96.0, 0.064], [97.0, 0.064], [98.0, 0.064], [99.0, 0.064], [100.0, 0.064]], 1, False, 0, 100)
    possible_keys_emr = np.array(list(employment_rates_by_age_dist_vars.keys()))

    input_work_parameters = dict(
        indices_employment_rates_by_age=param_values[:, param2ind['employment_rates_by_age']]
    )


    # virus parameters
    all_city_pars = []
    all_epid_pars = []
    for input_id in tqdm(range(param_values.shape[0])):
        for rand_seed in range(rand_seed_n):
            city_pars = dict(
                common_pars=lp.CommonParameters(
                    rand_seed=rand_seed,
                    location=f"Nsk_{input_id}_{rand_seed}",
                    n=pop_size
                ),
                household_pars=lp.HouseholdParameters(
                    household_size_distribution=household_size_distribution_vars[possible_keys_hssd[int(input_household_parameters['indices_household_size_distribution'][input_id])]],
                ),
                school_pars=lp.SchoolParameters(
                    average_student_teacher_ratio=int(input_school_parameters['average_student_teacher_ratio'][input_id]), # [10-30]
                    average_class_size=int(input_school_parameters['average_class_size'][input_id]), # [5-35]
                    average_student_all_staff_ratio=max(int(input_school_parameters['average_student_all_staff_ratio'][input_id]), 1), # [5-35]
                    employment_rates_by_age=employment_rates_by_age_dist_vars[possible_keys_emr[int(input_work_parameters['indices_employment_rates_by_age'][input_id])]],
                    enrollment_rates_by_age=enrollment_rates_by_age_dist_vars[possible_keys_enr[int(input_school_parameters['indices_enrollment_rates_by_age'][input_id])]],
                ),
                filename=None
            )
            all_epid_pars.append(dict(
                rel_beta=param_values[input_id, param2ind["rel_beta"]],
                rel_symp_prob=param_values[input_id, param2ind["rel_symp_prob"]],
                dur_asym2rec=param_values[input_id, param2ind["dur_asym2rec"]],
                dur_mild2rec=param_values[input_id, param2ind["dur_mild2rec"]],
            ))
            all_city_pars.append(city_pars)

    #with open('last_3_sobol_epid_pars.pickle', 'wb') as handle:
    #    pickle.dump(all_epid_pars, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('last_3_sobol_city_pars.pickle', 'wb') as handle:
    #    pickle.dump(all_city_pars, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return all_epid_pars, all_city_pars

def generate_multi_data():
    param_values, param2ind = generate_multi_params()
    #np.save("param_values_last_sobol_3", param_values)
    #param_values = np.load("new_param_values_multi_10k.npy")
    rand_seed_n = 2

    all_epid_pars, all_city_pars = make_multi_data(param_values, param2ind, rand_seed_n)
    #with open('new_morris_epid_pars.pickle', 'rb') as handle:
    #    all_epid_pars = pickle.load(handle)
    #with open('new_morris_city_pars.pickle', 'rb') as handle:
    #    all_city_pars = pickle.load(handle)
    
    step = 100
    all_count = len(all_city_pars)
    for i in tqdm(range(0, all_count, step)):
        with multiprocessing.Pool(processes=step) as pool:
            ress = pool.map(multi_run_wrapper, list(zip(all_epid_pars[i:i+step], all_city_pars[i:i+step])))
        msim = cv.MultiSim(list(ress))
        do_results(msim, rand_seed_n, f"peak_morris_3/synth_{i}")


def generate_params():
    params_inv = ['rel_beta',
                'rel_symp_prob',
                'rel_severe_prob',
                'rel_crit_prob',
                'rel_death_prob',
                'dur_exp2inf',
                'dur_inf2sym',
                'dur_sym2sev',
                'dur_sev2crit',
                'dur_asym2rec',
                'dur_mild2rec',
                'dur_sev2rec',
                'dur_crit2rec',
                'dur_crit2die',
                'oral_microbiota_percent',
                'oral_microbiota_factor',
                'n_imports',
                'indices_starting_months',
                'indices_weather',
                'indices_rel_sus_type']
    param2ind = dict(zip(params_inv, range(len(params_inv))))
    problem = {
        'num_vars': len(params_inv),
        'names': params_inv,
        'bounds': [[1.3194024973905982e-05, 1.9999425593941065],
                [0.0001059037659971, 1.99987452255794],
                [0.0002294889944518, 1.9999690385440527],
                [1.1662541377299718e-05, 1.999988850911703],
                [0.0003941551701103, 1.999698598745013],
                [2.000602033966775, 5.999937611927544],
                [9.033236261313248e-05, 1.99996917989662],
                [2.000555057201062, 9.999799633851948],
                [0.0001424376024165, 2.9993645217130664],
                [4.000210396272597, 11.999356371721072],
                [4.0002485424543, 11.998532442496924],
                [10.001949044431957, 29.999978630136265],
                [10.000866868808028, 29.99786878107664],
                [6.000592711534851, 14.99916146707454],
                [5.28217691269095e-05, 0.9999790775007206],
                [0.0001383684747588, 0.9999580139962424],
                [1, 999],
                [0, 3],
                [0, 2],
                [0, 8]]}

    moscow_weather = {
        'Outside Temperature (°C)': [-6.9, -6.1, -0.9, 6.9, 13.2, 17.2, 20.0, 17.9, 11.8, 5.8, -0.1, -4.3], 
        'Outside Humidity (%)': [0.86, 0.82, 0.74, 0.66, 0.65, 0.68, 0.72, 0.74, 0.8, 0.82, 0.85, 0.87], 
        'Inside Temperature (°C)': [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22]
    }
 
    #______________________________
    norislsk_weather = {
        'Outside Temperature (°C)': [-26.6, -22.2, -17.8, -9.2, -2.2, 9.7, 14.1, 11.3, 4.5, -4.7, -18.9, -21.1], 
        'Outside Humidity (%)': [0.76, 0.77, 0.78, 0.76, 0.71, 0.66, 0.67, 0.76, 0.82, 0.85, 0.84, 0.80], 
        'Inside Temperature (°C)': [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]
    }
    #______________________________
    sochi_weather = {
        'Outside Temperature (°C)': [6.5, 7.6, 9.1, 12.5, 16.9, 22.2, 24.5, 25.5, 21.6, 16.4, 11.9, 8.7], 
        'Outside Humidity (%)': [0.73, 0.73, 0.73, 0.73, 0.76, 0.76, 0.75, 0.73, 0.73, 0.73, 0.71, 0.73], 
        'Inside Temperature (°C)': [23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23]
    }

    choice_starting_months = np.array([
                'December',
                'March',
                'June',
                'September',
            ])
    choice_rel_sus_type = np.array([
        'constants',
        "normal_pos",
        "normal_pos_all",
        "lognormal",
        "lognormal_lite_all",
        "lognormal_hot_all",
        "beta_1_all",
        "beta_3_all",
        "uniform",
    ])
    label_weathers = np.array(["Moscow", "Norilsk", 'Sochi'])
    value_weathers = np.array([moscow_weather, norislsk_weather, sochi_weather])

    param_values = morris.sample(problem, 2000, num_levels=4)
    return param_values, value_weathers, choice_starting_months, choice_rel_sus_type, param2ind


def do_results(msim, rand_seed_n, csv_name):
    out_res = []
    sims_count = len(msim.sims)
    for (input_id, i) in enumerate(range(0, sims_count, rand_seed_n)):
        tmp_msim = cv.MultiSim(msim.sims[i: i + rand_seed_n])
        tmp_msim.mean()
        peak_day = np.argmax(np.array(tmp_msim.results['new_infections']))
        print(f"Peak day: {peak_day}; {tmp_msim.results['cum_infections'][-1]}")
        out_res.append(peak_day)
        #out_res.append(tmp_msim.results['cum_infections'][-1])

    num_ar = np.array(out_res)
    np.save(csv_name, num_ar)


def generate_data(param_values, start_ind, end_ind, 
                       value_weathers, choice_starting_months, 
                       choice_rel_sus_type, param2ind, filename):
    rand_seed_n = 2
    pop_size = 100000
    sims = []
    for input_id in range(start_ind, end_ind):
        for rand_seed in range(rand_seed_n):
            sims.append(
                cv.Sim(
                    pars={
                        "n_days": 300,
                        "pop_size": pop_size,
                        "monthly_weather": value_weathers[int(floor(param_values[input_id, param2ind["indices_weather"]]))],
                        "starting_month": choice_starting_months[int(floor(param_values[input_id, param2ind["indices_starting_months"]]))],
                        "rel_sus_type": choice_rel_sus_type[int(floor(param_values[input_id, param2ind["indices_rel_sus_type"]]))]
                    },
                    pop_type="synthpops", variants=cv.variant(
                    variant={
                        "rel_beta": param_values[input_id, param2ind["rel_beta"]], 
                        "rel_symp_prob": param_values[input_id, param2ind["rel_symp_prob"]], 
                        "rel_severe_prob": param_values[input_id, param2ind["rel_severe_prob"]], 
                        "rel_crit_prob": param_values[input_id, param2ind["rel_crit_prob"]], 
                        "rel_death_prob": param_values[input_id, param2ind["rel_death_prob"]],
                        "oral_microbiota_percent": param_values[input_id, param2ind["oral_microbiota_percent"]],
                        "oral_microbiota_factor": param_values[input_id, param2ind["oral_microbiota_factor"]],
                        "dur_exp2inf": dict(dist="lognormal_int", par1=param_values[input_id, param2ind['dur_exp2inf']], par2=1.5),
                        "dur_inf2sym": dict(dist="lognormal_int", par1=param_values[input_id, param2ind['dur_inf2sym']], par2=0.9),
                        "dur_sym2sev": dict(dist="lognormal_int", par1=param_values[input_id, param2ind['dur_sym2sev']], par2=4.9),
                        "dur_sev2crit": dict(dist="lognormal_int", par1=param_values[input_id, param2ind['dur_sev2crit']], par2=2.0),
                        "dur_asym2rec": dict(dist="lognormal_int", par1=param_values[input_id, param2ind['dur_asym2rec']], par2=2.0),
                        "dur_mild2rec": dict(dist="lognormal_int", par1=param_values[input_id, param2ind['dur_mild2rec']], par2=2.0),
                        "dur_sev2rec": dict(dist="lognormal_int", par1=param_values[input_id, param2ind['dur_sev2rec']], par2=6.3),
                        "dur_crit2rec": dict(dist="lognormal_int", par1=param_values[input_id, param2ind['dur_crit2rec']], par2=6.3),
                        "dur_crit2die": dict(dist="lognormal_int", par1=param_values[input_id, param2ind['dur_crit2die']], par2=4.8),
                    }, days=0), popfile=f"synthpops_files/synth_pop_{pop_size//1000}K.ppl", rand_seed=rand_seed, verbose=False))
    msim = cv.MultiSim(sims)
    msim.run()
    #arr = param_values[start_ind:start_ind + 20, param2ind["rel_beta"]]
    #arr1 = param_values[start_ind:start_ind + 20][param2ind["rel_beta"]]
    #print(np.min(arr), np.max(arr))
    #print(arr[:20])
    #print('******')
    #print(np.min(arr1), np.max(arr1))
    #print(arr1[:20])
    #


    do_results(msim, rand_seed_n, filename)

def main_generate_data():
    param_values, value_weathers, choice_starting_months, choice_rel_sus_type, param2ind = generate_params()
    #np.save("param_values_epid_10k", param_values)
    n_data = param_values.shape[0]
    step = 100
    for i in tqdm(range(0, n_data, step)):
        generate_data(
            param_values=param_values,
            value_weathers=value_weathers,
            choice_starting_months=choice_starting_months,
            choice_rel_sus_type=choice_rel_sus_type,
            param2ind=param2ind,
            start_ind=i, 
            end_ind=min(i+step, n_data),
            filename=f"peak_morris_1/synth_{i}"
        )

def merge_y_numpy(directory_path):
    import os

    np_arrs = []
    # Loop through all the files in the specified directory
    file_paths = []
    nums = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        file_paths.append(file_path)
        nums.append(int(file_path[len(f"{directory_path}/synth_"):-4]))
    file_paths = np.array(file_paths)[np.argsort(nums)]

    i=0
    for filename in file_paths:
        np_arr = np.load(filename)
        np_arrs.append(np_arr)
        i+=1
    return np.concatenate(np_arrs, axis=0)


def dddd():
    #main_generate_data()
    param_values = np.load("sobol_param_values_256.bin.npy")
    qq = merge_y_numpy("sobol_0")
    print(qq.shape)
    rel_beta = param_values[:, 0]
    inds = np.argsort(rel_beta)
    zz = list(zip(rel_beta[inds], qq[inds]))[:10]
    for z in zz:
        print(z)
#
    ##
    #tt = qq[:10752]
    np.save("analysis_model/sobol_y_virus_10k", qq)

if __name__ == '__main__':
    #main_generate_data()
    #generate_synth_data()
    generate_multi_data()
    
    #params, _ = generate_multi_params()
    #print(params.shape)
    #np.save("sobol_param_multi_256.bin", params)
    qq = merge_y_numpy("peak_morris_3")
    np.save("analysis_model/peak_morris_3_y", qq)
