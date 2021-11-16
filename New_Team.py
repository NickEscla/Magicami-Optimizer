# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
import csv
import pygad
import itertools
from tqdm import tqdm
global Dress_Database
global Orb_Database
global Wanted_Squad
NAME_COL = 0
GIRL_COL = 2
ELEMENT_COL = 3
HP_COL = 5
ATK_COL = 6
DEF_COL = 7
FCS_COL = 8
RES_COL = 9
AGI_COL = 10
JOKER_COL = 11
ORB_TYPE_COL = 1
ORB_MAIN_COL = 2
ORB_HP_COL = 3
ORB_ATK_COL = 4
ORB_DEF_COL = 5
ORB_AGI_COL = 6
ORB_FCS_COL = 7
ORB_RES_COL = 8
ORB_HP_PER_COL = 9
ORB_ATK_PER_COL = 10
ORB_DEF_PER_COL = 11
ORB_ALL_PER_COL = 12
ORB_ELEMENT_COL = 15
ORB_CC_COL = 13
ORB_CD_COL = 14


def fitness_func(solution, solution_idx):
    # solution is the array of genes
    # here we need to generate the full squad based on the genes
    global Wanted_Squad
    # four combined dresses get added

    solution_iter = 0
    full_squad = FullSquad()
    for i in range(0, 4):
        # make the dress based on wanted squad
        temp_dress = DressClass(int(Wanted_Squad[0][i]))
        # print(solution[solution_iter])
        temp_dress.add_orb(int(solution[solution_iter]))
        solution_iter += 1
        temp_combined_dress = CombinedDress(temp_dress)
        for a in range(0, 4):
            # create the four subdresses
            temp_dress = DressClass(int(solution[solution_iter]))
            # temp_dress_row = temp_dress.row
            solution_iter += 1
            temp_dress.add_orb(int(solution[solution_iter]))
            # temp_dress_orb = temp_dress.orb_row
            solution_iter += 1
            # check validity. dress is 3, orb is 2, full is 1. if dress is not valid go to next dress etc
            temp_combined_dress.add_dress(temp_dress)
            # to increase likelihood of valid teams, increase gene by 1 if it does not add a valid dress
            # while not code == 0:
            #     if code == 1:
            #         break
            #     if code == 2:
            #         temp_dress_orb += 1
            #     if code == 3:
            #         temp_dress_row += 1
            #     if temp_dress_orb >= len(Dress_Database) or temp_dress_orb >= len(Orb_Database):
            #         return 0
            #     temp_dress = DressClass(temp_dress_row)
            #     temp_dress.add_orb(temp_dress_orb)
            #     code = temp_combined_dress.add_dress(temp_dress)
        full_squad.add_dress(temp_combined_dress.get_as_array())

    fitness = 0
    # if the squad is not full, fitness is 0
    # if solution.get_full() < 4:
    #     return fitness
    # fitness will be each desired stat added up
    # crit_c and crit_d already accounted for in atk stat
    for i in range(0, len(full_squad.dress_arrays)):
        fitness += full_squad.dress_arrays[i][2][int(Wanted_Squad[1][i])]
    return fitness


def get_all_dress_orb():
    global Dress_Database
    global Orb_Database
    global Wanted_Squad
    Wanted_Squad = numpy.empty((0, 4), int)
    Dress_Database = numpy.genfromtxt("dresses.csv", delimiter=',', skip_header=1, dtype=str)
    Orb_Database = numpy.genfromtxt("orbs0.csv", delimiter=',', dtype=str)
    temp_wanted_squad = numpy.genfromtxt("wanted.csv", delimiter=',', dtype=str)
    stat_reference = ['HP', 'ATK', 'DEF', 'FCS', 'RES', 'AGI']
    new_row = numpy.array([])
    for col_iter in range(0, 4):
        new_row = numpy.append(new_row,
                               numpy.where(numpy.isin(Dress_Database[:, NAME_COL], temp_wanted_squad[0][col_iter])))
    Wanted_Squad = numpy.append(Wanted_Squad, [new_row], axis=0)
    new_row = numpy.array([])
    for col_iter in range(0, 4):
        new_row = numpy.append(new_row, stat_reference.index(temp_wanted_squad[1][col_iter]))
    Wanted_Squad = numpy.append(Wanted_Squad, [new_row], axis=0)
    # print(Wanted_Squad)


def clean_input_csv():
    # add 0 instead of blanks in the orb CSV.
    row_read = csv.reader(open("orbs.csv", "r", newline=''))
    row_write = csv.writer(open("orbs0.csv", "w", newline=''))
    for row in row_read:
        new_row = [val if val else "0" for val in row]
        row_write.writerow(new_row)


class DressClass:
    def __init__(self, row):
        self.name = Dress_Database[row][NAME_COL]
        self.row = row
        self.girl = Dress_Database[self.row][GIRL_COL]
        self.element = Dress_Database[self.row][ELEMENT_COL]
        self.hp = float(Dress_Database[self.row][HP_COL])
        self.atk = float(Dress_Database[self.row][ATK_COL])
        self.Def = float(Dress_Database[self.row][DEF_COL])
        self.fcs = float(Dress_Database[self.row][FCS_COL])
        self.res = float(Dress_Database[self.row][RES_COL])
        self.agi = float(Dress_Database[self.row][AGI_COL])
        self.multiplier = 0
        self.joker = Dress_Database[self.row][JOKER_COL]
        self.crit_c = 0
        self.crit_d = 0
        self.orb_row = -1
        self.orb_type = None

    def add_orb(self, orb_row):
        self.orb_row = orb_row
        self.orb_type = Orb_Database[self.orb_row][ORB_TYPE_COL]
        orb_main = int(Orb_Database[self.orb_row][ORB_MAIN_COL])
        hp_multi = 0
        atk_multi = 0
        def_multi = 0
        hp_add = 0
        atk_add = 0
        def_add = 0
        fcs_add = 0
        res_add = 0
        agi_add = 0
        if self.orb_type == "HP%":
            hp_multi = orb_main
        if self.orb_type == "ATK%":
            atk_multi = orb_main
        if self.orb_type == "DEF%":
            def_multi = orb_main
        if self.orb_type == "HP":
            hp_add = orb_main
        if self.orb_type == "ATK":
            atk_add = orb_main
        if self.orb_type == "DEF":
            def_add = orb_main
        if self.orb_type == "FCS":
            fcs_add = orb_main
        if self.orb_type == "RES":
            res_add = orb_main
        if self.orb_type == "AGI":
            agi_add = orb_main
        self.hp *= (1 + int(Orb_Database[self.orb_row][ORB_HP_PER_COL]) / 100 + int(
                    Orb_Database[self.orb_row][ORB_ALL_PER_COL]) / 100 + (hp_multi / 100))
        self.atk *= (1 + int(Orb_Database[self.orb_row][ORB_ATK_PER_COL]) / 100 + int(
                    Orb_Database[self.orb_row][ORB_ALL_PER_COL]) / 100 + (atk_multi / 100))
        self.Def *= (1 + int(Orb_Database[self.orb_row][ORB_DEF_PER_COL]) / 100 + int(
                    Orb_Database[self.orb_row][ORB_ALL_PER_COL]) / 100 + (def_multi / 100))
        self.fcs *= (1 + int(Orb_Database[self.orb_row][ORB_ALL_PER_COL]) / 100)
        self.res *= (1 + int(Orb_Database[self.orb_row][ORB_ALL_PER_COL]) / 100)
        self.agi *= (1 + int(Orb_Database[self.orb_row][ORB_ALL_PER_COL]) / 100)
        self.hp += int(Orb_Database[self.orb_row][ORB_HP_COL]) + hp_add
        self.atk += int(Orb_Database[self.orb_row][ORB_ATK_COL]) + atk_add
        self.Def += int(Orb_Database[self.orb_row][ORB_DEF_COL]) + def_add
        self.fcs += int(Orb_Database[self.orb_row][ORB_FCS_COL]) + fcs_add
        self.res += int(Orb_Database[self.orb_row][ORB_RES_COL]) + res_add
        self.agi += int(Orb_Database[self.orb_row][ORB_AGI_COL]) + agi_add
        self.crit_c += int(Orb_Database[self.orb_row][ORB_CC_COL])
        self.crit_d += int(Orb_Database[self.orb_row][ORB_CD_COL])
        # print("printing orb element: " + Orb_Database[self.orb_row][ORB_ELEMENT_COL])
        if not Orb_Database[self.orb_row][ORB_ELEMENT_COL] == "0":
            self.element += Orb_Database[self.orb_row][ORB_ELEMENT_COL]

    def set_multiplier(self, girl, element):
        self.multiplier = 0.2
        if self.girl == girl:
            self.multiplier += 0.05
        if element in self.element:
            self.multiplier += 0.05
        if self.multiplier > 0.3:
            print("multiplier is out of whack!")


class CombinedDress:
    def __init__(self, dress):
        self.main_element = dress.element
        self.main_girl = dress.girl
        self.total_hp = 0
        self.total_atk = 0
        self.total_def = 0
        self.total_fcs = 0
        self.total_res = 0
        self.total_agi = 0
        self.total_crit_c = 0
        self.total_crit_d = 0
        self.dress_rows = numpy.array([])
        self.orb_rows = numpy.array([])
        self.num_dress = 0
        self.add_dress(dress)

    def add_dress(self, dress):
        if self.num_dress == 0:
            self.dress_rows = numpy.append(self.dress_rows, dress.row)
            self.orb_rows = numpy.append(self.orb_rows, dress.orb_row)
            dress.multiplier = 1
            self.num_dress += 1
        elif dress.row in self.dress_rows:
            return 3
        elif dress.orb_row in self.orb_rows:
            return 2
        elif self.num_dress >= 5:
            return 1
        else:
            self.num_dress += 1
            dress.set_multiplier(self.main_girl, self.main_element)
            self.dress_rows = numpy.append(self.dress_rows, dress.row)
            self.orb_rows = numpy.append(self.orb_rows, dress.orb_row)
        self.total_hp += dress.hp * dress.multiplier
        self.total_atk += dress.atk * dress.multiplier
        self.total_def += dress.Def * dress.multiplier
        self.total_fcs += dress.fcs * dress.multiplier
        self.total_res += dress.res * dress.multiplier
        self.total_agi += dress.agi * dress.multiplier
        self.total_crit_c += dress.crit_c
        self.total_crit_d += dress.crit_d
        return 0

    def get_as_array(self):
        # 2D array. Dresses, Orbs, Stats
        gotten_array = [self.dress_rows.tolist(), self.orb_rows.tolist()]
        # gotten_array.append(gotten_array, self.dress_rows.tolist)
        # total stats order: hp atk def fcs res agi crit_c crit_d
        effective_crit_c = (self.total_crit_c + 30)/100
        effective_crit_d = (self.total_crit_d + 150)/100
        atk_with_crit = (1-effective_crit_c) * self.total_atk + effective_crit_c * self.total_atk*effective_crit_d
        total_stats = [self.total_hp, atk_with_crit, self.total_def, self.total_fcs, self.total_res, self.total_agi,
                       self.total_crit_c, self.total_crit_d]
        # total_stats = numpy.append(total_stats, self.total_hp)
        gotten_array.append(total_stats)
        return gotten_array


class FullSquad:
    def __init__(self):
        self.squad_points = 0
        self.dress_arrays = []

    def add_dress(self, combined_dress_array):
        if not self.check_orbs(combined_dress_array[1]):
            return 1
        if not self.check_dresses(combined_dress_array[0]):
            return 1
        if not self.check_speeds(combined_dress_array[2]):
            return 1
        self.dress_arrays.append(combined_dress_array)

    def check_orbs(self, orb_array):
        if len(self.dress_arrays) == 0:
            return True
        for dress_iter in range(0, len(self.dress_arrays)):
            for orb in orb_array:
                if orb in self.dress_arrays[dress_iter][1]:
                    return False
        return True

    def check_dresses(self, dress_array):
        if len(self.dress_arrays) == 0:
            return True
        for orb_iter in range(0, len(self.dress_arrays)):
            for dress in dress_array:
                if dress in self.dress_arrays[orb_iter][0]:
                    return False
        return True

    def check_speeds(self, stats_array):
        if len(self.dress_arrays) == 0:
            return True
        # compare stats_array speed to latest added speed
        if stats_array[5] > (self.dress_arrays[(len(self.dress_arrays) - 1)][2][5] - 1):
            return False
        return True

    def get_full(self):
        return len(self.dress_arrays)


def get_orb_info(row):
    temp_string = ''
    for column in range(0, 16):
        if Orb_Database[row][column] == '0':
            continue
        temp_string += Orb_Database[0][column] + ':' + Orb_Database[row][column] + '\n'
    return temp_string


def clean_output(solution, fitness):
    with open('output.csv', 'w', newline='') as csv_file:
        clean_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        clean_writer.writerow(['Main Dress'] + ['Main Orb'] + ['Subdress 1'] + ['SubOrb 1'] + ['Subdress 2'] + ['SubOrb 2'] + ['Subdress 3'] + ['SubOrb 3'] + ['Subdress 4'] + ['SubOrb 4'])
        solution_iter = 0
        for i in range(0, 4):
            # make the content of each row
            temp_row = [Dress_Database[int(Wanted_Squad[0][i])][NAME_COL]]
            # print(solution[solution_iter])
            temp_row += [get_orb_info(int(solution[solution_iter]))]
            solution_iter += 1
            for a in range(0, 4):
                temp_row += [Dress_Database[int(solution[solution_iter])][NAME_COL]]
                solution_iter += 1
                temp_row += [get_orb_info(int(solution[solution_iter]))]
                solution_iter += 1
            clean_writer.writerow(temp_row)
        clean_writer.writerow(['Fitness: ' + str(fitness)])


def bad_brute_force():
    # how to do it without making a giant matrix with all the dresses:
    # can get away with two for loops using itertools.permutations
    # first one is dresses
    max_fitness = 0
    max_perm = []
    for perm_dress in tqdm(itertools.permutations(range(0, len(Dress_Database)), 16)):
        for perm_orb in tqdm(itertools.permutations(range(1, len(Orb_Database)), 20), leave=False):
            # combine the current permutations (use the order that new_ga_run uses)
            curr_perm = []
            for n in range(0, 4):
                for j in range(0, 4):
                    curr_perm.append(perm_orb[j + 5 * n])
                    curr_perm.append(perm_dress[j + 4 * n])
                curr_perm.append(perm_orb[4 + 5 * n])
            # generate fitness for each array with fitness_func and save the maximum one
            temp_fitness = fitness_func(curr_perm, 0)
            if temp_fitness > max_fitness:
                max_fitness = temp_fitness
                max_perm = curr_perm
    return [max_perm, max_fitness]


def brute_force_after_ga(solution, fitness):
    # after the genetic algorithm is done, go through each element in the solution and iterate through it
    # without changing the other elements. if a better team is found, call this function again.
    max_fit = fitness
    for i in range(0, len(solution)):
        temp_solution = numpy.copy(solution)
        # determine if it's an orb or a dress value, then iterate through it
        # orbs are 0, 2, 4, 6, 8, 9, 11 etc
        # so if index modulo 9 is even, it is an orb
        # temp_len = 0
        temp_start = 0
        if ((i % 9) % 2) == 0:
            temp_len = len(Orb_Database)
            temp_start = 1
        else:
            temp_len = len(Dress_Database)
        for j in range(temp_start, temp_len):
            temp_solution[i] = j
            temp_fit = fitness_func(temp_solution, 0)
            if temp_fit > max_fit:
                # print("found a better team! New Fitness: " + str(temp_fit))
                return brute_force_after_ga(temp_solution, temp_fit)
    return [solution, fitness]


def new_ga_run():
    num_generations = 5000
    num_parents_mating = 4

    fitness_function = fitness_func

    sol_per_pop = 20
    num_genes = 36

    dress_range = range(0, len(Dress_Database))
    orb_range = range(1, len(Orb_Database))
    gene_space = []
    for n in range(0, 4):
        for j in range(0, 4):
            gene_space.append(orb_range)
            gene_space.append(dress_range)
        gene_space.append(orb_range)

    parent_selection_type = "sss"
    keep_parents = 2

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 15

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes)
    print("running genetic algorithm, please wait...")
    ga_instance.run()
    # ga_instance.plot_fitness()
    found_solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # print("Parameters of the best solution : {solution}".format(solution=found_solution))
    # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    # print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    # clean_output(found_solution, solution_fitness)
    print("GA has finished. Brute forcing the last bit of optimization, please wait...")
    output = brute_force_after_ga(found_solution, solution_fitness)
    clean_output(output[0], output[1])
    # filename = 'savedata'
    # ga_instance.save(filename=filename)


def load_ga():
    filename = 'savedata'
    ga_instance = pygad.load(filename=filename)
    ga_instance.run()
    # ga_instance.plot_fitness()
    found_solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # print("Parameters of the best solution : {solution}".format(solution=found_solution))
    # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    # print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    clean_output(found_solution, solution_fitness)
    ga_instance.save(filename=filename)


clean_input_csv()
get_all_dress_orb()
# output = brute_force()
# clean_output(output[0], output[1])
# test_girl = DressClass(0)
# test_girl.add_orb(3)
# test_combined = CombinedDress(test_girl)
# # # print(test_combined.dress_rows)
# test_girl = DressClass(5)
# test_girl.add_orb(18)
# test_combined.add_dress(test_girl)
# # print(test_combined.orb_rows)
# # print(test_combined.get_as_array())
# test_squad = FullSquad(test_combined.get_as_array())
# print(test_squad.add_dress(test_combined.get_as_array()))
new_ga_run()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
