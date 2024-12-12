import torch


# steps 1 & 2 of the Hungarian Algorithm
def init_candidate_solution(cost_matrix: torch.tensor) -> torch.tensor:
    # step 1
    (min_rows, _) = torch.min(cost_matrix, 1, True)
    cost_matrix_step1 = cost_matrix - min_rows
    # step 2
    (min_columns, _) = torch.min(cost_matrix_step1, 0, True)
    cost_matrix_step2 = cost_matrix_step1 - min_columns
    return cost_matrix_step2


# step 3 & 4 of the Hungarian Algorithm
def evaluate_optimality_criteria(Cost_matrix: torch.tensor) -> (bool, list, list):
    zeros_positions = (Cost_matrix == 0).nonzero(as_tuple=True)
    # marked with 0
    zeros_marked_by_line = torch.ones(zeros_positions[0].shape, dtype=torch.bool)
    rows_marked = []
    cols_marked = []
    while torch.sum(zeros_marked_by_line) > 0:
        (max_zeroes_row, ind_row) = zeros_positions[0][zeros_marked_by_line].mode()
        (max_zeroes_col, ind_col) = zeros_positions[1][zeros_marked_by_line].mode()
        freq_row = (zeros_positions[0].eq(max_zeroes_row)).sum()
        freq_col = (zeros_positions[1].eq(max_zeroes_col)).sum()
        if freq_row >= freq_col:
            rows_marked.append(int(max_zeroes_row))
            zeros_marked_by_line[zeros_positions[0] == max_zeroes_row] = 0
        else:
            cols_marked.append(int(max_zeroes_col))
            zeros_marked_by_line[zeros_positions[1] == max_zeroes_col] = 0
    return len(rows_marked) + len(cols_marked) >= Cost_matrix.size(0), rows_marked, cols_marked


# step 5 of the Hungarian Algorithm
def Update_Cost_Matrix(cost_matrix: torch.tensor, rows_marked: list, cols_marked: list) -> torch.tensor:
    # define the minimum scalar value of the cost matrix with the rows and columns marked
    min_val = torch.min(cost_matrix[~torch.isin(torch.arange(cost_matrix.size(0)), torch.tensor(rows_marked)), :]
                                   [:, ~torch.isin(torch.arange(cost_matrix.size(1)), torch.tensor(cols_marked))])
    cost_matrix[~torch.isin(torch.arange(cost_matrix.size(0)), torch.tensor(rows_marked)), :] = \
        cost_matrix[~torch.isin(torch.arange(cost_matrix.size(0)), torch.tensor(rows_marked)), :] - min_val
    cost_matrix[:, torch.isin(torch.arange(cost_matrix.size(1)), torch.tensor(cols_marked))] = \
        cost_matrix[:, torch.isin(torch.arange(cost_matrix.size(1)), torch.tensor(cols_marked))] + min_val
    return cost_matrix


def assesment(cost_function: torch.tensor, cost_matrix: torch.tensor) -> list:
    solution = []
    positions = (cost_matrix == 0).nonzero(as_tuple=True)
    while positions[0].size()[0] > 0:
        # find row
        _, assess = torch.unique_consecutive(positions[0], return_counts=True)
        min, index = torch.min(assess, 0)
        one_zero_row = positions[0][torch.sum(assess[:index])]
        one_zero_column = positions[1][torch.sum(assess[:index])]
        # add solution
        solution.append(cost_function[one_zero_row, one_zero_column])
        # remove row
        all_zeros_in_row = torch.isin(positions[0], one_zero_row)
        positions = (positions[0][all_zeros_in_row.logical_not()], positions[1][all_zeros_in_row.logical_not()])
        # remove column
        all_zeros_in_col = torch.isin(positions[1], one_zero_column)
        positions = (positions[0][all_zeros_in_col.logical_not()], positions[1][all_zeros_in_col.logical_not()])
    return solution


def run_hungarian_algorithm(cost_matrix: torch.tensor) -> None:
    print(f"First\n\t{cost_matrix}")
    cost_matrix_step2 = init_candidate_solution(cost_matrix)
    print(f"Initialized Solution:\n\t{cost_matrix_step2}\n")
    v, rows_marked, cols_marked = evaluate_optimality_criteria(cost_matrix_step2)
    new_matrix = cost_matrix_step2
    while not v:
        print(f"Is optimal? \t{v}")
        print(f"Rows marked: \t{rows_marked}")
        print(f"Cols marked: \t{cols_marked}")
        new_matrix = Update_Cost_Matrix(new_matrix, rows_marked, cols_marked)
        print(f"new iteration \n{new_matrix}\n")
        v, rows_marked, cols_marked = evaluate_optimality_criteria(new_matrix)
    print(f"Optimal = {v}")
    sol = assesment(cost_matrix, new_matrix)
    print(f"Assesments: \t{sol}")
    print(f"Solution:\t{torch.sum(torch.stack(sol)).item()}")




if __name__ == "__main__":
    Cost_function1 = torch.tensor(
        [[90.0, 75.0, 75.0, 80.0],
         [35.0, 85.0, 55.0, 65.0],
         [125., 95., 90., 105.],
         [45.0, 110.0, 95.0, 115.0]]
    )
    Cost_function2 = torch.tensor(
        [[1., 8., 15., 22.],
         [13., 18., 23., 28.],
         [13., 18., 23., 28.],
         [19., 23., 27., 31.]]
    )
    print(f"Cost Function 1:\n{Cost_function1}")
    print(f"Cost Function 2:\n{Cost_function2}")
    run_hungarian_algorithm(Cost_function1)
    run_hungarian_algorithm(Cost_function2)
