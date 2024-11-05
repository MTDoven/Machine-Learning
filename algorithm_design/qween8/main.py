def print_solution(board):
    for row in board:
        print(" ".join(str(x) for x in row))
    print("\n")


def is_safe(board, row, col):
    # if queen in left
    for i in range(col):
        if board[row][i] == 1:
            return False
    # if queen in diag
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    # if queen in below
    for i, j in zip(range(row, len(board), 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    return True


def solve_nq_util(board, col):
    if col >= len(board):  # found one solution
        print_solution(board)
    for i in range(len(board)):
        if is_safe(board, i, col):
            board[i][col] = 1
            solve_nq_util(board, col + 1)
            board[i][col] = 0


def solve_nq(n):
    board = [[0 for _ in range(n)] for _ in range(n)]
    if not solve_nq_util(board, 0):
        print("Solution does not exist")
        return False
    return True


if __name__ == "__main__":
    n = int(input("Enter the number of queens: "))
    import sys
    with open(f"./qween{n}_ans.txt", "w") as f:
        sys.stdout = f
        solve_nq(n)
