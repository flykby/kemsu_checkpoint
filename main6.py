import numpy as np
import matplotlib.pyplot as plt

# Генерация случайных коэффициентов
np.random.seed(0)
n_attempts = 100

def random_coefficients():
    return np.random.uniform(-10, 10, size=(3, 3)), np.random.uniform(-10, 10, size=3)

def gauss_elimination(A, B):
    n = len(B)
    for i in range(n):
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if A[max_row, i] == 0:
            raise np.linalg.LinAlgError("Матрица вырождена")
        A[[i, max_row]] = A[[max_row, i]]
        B[[i, max_row]] = B[[max_row, i]]
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            B[j] -= factor * B[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (B[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

def gauss_jordan_elimination(A, B):
    n = len(B)
    M = np.hstack((A, B.reshape(-1, 1)))
    for i in range(n):
        max_row = np.argmax(np.abs(M[i:, i])) + i
        if M[max_row, i] == 0:
            raise np.linalg.LinAlgError("Матрица вырождена")
        M[[i, max_row]] = M[[max_row, i]]
        M[i] = M[i] / M[i, i]
        for j in range(n):
            if i != j:
                M[j] -= M[j, i] * M[i]
    return M[:, -1]

def gauss_seidel(A, B, x0, epsilon=0.1, max_iter=1000):
    n = len(B)
    x = x0.copy()
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            if A[i, i] == 0:
                raise np.linalg.LinAlgError("Нулевой элемент на диагонали")
            x_new[i] = (B[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
            return x_new
        x = x_new
    raise np.linalg.LinAlgError("Метод Гаусса-Зейделя не сошелся")

# Попытка генерации случайных коэффициентов
for attempt in range(n_attempts):
    try:
        A, B = random_coefficients()
        solution_gauss = gauss_elimination(A.copy(), B.copy())
        solution_jordan = gauss_jordan_elimination(A.copy(), B.copy())
        solution_seidel = gauss_seidel(A.copy(), B.copy(), np.zeros(3))
        break
    except np.linalg.LinAlgError:
        if attempt == n_attempts - 1:
            A = np.zeros((3, 3))
            B = np.zeros(3)
            for i in range(3):
                A[i, :] = list(map(float, input(f"Введите коэффициенты a{i+1}, b{i+1}, c{i+1}: ").split()))
                B[i] = float(input(f"Введите свободный член для уравнения {i+1}: "))
            solution_gauss = gauss_elimination(A.copy(), B.copy())
            solution_jordan = gauss_jordan_elimination(A.copy(), B.copy())
            solution_seidel = gauss_seidel(A.copy(), B.copy(), np.zeros(3))

# Решение с использованием библиотеки numpy
solution_numpy = np.linalg.solve(A, B)

# Вывод решений
print("Решение методом Гаусса:", solution_gauss)
print("Решение методом Гаусса-Жордана:", solution_jordan)
print("Решение методом Гаусса-Зейделя:", solution_seidel)
print("Решение с использованием numpy:", solution_numpy)

# Построение графика для одной из систем
x_values = np.linspace(-10, 10, 100)
y_values = 5 - x_values

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label="a1 + b1 + c1 = 5")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.title("График уравнения")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
