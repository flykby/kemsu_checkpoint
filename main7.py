import numpy as np
import matplotlib.pyplot as plt

# Определение функции и её производной
def f(x):
    return x**3 - 2*x**2 - 4*x + 5

def f_prime(x):
    return 3*x**2 - 4*x - 4

# Метод Ньютона
def newton_method(f, f_prime, x0, epsilon=1e-6, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if fpx == 0:
            raise ValueError("Производная равна нулю, метод не применим.")
        x_new = x - fx / fpx
        if abs(x_new - x) < epsilon:
            return x_new
        x = x_new
    raise ValueError("Метод не сошелся за максимальное количество итераций")

# Функция для повторного ввода начального приближения
def find_root(f, f_prime, initial_guesses):
    for x0 in initial_guesses:
        try:
            root = newton_method(f, f_prime, x0)
            return root
        except ValueError as e:
            print(f"Начальное приближение x0={x0} не удалось: {e}")
    raise ValueError("Не удалось найти корень для заданных начальных приближений")

# Список начальных приближений
initial_guesses = [30.0, 20.0, 7.0, 4.0, 1.0, -0.5]

# Попытка найти корень
try:
    root = find_root(f, f_prime, initial_guesses)
    print(f"Найденный корень: {root}")
except ValueError as e:
    print(e)

# Построение графика функции
x_values = np.linspace(-3, 3, 400)
y_values = f(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label="y = x^3 - 2x^2 - 4x + 5")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
if 'root' in locals():
    plt.scatter(root, f(root), color='red', zorder=5, label=f"Root at x={root:.6f}")
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.title("График функции y = x^3 - 2x^2 - 4x + 5")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
