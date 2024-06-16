import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Устанавливаем seed для воспроизводимости результатов
np.random.seed(0)

# Создаем 11 случайных значений узлов сетки в диапазоне от 0 до 10
xn = np.sort(np.random.uniform(0, 10, 11))

# Создаем случайные значения функции в узлах сетки
yn = np.random.uniform(-10, 10, 11)


#-------------------- 1

# Строим интерполяционный многочлен Лагранжа
lagrange_poly = lagrange(xn, yn)

# Выводим коэффициенты многочлена Лагранжа
lagrange_poly_coefficients = lagrange_poly.coef

lagrange_poly_coefficients


#-------------------- 2

def newton_divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y

    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    
    return coef[0, :]

def newton_poly(coef, x_data, x):
    n = len(coef) - 1
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x - x_data[n-k])*p
    return p

# Вычисляем разделенные разности
coefficients = newton_divided_diff(xn, yn)

# Находим значение функции в произвольной точке, например x = 5
x_new = 10
y_newton = newton_poly(coefficients, xn, x_new)



#-------------------- 3


# Создаем интерполяционную функцию
f_interp = interp1d(xn, yn, kind='cubic')

# Находим значение функции в произвольной точке, например x = 5
y_interp = f_interp(5)




#-------------------- 4


# Интерполяция с использованием метода Лагранжа
x_vals = np.linspace(min(xn), max(xn), 1000)
y_lagrange = lagrange_poly(x_vals)

# Интерполяция с использованием метода Ньютона
y_newton_vals = [newton_poly(coefficients, xn, x) for x in x_vals]

# Интерполяция с использованием библиотеки scipy
y_interp_vals = f_interp(x_vals)

plt.figure(figsize=(12, 6))
plt.plot(xn, yn, 'o', label='Узлы сетки')
plt.plot(x_vals, y_lagrange, label='Лагранж')
plt.plot(x_vals, y_newton_vals, label='Ньютон')
plt.plot(x_vals, y_interp_vals, label='Scipy cubic')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Интерполяционные функции')
plt.grid(True)
plt.show()
