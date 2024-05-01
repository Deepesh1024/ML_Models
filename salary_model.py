import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_dataset.csv')

def cos_func(m,b,points):
    total_error = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        total_error += (y - (m*x + b))**2
    return total_error/float(len(points))

def grad_dec(m_grad, b_grad, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for  i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        m_gradient += (-2/n)*x*(y-(x*m_grad + b_grad))
        b_gradient += (-2/n)*(y-(x*m_grad + b_grad))

    m = m_grad - m_gradient*L
    b = b_grad - b_gradient*L
    return  m,b

def model(years):
    return (m*years + b)

m = 0
b = 0
L = 0.01
epochs = 1000

for i in range(epochs):
    if i%50 == 0:
        print(f"Epoch {i}, Loss: {cos_func(m,b,data)}")
        m,b = grad_dec(m,b,data,L)
    print(f"Slop{m}, Intercept {b}")

exp = float(input("Enter the years of expirience you have :  "))
print(model(exp))

plt.scatter(data.YearsExperience, data.Salary, color = "black")
plt.plot(list(range(0,10)), [m*x +b for x in range(0,10)], color = "red")
plt.show()