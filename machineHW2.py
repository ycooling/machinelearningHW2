import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rc

# 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"   # 윈도우 기본 맑은고딕 폰트
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
except:
    print("한글 폰트 설정 실패. 시스템에 맑은고딕 폰트가 있는지 확인하세요.")

# 데이터 로드 - 정확한 경로 사용
data_path = r'C:\Kdigital\data0405.csv'
df = pd.read_csv(data_path)

# 데이터를 x, y 배열로 변환
x = df.iloc[:, 0].values  # 첫 번째 열을 x 값으로
y = df.iloc[:, 1].values  # 두 번째 열을 y 값으로

# 데이터 확인
print(f"불러온 데이터 포인트 수: {len(x)}")
print(f"처음 5개 데이터 포인트: {list(zip(x[:5], y[:5]))}")

# 데이터 시각화
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='데이터 포인트')
plt.xlabel('x')
plt.ylabel('y')
plt.title('데이터 분포')
plt.grid(True)
plt.legend()
plt.show()

# 1. 최소제곱법을 이용한 예측
# 1-1. 선형 예측 모델
def least_squares_linear(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    a = numerator / denominator
    b = y_mean - a * x_mean
    
    return a, b

# 1-2. 다항 회귀 (비선형 예측 모델)
def least_squares_polynomial(x, y, degree):
    X = np.column_stack([x**i for i in range(degree+1)])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs

# 최소제곱법 선형 모델 피팅
a_ls, b_ls = least_squares_linear(x, y)
print(f"1-1. 최소제곱법 선형 모델 파라미터: y = {a_ls:.4f}x + {b_ls:.4f}")

# 2차 다항식 모델 피팅
quad_coeffs_ls = least_squares_polynomial(x, y, 2)
print(f"1-2. 최소제곱법 2차 모델 파라미터: y = {quad_coeffs_ls[2]:.4f}x^2 + {quad_coeffs_ls[1]:.4f}x + {quad_coeffs_ls[0]:.4f}")

# 10차 다항식 모델 피팅
tenth_coeffs_ls = least_squares_polynomial(x, y, 10)
print(f"1-2. 최소제곱법 10차 모델 파라미터:")
for i, coef in enumerate(tenth_coeffs_ls):
    print(f"    계수 {i}: {coef:.4f}")

# 최소제곱법 모델 시각화
x_new = np.linspace(-2.1, 1.1, 100)
y_linear_ls = a_ls * x_new + b_ls
y_quad_ls = np.sum([quad_coeffs_ls[i] * x_new**i for i in range(3)], axis=0)
y_tenth_ls = np.sum([tenth_coeffs_ls[i] * x_new**i for i in range(11)], axis=0)

plt.figure(figsize=(15, 10))

# 선형 모델
plt.subplot(2, 2, 1)
plt.scatter(x, y, label='데이터 포인트')
plt.plot(x_new, y_linear_ls, 'r-', label=f'선형 모델: y = {a_ls:.4f}x + {b_ls:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('1-1. 최소제곱법 선형 모델')
plt.grid(True)
plt.legend()

# 비선형 모델 (2차 및 10차)
plt.subplot(2, 2, 2)
plt.scatter(x, y, label='데이터 포인트')
plt.plot(x_new, y_quad_ls, 'r-', label='2차 다항식')
plt.plot(x_new, y_tenth_ls, 'g-', label='10차 다항식')
plt.xlabel('x')
plt.ylabel('y')
plt.title('1-2. 최소제곱법 비선형 모델')
plt.grid(True)
plt.legend()

# 2. 경사하강법을 이용한 예측
# 2-1. 선형 모델 (경사하강법)
def gradient_descent_linear(x, y, learning_rate=0.001, epochs=500):
    a = 0
    b = 0
    n = len(x)
    
    a_history = []
    b_history = []
    loss_history = []
    
    for epoch in range(epochs):
        # 예측값 계산
        y_pred = a * x + b
        
        # 오차 계산
        error = y_pred - y
        
        # 기울기 계산
        grad_a = (2/n) * np.sum(error * x)
        grad_b = (2/n) * np.sum(error)
        
        # 파라미터 업데이트
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        
        # 손실 계산
        loss = np.mean((y_pred - y) ** 2)
        
        a_history.append(a)
        b_history.append(b)
        loss_history.append(loss)
    
    return a, b, a_history, b_history, loss_history

# 2-2. 다항 회귀 모델 (경사하강법)
def gradient_descent_polynomial(x, y, degree, learning_rate=0.001, epochs=500):
    coeffs = np.zeros(degree + 1)
    n = len(x)
    
    # 특성 행렬 생성
    X = np.column_stack([x**i for i in range(degree+1)])
    
    coeff_history = []
    loss_history = []
    
    for epoch in range(epochs):
        # 예측값 계산
        y_pred = np.sum([coeffs[i] * x**i for i in range(degree+1)], axis=0)
        
        # 오차 계산
        error = y_pred - y
        
        # 기울기 계산
        gradients = np.array([(2/n) * np.sum(error * x**i) for i in range(degree+1)])
        
        # 파라미터 업데이트
        coeffs -= learning_rate * gradients
        
        # 손실 계산
        loss = np.mean((y_pred - y) ** 2)
        
        coeff_history.append(coeffs.copy())
        loss_history.append(loss)
    
    return coeffs, coeff_history, loss_history

# 경사하강법 선형 모델 피팅
learning_rate = 0.001  # 요구사항에 명시된 학습률
epochs = 500  # 요구사항에 명시된 반복 횟수
a_gd, b_gd, a_history, b_history, loss_history_linear = gradient_descent_linear(x, y, learning_rate, epochs)
print(f"2-1. 경사하강법 선형 모델 파라미터: y = {a_gd:.4f}x + {b_gd:.4f}")
print(f"    최종 손실: {loss_history_linear[-1]:.4f}")
print(f"    학습률: {learning_rate}, 반복 횟수: {epochs}")

# 경사하강법 2차 모델 피팅
quad_coeffs_gd, quad_coeff_history, loss_history_quad = gradient_descent_polynomial(x, y, 2, learning_rate, epochs)
print(f"2-2. 경사하강법 2차 모델 파라미터: y = {quad_coeffs_gd[2]:.4f}x^2 + {quad_coeffs_gd[1]:.4f}x + {quad_coeffs_gd[0]:.4f}")
print(f"    최종 손실: {loss_history_quad[-1]:.4f}")

# 경사하강법 4차 모델 피팅 (과제 요구사항에서 4차 다항식 사용)
quartic_coeffs_gd, quartic_coeff_history, loss_history_quartic = gradient_descent_polynomial(x, y, 4, learning_rate, epochs)
print(f"2-2. 경사하강법 4차 모델 파라미터:")
for i, coef in enumerate(quartic_coeffs_gd):
    print(f"    계수 {i}: {coef:.4f}")
print(f"    최종 손실: {loss_history_quartic[-1]:.4f}")

# 경사하강법 모델 시각화
y_linear_gd = a_gd * x_new + b_gd
y_quad_gd = np.sum([quad_coeffs_gd[i] * x_new**i for i in range(3)], axis=0)
y_quartic_gd = np.sum([quartic_coeffs_gd[i] * x_new**i for i in range(5)], axis=0)

# 경사하강법 선형 모델
plt.subplot(2, 2, 3)
plt.scatter(x, y, label='데이터 포인트')
plt.plot(x_new, y_linear_gd, 'r-', label=f'선형 모델: y = {a_gd:.4f}x + {b_gd:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2-1. 경사하강법 선형 모델')
plt.grid(True)
plt.legend()

# 경사하강법 비선형 모델
plt.subplot(2, 2, 4)
plt.scatter(x, y, label='데이터 포인트')
plt.plot(x_new, y_quad_gd, 'r-', label='2차 다항식')
plt.plot(x_new, y_quartic_gd, 'g-', label='4차 다항식')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2-2. 경사하강법 비선형 모델')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 3. 두 방법 비교
print("\n3. 두 방법 비교:")
print("선형 모델 파라미터 비교:")
print(f"최소제곱법: y = {a_ls:.4f}x + {b_ls:.4f}")
print(f"경사하강법: y = {a_gd:.4f}x + {b_gd:.4f}")
print("\n2차 모델 파라미터 비교:")
print(f"최소제곱법: y = {quad_coeffs_ls[2]:.4f}x^2 + {quad_coeffs_ls[1]:.4f}x + {quad_coeffs_ls[0]:.4f}")
print(f"경사하강법: y = {quad_coeffs_gd[2]:.4f}x^2 + {quad_coeffs_gd[1]:.4f}x + {quad_coeffs_gd[0]:.4f}")

# 두 방법 비교 시각화
plt.figure(figsize=(15, 10))

# 선형 모델 비교
plt.subplot(2, 1, 1)
plt.scatter(x, y, label='데이터 포인트')
plt.plot(x_new, y_linear_ls, 'r-', label=f'최소제곱법: y = {a_ls:.4f}x + {b_ls:.4f}')
plt.plot(x_new, y_linear_gd, 'g-', label=f'경사하강법: y = {a_gd:.4f}x + {b_gd:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('선형 모델 비교')
plt.grid(True)
plt.legend()

# 2차 모델 비교
plt.subplot(2, 1, 2)
plt.scatter(x, y, label='데이터 포인트')
plt.plot(x_new, y_quad_ls, 'r-', label=f'최소제곱법: y = {quad_coeffs_ls[2]:.4f}x^2 + {quad_coeffs_ls[1]:.4f}x + {quad_coeffs_ls[0]:.4f}')
plt.plot(x_new, y_quad_gd, 'g-', label=f'경사하강법: y = {quad_coeffs_gd[2]:.4f}x^2 + {quad_coeffs_gd[1]:.4f}x + {quad_coeffs_gd[0]:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2차 모델 비교')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 경사하강법 학습 과정 시각화
plt.figure(figsize=(15, 5))

# 손실 변화 그래프
plt.subplot(1, 3, 1)
plt.plot(loss_history_linear, label='선형 모델')
plt.plot(loss_history_quad, label='2차 모델')
plt.plot(loss_history_quartic, label='4차 모델')
plt.xlabel('Epoch')
plt.ylabel('손실')
plt.title('경사하강법 학습 과정: 손실 변화')
plt.grid(True)
plt.legend()

# 선형 모델 파라미터 변화
plt.subplot(1, 3, 2)
plt.plot(a_history, label='기울기 (a)')
plt.plot(b_history, label='절편 (b)')
plt.xlabel('Epoch')
plt.ylabel('파라미터 값')
plt.title('선형 모델 파라미터 변화')
plt.grid(True)
plt.legend()

# 다항식 모델 파라미터 변화 (2차 모델)
plt.subplot(1, 3, 3)
plt.plot([coeff[0] for coeff in quad_coeff_history], label='상수항')
plt.plot([coeff[1] for coeff in quad_coeff_history], label='1차항')
plt.plot([coeff[2] for coeff in quad_coeff_history], label='2차항')
plt.xlabel('Epoch')
plt.ylabel('파라미터 값')
plt.title('2차 모델 파라미터 변화')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()