from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.features import Rank2D
from seaborn import heatmap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

global X_X
X_X=[]


def make_graphic_ost(ishod_data,Y_prognoz_matrix):
    ost_matrix=[[0.0]*len(ishod_data[0])]
    h=len(ishod_data)-1
    for e in range(len(ishod_data[0])):
        ost_matrix[0][e]=float(ishod_data[h][e]-Y_prognoz_matrix[e])

    fig,axs=plt.subplots(nrows=h,ncols=1)

    fig.suptitle('Графики остатков')

    for k in range(len(ishod_data)-1):
        axs[k].scatter(ishod_data[k],ost_matrix)
        axs[k].set_title("X"+str(k+1),loc='left',fontsize=10)

    plt.show()


def find_correlation(ishod_data,n,matrix_corr):
    global X_X
    buff=-1
    for i in range(len(matrix_corr)-1):
        for i1 in range(len(matrix_corr[i])-1):
            if matrix_corr[i][i1]==1.0:
                None
            elif matrix_corr[i][i1]>=0.8:
                print("Наличие мультиколлинеарности между "+"X"+str(i+1)+" и "+"X"+str(i1+1))
                buff=i
    if buff<=-1:
        return matrix_corr
    else:
        print("Исключаем из регрессии X"+str(buff+1))
        X_X.append(buff)
        X_X.sort()
        ishod_data1=[]
        for i2 in range(len(ishod_data)):
            if (i2==buff):
                i2+=1
            else:
                ishod_data1.append(ishod_data[i2])
        matrix_corr1=matrix_correlation(ishod_data1,n)
        return find_correlation(ishod_data1,n,matrix_corr1)


def Student_t_distribution(matrix_corr,n,p):
    DF=n-p
    Student_DF=2.642983063369744
    i=len(matrix_corr)-1
    matrix_t_see=[[0.0]*i]
    for i1 in range(len(matrix_corr[i])-1):
        matrix_t_see[0][i1]=float(abs(matrix_corr[i][i1])*((sqrt(DF-1))/(sqrt(1-matrix_corr[i][i1]*matrix_corr[i][i1]))))
        if matrix_t_see[0][i1]>Student_DF:
            print("Коэффициент X"+str(i1+1)+" корреляции статистически значим")
        else:
            print("Коэффициент X"+str(i1+1)+" корреляции статистически не значим")
    print("-"*100)


def make_normalized(matrix,index):
    matrix_norm=transpose_matrix(matrix)
    matrix_norm=pd.DataFrame(matrix_norm,columns=['X1','X2','X3','X4','X5','X6','Y'])
    df_norm = (matrix_norm-matrix_norm.min ())/ (matrix_norm.max () - matrix_norm.min ())
    df_norm=np.array(df_norm).round(5)
    np.savetxt(f'Normalized_value_matrix{index}.txt', df_norm)
    print("Нормализация данных")
    print_matrix(df_norm)
    print("-"*100)
    return df_norm


def final(n,p,k,Y_prognoz_matrix,ishod_data,srez_massiv,matrix_koeff):
    print("-"*100)
    print(" n = "+str(n))
    print(" p = "+str(p))
    print(" k = "+str(k))
    print(" DF = "+str(n-p))
    for i in range(len(srez_massiv)-1):
        t=i+1
        print(" Сред.знач  X"+str(t)+" : "+str(srez_massiv[i]))
    print(" Сред.знач  Y : "+str(srez_massiv[t]))
    sum_Y=0.0
    matrix_y=ishod_data[t]
    for i in range(len(ishod_data[t])):
        sum_Y=sum_Y+ishod_data[t][i]
    sum_Y=sum_Y/n
    sum_Y_2=0.0
    for i in range(len(ishod_data[t])):
        sum_Y_2=sum_Y_2+ishod_data[t][i]*ishod_data[t][i]
    sum_Y_2=sum_Y_2/n
    SST=(sum_Y_2-sum_Y*sum_Y)*n
    print(" Общая сумма квадратов SST : "+str(SST))
    sum_prog_Y=0.0
    for i in range(len(Y_prognoz_matrix)):
        sum_prog_Y=sum_prog_Y+Y_prognoz_matrix[i]
    sum_prog_Y=sum_prog_Y/n
    sum_prog_Y_2=0.0
    for i in range(len(Y_prognoz_matrix)):
        sum_prog_Y_2=sum_prog_Y_2+Y_prognoz_matrix[i]*Y_prognoz_matrix[i]
    sum_prog_Y_2=sum_prog_Y_2/n
    SSR=(sum_prog_Y_2-sum_prog_Y*sum_prog_Y)*n
    print(" Регрессия суммы квадратов SSR : "+str(SSR))
    print(" Ошибка суммы квадратов SSE : "+str(SST-SSR))
    print(" Средний квадрат регрессии MSR : "+str(SSR/k))
    print(" Cреднеквадратичная разница MSE : "+str((SST-SSR)/(n-p)))
    print(" Стандартная ошибка регрессии SEY : "+str(sqrt((SST-SSR)/(n-p))))
    print(" F-статистика : "+str((SSR/k)/((SST-SSR)/(n-p))))
    print(" Коэффициент детерминации R^2 : "+str(SSR/SST))
    print(" Скорректированный коэффициент детерминации R^2 adj : "+str(1-((SST-SSR)/(n-p))/(SST/(n-1))))
    matrix_ost_y=[]
    for i in range(len(matrix_y)):
        matrix_ost_y.append(float(matrix_y[i]-Y_prognoz_matrix[i]))
    # print(matrix_ost_y)
    ma1=[]
    for i in range(len(matrix_ost_y)-1):
        ma1.append(float(matrix_ost_y[i]))
    print("\n Остатки = Y - предсказанное Y \n")
    print(ma1)
    ma2=[]
    for i1 in range(len(matrix_ost_y)-1):
        ma2.append(float(matrix_ost_y[i1+1]))
    print()
    ma3=[]
    for i2 in range(len(ma1)):
        ma3.append(float((ma1[i2]-ma2[i2])*(ma1[i2]-ma2[i2])))
    base=0.0
    for i3 in range(len(ma3)):
        base=base+ma3[i3]
    base1=0.0
    for i4 in range(len(matrix_ost_y)):
        base1=base1+(matrix_ost_y[i4]*matrix_ost_y[i4])
    print("\n Статистика теста Дарбина-Уотсона DW : "+str(base/base1))
    if float(base/base1)<1.5 or float(base/base1)>2.5:
        if (float(base/base1)<1.5 and float(base/base1)>=1.4) or (float(base/base1)>2.5 and float(base/base1)<=2.6):
            print(" существует незначительная проблема автокорреляции")
        else:
            print(" существует проблема автокорреляции")
    elif float(base/base1)>1.5 or float(base/base1)<2.5:
        print(" автокорреляция не вызывает беспокойства")
    print("-"*100)
    print(" Уравнение")
    buff=" Y = "+"("+str(matrix_koeff[0])+") + "
    counter=len(matrix_koeff)-1
    for i in range(counter):
            buff=buff+"("+str(matrix_koeff[i+1])+")*X"+str(i+1)+" + "
    print(buff[:-2])
    print("-"*100)


def print_prognoz_Y(massiv_prognoz_Y):
    print("Прогноз по Y (собственная реализация)")
    for k in range(len(massiv_prognoz_Y)):
        print(str(massiv_prognoz_Y[k]))


def prognoz_Y(matrix_koeff,ishod_data):
    matrix=[]
    counter=len(matrix_koeff)-1
    buff=0.0
    for i1 in range(len(ishod_data[0])):
        buff=matrix_koeff[0]
        for i in range(counter):
            buff=buff+matrix_koeff[i+1]*ishod_data[i][i1]
        matrix.append(buff)
    return matrix


def matrix_library_regression(ishod_data_new):
    matrix2=transpose_matrix(ishod_data_new)
    matrix2 = pd.DataFrame(matrix2,columns=['X1','X2','X3','X4','X5','X6','Y'])
    X=matrix2[["X1","X2","X3","X4","X5","X6"]]
    Y=matrix2[["Y"]]
    coef_matrix = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    print("-"*100)
    print(f'Коэффициенты рассчитанные по формуле c использованием библиотеки pandas {coef_matrix.T}')
    print("-"*100)
    model = LinearRegression().fit(X, Y)
    coef_sklearn = model.coef_.T
    print(f'Коэффициенты рассчитанные с использованием библиотеки sklearn {coef_sklearn.T[0]}')


def print_standardized(ishod_data_new):
    print("-"*100)
    matrix1=transpose_matrix(ishod_data_new)
    print("Стандартизированные переменные (собственная реализация) у, x1, x2, x3, x4, x5, x6 :")
    print_matrix(matrix1)
    matrix2=transpose_matrix(ishod_data_new)
    matrix2 = pd.DataFrame(matrix2,columns=['X1','X2','X3','X4','X5','X6','Y'])
    df_new = (matrix2-matrix2.mean ())/matrix2.std()
    print("-"*100)
    print("Стандартизированные переменные (библиотека pandas) у, x1, x2, x3, x4, x5, x6 :")
    print(df_new)


def eliminate(r1, r2, col, target=0):
    fac = (r2[col] - target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]


def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i + 1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                raise ValueError("Matrix is not invertible")
        for j in range(i + 1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a) - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a


def inverse(a):
    tmp = [[] for _ in a]
    for i, row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0] * i + [1] + [0] * (len(a) - i - 1))
    gauss(tmp)
    ret = []
    for i in range(len(tmp)):
        ret.append(tmp[i][len(tmp[i]) // 2:])
    return ret


def mul_matrix(matrix1,matrix2):
    prom_res = [[0.0] * len(matrix1) for i in range(len(matrix2[0]))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                prom_res[i][j] += matrix1[i][k] * matrix2[k][j]
    return prom_res


def next_mul(matrix1,matrix2):
    buff=[[0.0]*len(matrix1[0]) for i in range(len(matrix2))]
    for i in range(len(matrix2)):
        for j in range(len(matrix1[0])):
            for k in range(len(matrix2[0])):
                buff[i][j] += matrix2[i][k] * matrix1[k][j]
    return buff


def matrix_regressin(ishod_data):
    print("-"*100)
    print("Коэффициенты регрессии (собственная реализация)")
    t = 0
    x_size_n = len(ishod_data) - 1
    x_size_m = len(ishod_data[0])
    matrix_x = [[1] * (x_size_m) for i in range(x_size_n + 1)]
    for i in range(len(matrix_x) - 1):
        t+=1
        for i1 in range(len(matrix_x[0])):
            matrix_x[i + 1][i1] = float(ishod_data[i][i1])

    matrix_x_transpose = transpose_matrix(matrix_x)
    # print("Transpose X : " + str(matrix_x_transpose))

    matrix_y=[]
    matrix_y.append(ishod_data[t])

    prom_res = mul_matrix(matrix_x,matrix_x_transpose)
    # print("Mul_matrix : X*X(transpose)"+str(prom_res))

    prom_new = inverse(prom_res)
    # print(" inverse_matrix X*X(transpose) : "+str(prom_new))

    matrix_y_transpose=transpose_matrix(matrix_y)
    # print("Transpose Y :"+ str(matrix_y_transpose))

    buff=next_mul(matrix_y_transpose,matrix_x)
    result=[[0.0]*len(buff[0]) for i in range(len(prom_new))]
    for i in range(len(prom_new)):
        for j in range(len(buff[0])):
            for k in range(len(prom_new[0])):
                result[i][j] += prom_new[i][k] * buff[k][j]
    
    for fu in range(len(result)):
        print("b"+str(fu)+" = "+str(result[fu]))
    print("-"*100)
    result_fin=[]
    for i in range(len(result)):
        for i1 in range(len(result[0])):
            result_fin.append(float(result[i][i1]))
    return result_fin


def make_standardized(massiv_deviation_average,massiv_standart_devition,index):
    massiv_standart_value=[[0.0]*len(massiv_deviation_average[0]) for k in range(len(massiv_deviation_average))]
    for r in range(len(massiv_deviation_average)):
        for r1 in range(len(massiv_deviation_average[r])):
            massiv_standart_value[r][r1]=float(massiv_deviation_average[r][r1]/massiv_standart_devition[r])
    print("Save Standardized_value_matrix.txt in folder")
    res = np.array(massiv_standart_value).round(5)
    np.savetxt(f'Standardized_value_matrix{index}.txt', res)
    return massiv_standart_value


def print_standart_devition(massiv_standart_devition):
    print("-"*100)
    print("Стандарт отклонения (собственная реализация)")
    t=0
    for i in range(len(massiv_standart_devition)-1):
        t=i+1
        print("Стандарт отклонения  X"+str(t)+" : "+str(massiv_standart_devition[i]))
    print("Стандарт отклонения  Y : "+str(massiv_standart_devition[t]))
    print("-"*100+"\n")


def filas(lista):
    summed_list = [sum(i) for i in lista]
    return summed_list


def standart_devition(massiv_deviation_average,n):
    massiv_deviation_average1=[[0.0]*len(massiv_deviation_average[0]) for k in range(len(massiv_deviation_average))]
    for r in range(len(massiv_deviation_average)):
        for r1 in range(len(massiv_deviation_average[r])):
            massiv_deviation_average1[r][r1]=float(massiv_deviation_average[r][r1])
    n=n-1
    for i in range(len(massiv_deviation_average1)):
        for i1 in range(len(massiv_deviation_average1[i])):
            massiv_deviation_average1[i][i1]=float(massiv_deviation_average1[i][i1]*massiv_deviation_average1[i][i1])
    standart=filas(massiv_deviation_average1)
    for t in range(len(standart)):
        standart[t]=float(sqrt(standart[t]/n))
    return standart


def deviation_from_the_average(ishod_data,srez_massiv):
    matrix=[]
    i=0
    for i in range(len(ishod_data)):
        for i1 in range(len(ishod_data[i])):
            matrix.append(float(ishod_data[i][i1]-srez_massiv[i]))
    matrix1=[[0]*len(ishod_data[i]) for t1 in range(len(ishod_data))]
    c=0
    for t in range(len(ishod_data)):
        for i1 in range(len(ishod_data[i])):
            matrix1[t][i1]=matrix[c]
            c+=1
    return matrix1


def print_deviation_from_the_average(massiv_otkl):
    print("-"*100)
    print("Отклонение от среднего переменных (собственная реализация)")
    t=0
    for i in range(len(massiv_otkl)-1):
        t=i+1
        print("Отклонение от среднего  X"+str(t)+" : "+str(massiv_otkl[i])+"\n")
    print("Отклонение от среднего  Y : "+str(massiv_otkl[t]))
    print("-"*100+"\n")


def print_average_value(srez_massiv):
    print("-"*100)
    print("собственная реализация")
    t=0
    for i in range(len(srez_massiv)-1):
        t=i+1
        print("Сред.знач  X"+str(t)+" : "+str(srez_massiv[i]))
    print("Сред.знач  Y : "+str(srez_massiv[t]))
    print("-"*100+"\n")


def make_massiv_average(ishod_data):
    srez_massiv = []
    i = 0
    for i in range(len(ishod_data)):
        srez_massiv.append(srez_znack(n, ishod_data, i))
    return srez_massiv


def srez_znack(len_n,ishod_data,index):
    srez_znack=float(0)
    for i in range(len_n):
        srez_znack+=ishod_data[index][i]
    srez_znack=float(srez_znack/len_n)
    return srez_znack


def save_to_csv_matrix(matrix1,index):
    matrix1.to_csv(f"DATA_MATRIX{index}.csv")


def print_matrix(a):
    max_len = max([len(str(e)) for r in a for e in r])
    for row in a:
        print(*list(map('{{:>{length}}}'.format(length=max_len).format, row)))


def transpose_matrix(matrix1):
    x_size_n = len(matrix1) - 1
    x_size_m = len(matrix1[0])
    matrix_x_transpose = [[0.0] * (x_size_n + 1) for i in range(x_size_m)]
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            matrix_x_transpose[j][i] = float(matrix1[i][j])
    return matrix_x_transpose


def pandas_Pearson_matrix(ishod_data):
    ishod_data=transpose_matrix(ishod_data)
    corr_coef_matrix=pd.DataFrame(ishod_data,columns=['X1','X2','X3','X4','X5','X6','Y'])
    return corr_coef_matrix.corr().astype(float)


def correlation_koeff(X, Y, n) :
    sum_X = 0
    sum_Y = 0
    sum_XY = 0
    squareSum_X = 0
    squareSum_Y = 0
    i = 0
    while i < n :
        sum_X = sum_X + X[i]
        sum_Y = sum_Y + Y[i]
        sum_XY = sum_XY + X[i] * Y[i]
        squareSum_X = squareSum_X + X[i] * X[i]
        squareSum_Y = squareSum_Y + Y[i] * Y[i]
        i = i + 1
    corr = float((n * sum_XY - sum_X * sum_Y))/(float(sqrt((n * squareSum_X - sum_X * sum_X)* (n * squareSum_Y - sum_Y * sum_Y))))
    return corr


def matrix_correlation(ishod_data,n):
    matrix_correlation = [[0.0] * (len(ishod_data)) for i in range(len(ishod_data))]
    for i in range(len(ishod_data)):
        for i1 in range(len(ishod_data)):
            matrix_correlation[i][i1]=correlation_koeff(ishod_data[i],ishod_data[i1],n)
    # matrix_correlation=np.array(matrix_correlation)
    return matrix_correlation


def print_Pearson_corr_matrix(Pearson_correlation_coefficient_matrix,Pearson_correlation_coefficient_matrix1):
    print("Матрица коэффициентов корреляции Пирсона (собственная реализация)")
    print("-"*100)
    print_matrix(Pearson_correlation_coefficient_matrix)
    print("\n Матрица коэффициентов корреляции Пирсона (реализация из библиотеки pandas)")
    print("-"*100)
    print(Pearson_correlation_coefficient_matrix1)


def pairwise_comparison_graph_yellowbrick(ishod_data):
    X=ishod_data[["X1","X2","X3","X4","X5","X6"]]
    Y=ishod_data[["Y"]]
    fig,ax=plt.subplots(figsize=(10,10))
    visualizer = Rank2D(algorithm='pearson')
    visualizer.fit(X, Y)
    visualizer.transform(X)
    print("-"*100)
    print("Save (library yellowbrick) rating_Pearson_next.png in folder")
    fig.savefig(
        "rating_Pearson.png",dpi=300,bbox_inches="tight"
    )


def pairwise_comparison_graph_seaborn(Pearson_correlation_coefficient_matrix1):
    fig,ax=plt.subplots(figsize=(10,10))
    X=Pearson_correlation_coefficient_matrix1[["X1","X2","X3","X4","X5","X6"]]
    Y=Pearson_correlation_coefficient_matrix1[["Y"]]
    ax = heatmap(X.corr(),fmt=".2f",annot=True,ax=ax,cmap="RdBu_r",vmin=-1,vmax=1)
    print("-"*100)
    print("Save (library seaborn) rating_Pearson_next.png in folder")
    fig.savefig(
        "rating_Pearson_next.png",dpi=300,bbox_inches="tight"
    )


def pairwise_comparison_graph(Pearson_correlation_coefficient_matrix1,ishod_data):
    pairwise_comparison_graph_yellowbrick(Pearson_correlation_coefficient_matrix1)
    ishod_data1=transpose_matrix(ishod_data)
    ishod_data1=pd.DataFrame(ishod_data1,columns=['X1','X2','X3','X4','X5','X6','Y'])
    pairwise_comparison_graph_seaborn(ishod_data1)


if __name__=="__main__": 
    ishod_data = []
    with open("information.txt") as f:
        for line in f:
            ishod_data.append([float(x) for x in line.split()])

    n = len(ishod_data[0])
    p = len(ishod_data)
    k = len(ishod_data)-1

    Pearson_correlation_coefficient_matrix=matrix_correlation(ishod_data,n)
    Pearson_correlation_coefficient_matrix1=pandas_Pearson_matrix(ishod_data)

    print_Pearson_corr_matrix(Pearson_correlation_coefficient_matrix,Pearson_correlation_coefficient_matrix1)
    save_to_csv_matrix(Pearson_correlation_coefficient_matrix1,1)
    pairwise_comparison_graph(Pearson_correlation_coefficient_matrix1,ishod_data)

    massiv_average_value=make_massiv_average(ishod_data)
    print_average_value(massiv_average_value)

    massiv_deviation_average=deviation_from_the_average(ishod_data,massiv_average_value)
    print_deviation_from_the_average(massiv_deviation_average)

    massiv_standart_devition=standart_devition(massiv_deviation_average,n)
    print_standart_devition(massiv_standart_devition)

    ishod_data_new=make_standardized(massiv_deviation_average,massiv_standart_devition,1)
    print_standardized(ishod_data_new)

    print(ishod_data_new)
    print_matrix(matrix_correlation(ishod_data_new,n))

    matrix_library_regression(ishod_data_new)
    b_coeff=matrix_regressin(ishod_data_new)
    massiv_prognoz_Y=prognoz_Y(b_coeff,ishod_data_new)
    print_prognoz_Y(massiv_prognoz_Y)

    final(n,p,k,massiv_prognoz_Y,ishod_data_new,massiv_average_value,b_coeff)

    ishod_data_new1=make_normalized(ishod_data,1)

    buff_new=matrix_correlation(ishod_data_new,n)
    Student_t_distribution(buff_new,n,p)
    find_correlation(ishod_data_new,n,buff_new)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# рассчитывание модельных значений после отбора
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------  
    print("-"*100)
    print("рассчитывание модельных значений после отбора")
    
    result_matrix = []
    with open("information.txt") as f:
        for line in f:
            result_matrix.append([float(x) for x in line.split()])
    
    del result_matrix[X_X[0]]
    n_new = len(result_matrix[0])
    p_new = len(result_matrix)
    k_new = len(result_matrix)-1

    Pearson_correlation_coefficient_matrix_new=matrix_correlation(result_matrix,n_new)

    massiv_average_value_new=make_massiv_average(result_matrix)
    print_average_value(massiv_average_value_new)

    massiv_deviation_average_new=deviation_from_the_average(result_matrix,massiv_average_value_new)
    print_deviation_from_the_average(massiv_deviation_average_new)

    massiv_standart_devition_new=standart_devition(massiv_deviation_average_new,n_new)
    print_standart_devition(massiv_standart_devition_new)

    result_matrix_new=make_standardized(massiv_deviation_average_new,massiv_standart_devition_new,2)

    print(result_matrix_new)
    print_matrix(matrix_correlation(result_matrix_new,n_new))

    b_coeff_new=matrix_regressin(result_matrix_new)
    massiv_prognoz_Y_new=prognoz_Y(b_coeff_new,result_matrix_new)
    print_prognoz_Y(massiv_prognoz_Y_new)

    final(n_new,p_new,k_new,massiv_prognoz_Y_new,result_matrix_new,massiv_average_value_new,b_coeff_new)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# рассчитывание модельных значений после отбора и методом KNeighborsRegressor
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------  
    print("-"*100)
    print("Рассчитывание модельных значений после отбора и методом KNeighborsRegressor")
    print("-"*100)

    matrix_neighbors = []
    with open("information.txt") as f:
        for line in f:
            matrix_neighbors.append([float(x) for x in line.split()])
    
    del matrix_neighbors[X_X[0]]
    matrix_neighbors=transpose_matrix(matrix_neighbors)
    matrix_neighbors=pd.DataFrame(matrix_neighbors,columns=[["X1","X2","X4","X5","X6","Y"]])
    df_neighbors = (matrix_neighbors-matrix_neighbors.min ())/ (matrix_neighbors.max () - matrix_neighbors.min ())
    y = matrix_neighbors["Y"]
    X = matrix_neighbors[["X1","X2","X4","X5","X6"]]


    X.describe().T
    print("Переменные X")
    print(X.describe().T)
    print("-"*100)

    y = df_neighbors["Y"]
    X = df_neighbors[["X1","X2","X4","X5","X6"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("-"*100)
    print(f"Длина одной переменной X : {len(X)}")
    print(f"Длина переменной для X_train : {len(X_train)}")
    print(f"Длина переменной для X_test : {len(X_test)}")
    print("-"*100)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    error = []
    for i in range(1, 60):
        knn = KNeighborsRegressor(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        mae = mean_absolute_error(y_test, pred_i)
        error.append(mae)

    n_neighbors=np.array(error).argmin()

    regressor = KNeighborsRegressor(n_neighbors=n_neighbors,algorithm="kd_tree")
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("-"*100)
    print(f'Средняя абсолютная ошибка MAE : {mae}')
    print(f'Среднеквадратичная разница MSE : {mse}')
    print(f'Среднеквадратичная ошибка RMSE : {rmse}')
    sse=mse*(len(y_pred)-2)
    print(f'Ошибка суммы квадратов SSE : {sse*10}')
    sey=sqrt((sse)/(len(y_pred)-2))
    print(f"Стандартная ошибка регрессии SEY : {sey}")
    print(f'Коэффициент детерминации R^2 : {regressor.score(X_test, y_test)}')
    sst = float(((y - y.mean()) ** 2).sum())
    print(f"Общая сумма квадратов SST : {sst*10}")
    print(f"Регрессия суммы квадратов SSR : {(sst-sse)*10}")
    print(f"F-статисттка : {(((sst-sse)/(len(y_pred)-1))/((sst-(sst-sse))/(len(y_pred)-2)))*10}")
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# рассчитывание модельных значений после отбора и методом DecisionTreeRegressor
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------  
    print("-"*100)
    print("Рассчитывание модельных значений после отбора и методом DecisionTreeRegressor")
    print("-"*100)

    matrix_tree = []
    with open("information.txt") as f:
        for line in f:
            matrix_tree.append([float(x) for x in line.split()])
    
    del matrix_tree[X_X[0]]
    matrix_tree=transpose_matrix(matrix_tree)
    matrix_tree=pd.DataFrame(matrix_tree,columns=[["X1","X2","X4","X5","X6","Y"]])
    df_tree = (matrix_tree-matrix_tree.min ())/ (matrix_tree.max () - matrix_tree.min ())
    
    Y1 = matrix_tree["Y"]
    X1 = matrix_tree[["X1","X2","X4","X5","X6"]]
    
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X1_train)

    X1_train = scaler.transform(X1_train)
    X1_test = scaler.transform(X1_test)

    error_new = []
    for i in range(1, 60):
        dtr = DecisionTreeRegressor(random_state=i)
        dtr.fit(X1_train, Y1_train)
        pred_i1 = dtr.predict(X1_test)
        mae_new = mean_absolute_error(Y1_test, pred_i1)
        error_new.append(mae_new)

    n_tree=np.array(error_new).argmin()


    treeRegressor =DecisionTreeRegressor(random_state=n_tree,min_impurity_decrease=0.0, min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0)
    treeRegressor.fit(X1_train, Y1_train)

    y1_pred = treeRegressor.predict(X1_test)

    mae_new = mean_absolute_error(Y1_test, y1_pred)
    mse_new = mean_squared_error(Y1_test, y1_pred)
    rmse_new = mean_squared_error(Y1_test, y1_pred, squared=False)
    print("-"*100)
    print(f'Средняя абсолютная ошибка MAE : {mae_new}')
    print(f'Среднеквадратичная разница MSE : {mse_new}')
    print(f'Среднеквадратичная ошибка RMSE : {rmse_new}')
    sse_new=mse_new*(len(y1_pred)-2)
    print(f'Ошибка суммы квадратов SSE : {sse_new}')
    sey_new=sqrt((sse_new)/(len(y1_pred)-2))
    print(f"Стандартная ошибка регрессии SEY : {sey_new}")
    print(f'Коэффициент детерминации R^2 : {treeRegressor.score(X1_test, Y1_test)}')
    sst_new = float(((Y1 - Y1.mean()) ** 2).sum())
    print(f"Общая сумма квадратов SST : {sst_new}")
    print(f"Регрессия суммы квадратов SSR : {(sst_new-sse_new)}")
    print(f"F-статисттка : {(((sst_new-sse_new)/(len(y1_pred)-1))/((sst_new-(sst_new-sse_new))/(len(y1_pred)-2)))*10}")
    
    make_graphic_ost(result_matrix,massiv_prognoz_Y_new)

