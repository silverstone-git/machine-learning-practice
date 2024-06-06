import pandas as pd
import matops


def lin_reg(design_matrix, target_variable):

    # 1s column to allow for constant term (theta_0, ie, intercept) in params
    design_matrix.insert(0, len(design_matrix[0]) * [1])
    no_of_features = len(design_matrix)
    learning_rate = 1e-10

    params = [list([0.5 for x in range(no_of_features)])]


    def hypothesis(theta, x):
        theta_t = matops.transpose(theta)
        hypo = matops.coremult(theta_t, x)[0][0]
        return hypo


    for rep in range(1000):
        sigma_error_x = [(no_of_features) * [0]]
        for sample_index in range(len(design_matrix[0])):
            x_for_sample = [list([x[sample_index] for x in design_matrix])]

            error = target_variable[sample_index] - hypothesis(params, x_for_sample)
            #print("error is => ", error)

            lr_step = matops.matscalmult(error, x_for_sample)
            sigma_error_x = matops.addmat(sigma_error_x, lr_step)

        learning = matops.matscalmult(learning_rate, sigma_error_x)
        print("learning is => ", learning)
        params = matops.addmat(params, learning)
        print("batch done, new params => ", params)

    return params


def main():

    df = pd.read_csv('melb_data.csv')
    no_of_samples = 1000
    lin_reg_data1 = df.loc[0:no_of_samples, ['Landsize', 'Rooms', 'Price']][(df['Landsize'] != 0) & (df['Price'] != 0)]
    lin_reg_data1 = lin_reg_data1[lin_reg_data1['Landsize'] < 1000]

    landsize = lin_reg_data1['Landsize'].to_list()
    price = lin_reg_data1['Price'].to_list()
    rooms = lin_reg_data1['Rooms'].to_list()

    params_final = lin_reg([landsize, rooms], price)

    print(params_final)




if __name__ == '__main__':
    main()




