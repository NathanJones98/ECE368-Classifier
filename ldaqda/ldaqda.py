import numpy as np
import matplotlib.pyplot as plt
import util

def LDA_analysis(data):

    plt.close()

    cov = data[0]
    male_height_data = data[1]
    male_weight_data = data[2]
    female_height_data = data[3]
    female_weight_data = data[4]
    all_height_data = data[5]
    all_weight_data = data[6]
    mu_male = data[7]
    mu_female = data[8]

    plt.figure(figsize=(8, 6))
    plt.scatter(male_height_data, male_weight_data, color='blue', label='male')
    plt.scatter(female_height_data, female_weight_data, color='red', label='female')

    #inverse covariance, the difference between the mean vectors of the two classes.
    lda_coeff = np.dot(np.linalg.inv(cov), (mu_male - mu_female))

    #Constant Term
    lda_bias = 0.5 * (np.dot(mu_female, np.dot(np.linalg.inv(cov), mu_female)) 
    - np.dot(mu_male, np.dot(np.linalg.inv(cov), mu_male)))

    #Input / Output Values for LDA Line
    lda_x_vals = np.array([min(all_height_data), max(all_height_data)])
    lda_y_vals = (-lda_bias - lda_coeff[0] * lda_x_vals) / lda_coeff[1]
    
    plt.plot(lda_x_vals, lda_y_vals, color='black', label='LDA Boundry')
    plt.legend()
    plt.title('LDA Model')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    #plt.show()

    plt.savefig("lda.pdf")

def QDA_analysis(data):
    
    plt.close()

    cov = data[0]
    male_height_data = data[1]
    male_weight_data = data[2]
    female_height_data = data[3]
    female_weight_data = data[4]
    all_height_data = data[5]
    all_weight_data = data[6]
    mu_male = data[7]
    mu_female = data[8]
    cov_male = data[9]
    cov_female = data[10]
    
    plt.figure(figsize=(8, 6))
    male = plt.scatter(male_height_data, male_weight_data, color='blue', label='male')
    female = plt.scatter(female_height_data, female_weight_data, color='red', label='female')

    inv_cov_male = np.linalg.inv(cov_male)
    inv_cov_female = np.linalg.inv(cov_female)
    qda_coeff_male = inv_cov_male
    qda_coeff_female = inv_cov_female
    
    #coefficient matrix 
    qda_a = 0.5 * (qda_coeff_male - qda_coeff_female)

    # coefficient vector
    qda_b = np.dot(mu_female, qda_coeff_female) - np.dot(mu_male, qda_coeff_male)

    #Constant Term
    qda_c = (0.5 * (np.dot(mu_male, np.dot(inv_cov_male, mu_male.T)) 
                    - np.dot(mu_female, np.dot(inv_cov_female, mu_female.T)))
             - 0.5 * np.log(np.linalg.det(cov_male) / np.linalg.det(cov_female)))

    #Plotting Values
    qda_x_vals = np.linspace(min(all_height_data), max(all_height_data), 100)
    qda_y_vals = np.linspace(min(all_weight_data), max(all_weight_data), 100)
    qda_x, qda_y = np.meshgrid(qda_x_vals, qda_y_vals)
    qda_z = qda_a[0, 0] * qda_x**2 + (qda_a[0, 1] + qda_a[1, 0]) * qda_x * qda_y + qda_a[1, 1] * qda_y**2 + qda_b[0] * qda_x + qda_b[1] * qda_y + qda_c
    
    contour = plt.contour(qda_x, qda_y, qda_z, levels=[0], colors='black', label='QDA Boundry', linestyles='dashed')
    #plt.legend(handles=[male,female, contour.levels], labels=['Male', 'Female', 'QDA Boundry'])
    plt.legend()
    plt.title('QDA Model')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    #plt.show()

    plt.savefig("qda.pdf")

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
   
    #separate male and female samples
    male_samples = x[y==1]
    female_samples = x[y==2]

    #calculate the mean vectors of male and female samples
    mu_male = np.mean(male_samples, axis=0)
    mu_female = np.mean(female_samples, axis=0)

    #calculate the covariance matrix of all samples
    cov = np.cov(x.T)

    #calculate the covariance matrices of male and female samples
    cov_male = np.cov(male_samples.T)
    cov_female = np.cov(female_samples.T)

    male_height_data = male_samples[:,0]
    male_weight_data = male_samples[:,1]

    female_height_data = female_samples[:,0]
    female_weight_data = female_samples[:,1]

    all_height_data = x[:, 0]
    all_weight_data = x[:, 1]

    data = [cov, 
    male_height_data, 
    male_weight_data, 
    female_height_data, 
    female_weight_data, 
    all_height_data, 
    all_weight_data,
    mu_male,
    mu_female,
    cov_male,
    cov_female]

    # plot LDA
    LDA_analysis(data)

    # plot QDA
    QDA_analysis(data)

    return (mu_male,mu_female,cov,cov_male,cov_female)

def misRate(mu_male, mu_female, cov, cov_male, cov_female, x, y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here

    #Compute the densities of male and female groups
    male_densities = util.density_Gaussian(mu_male, cov, x)
    female_densities = util.density_Gaussian(mu_female, cov, x)

    #Compute the probabilities of male and female groups
    lda_male_probs = male_densities / (male_densities + female_densities)
    lda_female_probs = 1 - lda_male_probs

    #Compute the predicted labels LDA
    lda_pred_labels = 1 + (lda_female_probs > lda_male_probs)

    #Compute the misclassification rate for LDA
    mis_lda = np.mean(lda_pred_labels != y)

    #Compute the probabilities of male and female groups
    qda_male_probs = util.density_Gaussian(mu_male, cov_male, x)
    qda_female_probs = util.density_Gaussian(mu_female, cov_female, x)

    #Compute the predicted labels QDA
    qda_pred_labels = 1 + (qda_female_probs > qda_male_probs)

    #Compute the misclassification rate for QDA
    mis_qda = np.mean(qda_pred_labels != y)

    return (mis_lda, mis_qda)

if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)

    print(mis_LDA,mis_QDA)