import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report, roc_curve, roc_auc_score
from DT_Classifier import DecisionTree

if __name__ == '__main__':
    d = pd.read_csv("train.csv")[['Age', 'Fare', 'Survived']].dropna()

    # Constructing the X and Y matrices
    X = d[['Age', 'Fare']]
    Y = d['Survived']


    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)


    # root = DecisionTree(X, Y, max_depth=3, min_samples_split=100, criterion='entropy')
    root = DecisionTree(X_train, Y_train, max_depth=5, min_samples_split=100, split='mid',criterion='entropy')
    root._build_tree()
    root.print_tree()
    X_temp = X_test.copy()
    X_temp['predicted_y']= root._predict(X_temp)
    print(X_temp)

    conf_mat = confusion_matrix(Y_test, X_temp['predicted_y'])
    print("conf_mat is ",conf_mat)

    print(classification_report(Y_test, X_temp['predicted_y']))

    # dt1 = DecisionTree(X,Y)
    # dt1._get_best_split()

    #X['Fare'][5]
    # root = DecisionTree(X, Y, max_depth=3, min_samples_split=100, criterion='entropy')
    # root._build_tree()
    # root.print_tree()
    # X_temp = X.copy()
    # X_temp['predicted_y']= root._predict(X)
    # print(X_temp)
    #
    # conf_mat = confusion_matrix(Y, X_temp['predicted_y'])
    # print("conf_mat is ",conf_mat)
    #
    # true_positive = conf_mat[0][0]
    # false_positive = conf_mat[0][1]
    # false_negative = conf_mat[1][0]
    # true_negative = conf_mat[1][1]
    #
    # # Accuracy
    # Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
    # Accuracy  # 0.98
    #
    # # Precison
    # Precision = true_positive / (true_positive + false_positive)
    # Precision  # 0.97
    #
    # # Recall
    # Recall = true_positive / (true_positive + false_negative)
    # Recall  # 1.0
    #
    # # F1 Score
    # F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
    # F1_Score  # 0.98
    #
    # # Area Under Curve
    # auc = roc_auc_score(Y, X_temp['predicted_y'])
    # auc  # 0.98
    #
    # print("Accuracy , Precision, Recall, F1_Score, auc",Accuracy , Precision, Recall, F1_Score, auc)

# if __name__ == '__main__':
#     col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
#     data = pd.read_csv("iris.data.csv", skiprows=1, header=None, names=col_names)
#     data.head(10)
#
#     # X = data.iloc[:, :-1].values
#     # Y = data.iloc[:, -1].values.reshape(-1, 1)
#     X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
#     Y = data['type']
#
#     from sklearn.model_selection import train_test_split
#
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
#
#
#     # root = DecisionTree(X, Y, max_depth=3, min_samples_split=100, criterion='entropy')
#     root = DecisionTree(X_train, Y_train, max_depth=5, min_samples_split=3, split='unique',criterion='entropy')
#     root._build_tree()
#     root.print_tree()
#     X_temp = X_test.copy()
#     X_temp['predicted_y']= root._predict(X_temp)
#     print(X_temp)
#
#     conf_mat = confusion_matrix(Y_test, X_temp['predicted_y'])
#     print("conf_mat is ",conf_mat)
#
#     print(classification_report(Y_test, X_temp['predicted_y']))
#
#     # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
#     cm_df = pd.DataFrame(conf_mat,
#                          index=['SETOSA', 'VERSICOLR', 'VIRGINICA'],
#                          columns=['SETOSA', 'VERSICOLR', 'VIRGINICA'])
#
#     # Plotting the confusion matrix
#     plt.figure(figsize=(5, 4))
#     sns.heatmap(cm_df, annot=True)
#     plt.title('Confusion Matrix')
#     plt.ylabel('Actal Values')
#     plt.xlabel('Predicted Values')
#     plt.show()
#
#     true_positive = conf_mat[0][0]
#     false_positive = conf_mat[1][0] + conf_mat[2][0]
#     false_negative = conf_mat[0][1] + conf_mat[0][2]
#     true_negative = conf_mat[1][1] + conf_mat[1][2] + conf_mat[2][1] + conf_mat[2][2]
#
#     # Accuracy
#     Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
#     Accuracy
#
#     # Precison
#     Precision = true_positive / (true_positive + false_positive)
#     Precision
#
#     # Recall
#     Recall = true_positive / (true_positive + false_negative)
#     Recall
#
#     # F1 Score
#     F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
#     F1_Score
#
#     #Area Under Curve
#     # auc = roc_auc_score(Y, X_temp['predicted_y'])
#     # auc  # 0.98
#
#     print("Accuracy , Precision, Recall, F1_Score ",Accuracy , Precision, Recall, F1_Score)

